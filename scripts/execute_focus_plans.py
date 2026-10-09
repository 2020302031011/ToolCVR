#!/usr/bin/env python3
"""Stage 4: run a round of focus plans against the real videos.

This is the other half of the planning pair.  ``plan_focus.py`` writes plans and
never touches a pixel; this script is the only place in the repository that
turns a planned window into frames.  It decodes video and, unless
``--mock-reporter`` is used, calls a visual model.

One look is one ``FocusTool`` call on one video.  A round is a *sequence* of
those calls, never a single request holding frames from several videos, which
is why this loop is written per look and why nothing here concatenates two
results into one report.  The round's evidence leaves as a bundle; merging it
belongs to the state update, and a merged report text would be a claim about
several videos at once -- exactly what ``AGENTS.md`` forbids one visual-model
request from making.

The output is evidence, not an answer.  No state delta is generated, no review
is performed, and nothing is written back into the initial state.

``TASK/SAMPLE/focus_evidence.json`` is written per sample; ``--resume`` reuses
one only when the plan it came from and the reporter identity both still match.

Provenance is checked, not assumed, in both directions: the plan must have been
planned from the state and the report this run reads, and the evidence records
the digest of every input it consumed so that its own consumer can check it the
same way.  See :mod:`toolcvr.provenance`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toolcvr.agents import ObserverAgent  # noqa: E402
from toolcvr.crossvid import CrossVidAuditAdapter  # noqa: E402
from toolcvr.focus import FocusConfig, FocusRequest, FocusTool  # noqa: E402
from toolcvr.focus_execution import execute_focus_plan  # noqa: E402
from toolcvr.focus_planner import FocusLook, FocusPlan  # noqa: E402
from toolcvr.provenance import (  # noqa: E402
    MOCK_REPORTER_IDENTITY,
    ProvenanceError,
    file_digest,
    local_qwen3vl_identity_suffix,
    reuse_decision,
    verify_recorded_input,
)
from toolcvr.report_generation import atomic_write_result  # noqa: E402
from toolcvr.reporting import (  # noqa: E402
    DeterministicMockEventReporter,
    LocalQwen3VLReporter,
)

# The mock reporter's name is the same string the warm observation writes, so
# that a focus round can be checked for having looked through the same eyes as
# the report whose events it was planned from.  Both are read back into the
# output and compared rather than assumed -- which is why the name is imported
# rather than repeated here; the two files that had a copy each had already
# stopped agreeing about the model's identity.
DEFAULT_CROSSVID_ROOT = Path("/media/data6/xuejj/CrossVid")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        required=True,
        help="root containing TASK/SAMPLE/focus_plan.json, written by plan_focus.py",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="root containing TASK/SAMPLE/report.json, for the question and decode settings",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crossvid-root", type=Path, default=DEFAULT_CROSSVID_ROOT)
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "crossvid-audit-proxies",
        help="where materialized UAV views are cached; must match the selection run",
    )
    parser.add_argument("--tasks", help="comma-separated task filter, e.g. PSS,CC,FSA")
    parser.add_argument("--sample-ids", help="comma-separated exact ids, e.g. CC-474")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-builder-downgrade",
        action="store_true",
        help=(
            "let --resume overwrite evidence a visual model produced with "
            "evidence this run produces synthetically; without it such a "
            "resume stops, because a mock bundle that replaced a model's is "
            "indistinguishable afterwards from one no model ever touched"
        ),
    )
    parser.add_argument(
        "--mock-reporter",
        action="store_true",
        help="deterministic CPU reporter; exercises the path without a visual model",
    )
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--max-vlm-tokens", type=int)
    parser.add_argument("--maximum-events-per-batch", type=int)
    parser.add_argument(
        "--decode-long-edge",
        type=int,
        help="decoded frame size; defaults to the value the report was generated with",
    )
    parser.add_argument(
        "--overlay-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "burn each frame's timestamp into the image, as the observation does; "
            "defaults to the observation's setting, and is named in reporter_identity "
            "so a run that differs from the report it executes says so on disk"
        ),
    )
    parser.add_argument("--frames-per-call", type=int, default=16)
    return parser


def _reporter_identity_suffix(
    *,
    model: Path,
    adapter: Path | None,
    maximum_new_tokens: int,
    maximum_events: int,
    overlay_timestamps: bool,
) -> str:
    """How this run's reporter is named, by the one recipe both stages share.

    Identical to the observation's -- same recipe, same parameters -- when this
    run uses the same model, so the ``reporter_identity`` this writes into a
    bundle can be read against the ``observation_reporter_identity`` written
    beside it.  Two copies of this string-building had already drifted: they
    disagreed about the model digest and one of them named no overlay setting
    at all.  See :func:`toolcvr.provenance.local_qwen3vl_identity_suffix`.
    """
    return local_qwen3vl_identity_suffix(
        model=model,
        adapter=adapter,
        maximum_new_tokens=maximum_new_tokens,
        maximum_events=maximum_events,
        overlay_timestamps=overlay_timestamps,
    )


def _reporter(args: argparse.Namespace):
    """The same two reporters the warm observation uses, built the same way."""
    if args.mock_reporter:
        if args.vlm_model is not None or args.vlm_adapter is not None:
            raise ValueError("--mock-reporter cannot be combined with VLM model/adapter paths")
        return DeterministicMockEventReporter(), MOCK_REPORTER_IDENTITY
    if args.vlm_model is None:
        raise ValueError("--vlm-model is required unless --mock-reporter is used")
    model = args.vlm_model.expanduser().resolve()
    if not model.exists():
        raise FileNotFoundError(f"VLM model does not exist: {model}")
    adapter: Path | None = None
    if args.vlm_adapter is not None:
        adapter = args.vlm_adapter.expanduser().resolve()
        if not adapter.exists():
            raise FileNotFoundError(f"VLM adapter does not exist: {adapter}")
    maximum_tokens = 2048 if args.max_vlm_tokens is None else args.max_vlm_tokens
    maximum_events = 8 if args.maximum_events_per_batch is None else args.maximum_events_per_batch
    if maximum_tokens <= 0 or maximum_events <= 0:
        raise ValueError("VLM token/event limits must be positive")
    suffix = _reporter_identity_suffix(
        model=model,
        adapter=adapter,
        maximum_new_tokens=maximum_tokens,
        maximum_events=maximum_events,
        overlay_timestamps=args.overlay_timestamps,
    )
    return (
        LocalQwen3VLReporter(
            str(model),
            device=args.vlm_device,
            adapter_path=str(adapter) if adapter is not None else None,
            maximum_new_tokens=maximum_tokens,
            maximum_events=maximum_events,
            overlay_timestamps=args.overlay_timestamps,
        ),
        f"local-qwen3vl-v6-frame-manifest:{suffix}",
    )


def _discover(root: Path, tasks: set[str] | None, sample_ids: set[str] | None) -> list[Path]:
    paths = sorted(root.glob("*/*/focus_plan.json"))
    selected = [
        path
        for path in paths
        if (tasks is None or path.parent.parent.name.upper() in tasks)
        and (sample_ids is None or path.parent.name in sample_ids)
    ]
    if sample_ids is not None:
        missing = sorted(sample_ids - {path.parent.name for path in selected})
        if missing:
            raise FileNotFoundError(f"focus plans not found for: {missing}")
    return selected


def _load_rows(root: Path, task: str) -> dict[str, tuple[int, dict[str, object]]]:
    path = root / "QA" / f"{task}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError(f"CrossVid QA file must contain an object list: {path}")
    rows: dict[str, tuple[int, dict[str, object]]] = {}
    for index, row in enumerate(raw):
        sample_id = f"{task}-{row.get('id', index)}"
        if sample_id in rows:
            raise ValueError(f"duplicate CrossVid sample ID: {sample_id}")
        rows[sample_id] = (index, row)
    return rows


# The look shape this executor knows how to read.  A plan file names the
# planner that wrote it, so a file missing one of these was written before the
# current shape and cannot be executed as if the field had been there: a
# defaulted ``window_clipped_to_medium`` would put a claim in the evidence that
# the plan never made.  Refused by name, before a frame is decoded.
_REQUIRED_LOOK_FIELDS = (
    "video_id",
    "time_window",
    "anchor_event_ids",
    "anchor_source",
    "dropped_anchor_event_ids",
    "window_budget_exceeded",
    "window_clipped_to_medium",
)


def _rebuild_plan(payload: dict[str, Any], *, source: Path) -> FocusPlan:
    """Rebuild the plan from disk, exactly as the executor receives it.

    Reading the JSON back rather than holding the in-memory object is the
    point: what is executed is what was written down, so a plan that cannot be
    turned back into a plan fails here instead of at review time.
    """
    plan_payload = payload["plan"]
    for look in plan_payload["looks"]:
        missing = [field for field in _REQUIRED_LOOK_FIELDS if field not in look]
        if missing:
            written_by = payload.get("run", {}).get("planner_identity")
            raise ValueError(
                f"{source} was written by {written_by!r} and its looks have no "
                f"{missing}; this executor reads the shape the current planner "
                f"writes -- replan it with scripts/plan_focus.py"
            )
    looks = tuple(
        FocusLook(
            request=FocusRequest(
                video_id=look["video_id"],
                evidence_need=plan_payload["evidence_need"],
                time_span=tuple(look["time_window"]),
            ),
            anchor_event_ids=tuple(look["anchor_event_ids"]),
            anchor_source=look["anchor_source"],
            dropped_anchor_event_ids=tuple(look["dropped_anchor_event_ids"]),
            window_budget_exceeded=look["window_budget_exceeded"],
            window_clipped_to_medium=look["window_clipped_to_medium"],
        )
        for look in plan_payload["looks"]
    )
    return FocusPlan(
        issue_id=plan_payload["issue_id"],
        issue_class=plan_payload["issue_class"],
        evidence_need=plan_payload["evidence_need"],
        looks=looks,
        bundle_videos=tuple(plan_payload["bundle_videos"]),
        unanchorable_videos=tuple(plan_payload["unanchorable_videos"]),
        evidence_need_source=plan_payload["evidence_need_source"],
        evidence_need_note=plan_payload["evidence_need_note"],
        request_target_videos=tuple(plan_payload["request_target_videos"]),
    )


def _reuse_decision(
    run: dict[str, Any],
    plan_digest: str,
    reporter_identity: str,
    *,
    destination: Path,
    allow_builder_downgrade: bool,
) -> str | None:
    """Whether completed evidence on disk may be reused, and if not, why not.

    The rule itself lives in :mod:`toolcvr.provenance`, shared with the other
    direction of the same question -- an artifact about to be *used* rather
    than reused (see :func:`_verify_plan_inputs`).  Reuse is decided by the
    producer as well as by the input: evidence produced by a different reporter
    is different evidence, however unchanged the plan is.

    This is the stage the mock/model distinction is *normal* in -- the corpus
    runs a mock reporter against reports a visual model wrote -- so the
    refusal only fires when the artifact on disk was made by the model.
    """
    return reuse_decision(
        run,
        subject="evidence",
        identities={"reporter_identity": reporter_identity},
        inputs={"plan_sha256": ("the plan", plan_digest)},
        destination=destination,
        allow_downgrade=allow_builder_downgrade,
    )


def _verify_plan_inputs(
    plan_run: dict[str, Any], *, plan_path: Path, report_path: Path
) -> None:
    """The plan in hand must have been planned from the files in hand.

    A plan names the state it was planned from and the report that state came
    from.  Executing its windows while reading another observation's report
    would produce a bundle whose every field is well formed and whose
    provenance is a lie, so both are checked -- the report against the file
    this run reads for the question, the state against the file it names.

    Provenance is checked before the per-sample ``try``: it is a property of
    how the run was invoked, not of the sample, so it is one wiring mistake to
    fix rather than an error line per sample to read.
    """
    verify_recorded_input(plan_run, prefix="report", path=report_path)
    state_source = plan_run.get("state_source")
    if not isinstance(state_source, str) or not state_source:
        raise ProvenanceError(
            f"{plan_path} records no state_source, so the state its anchors were "
            f"inherited from cannot be checked"
        )
    verify_recorded_input(plan_run, prefix="state", path=Path(state_source))


def main() -> int:
    args = _parser().parse_args()
    plan_root = args.plan_dir.expanduser().resolve()
    reports_root = args.reports_dir.expanduser().resolve()
    tasks = {item.strip().upper() for item in args.tasks.split(",")} if args.tasks else None
    sample_ids = {item.strip() for item in args.sample_ids.split(",")} if args.sample_ids else None
    plan_paths = _discover(plan_root, tasks, sample_ids)
    if args.limit is not None:
        plan_paths = plan_paths[: args.limit]
    if not plan_paths:
        raise RuntimeError("no focus plans matched the requested filters")

    reporter, reporter_identity = _reporter(args)
    adapter = CrossVidAuditAdapter(
        args.crossvid_root.expanduser().resolve(),
        proxy_directory=args.proxy_dir.expanduser().resolve(),
    )
    print(
        f"[focus] {len(plan_paths)} plans reporter={reporter_identity}",
        flush=True,
    )
    rows_by_task: dict[str, dict[str, tuple[int, dict[str, object]]]] = {}
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    executed = skipped = 0
    totals = {
        "looks": 0,
        "frames": 0,
        "events": 0,
        "gaps": 0,
        "partial": 0,
        "clipped": 0,
    }
    for position, plan_path in enumerate(plan_paths, start=1):
        task, sample_id = plan_path.parent.parent.name, plan_path.parent.name
        report_path = reports_root / task / sample_id / "report.json"
        if not report_path.is_file():
            print(f"[{position}/{len(plan_paths)}][{sample_id}] SKIP no report", flush=True)
            continue
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan_payload.get("plan") is None:
            print(f"[{position}/{len(plan_paths)}][{sample_id}] SKIP no plan in {plan_path}", flush=True)
            continue

        # Checked before the resume shortcut, so that a skip cannot carry a
        # plan/report pairing that was never verified; the report digest it
        # computes is recorded below either way.
        _verify_plan_inputs(
            plan_payload.get("run", {}), plan_path=plan_path, report_path=report_path
        )
        report_digest = file_digest(report_path)

        final_path = output_root / task / sample_id / "focus_evidence.json"
        plan_digest = file_digest(plan_path)
        if args.resume and final_path.is_file():
            existing = json.loads(final_path.read_text(encoding="utf-8"))
            stale = _reuse_decision(
                existing.get("run", {}),
                plan_digest,
                reporter_identity,
                destination=final_path,
                allow_builder_downgrade=args.allow_builder_downgrade,
            )
            if stale is None:
                skipped += 1
                print(f"[{position}/{len(plan_paths)}][{sample_id}] resume skip complete", flush=True)
                continue
            print(f"[{position}/{len(plan_paths)}][{sample_id}] regenerating: {stale}", flush=True)

        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        question = str(report_payload.get("question", "")).strip()
        if not question:
            raise ValueError(f"report has no question: {report_path}")
        decode_long_edge = (
            args.decode_long_edge
            if args.decode_long_edge is not None
            else int(report_payload["run"]["decode_long_edge"])
        )

        rows = rows_by_task.setdefault(task, _load_rows(args.crossvid_root.expanduser().resolve(), task))
        if sample_id not in rows:
            raise KeyError(f"CrossVid source row not found for {sample_id}")
        source_index, row = rows[sample_id]
        sample = adapter.normalize(task, row, source_index)

        plan = _rebuild_plan(plan_payload, source=plan_path)
        try:
            bundle = execute_focus_plan(
                plan=plan,
                videos=sample.videos,
                tool=FocusTool(
                    observer=ObserverAgent(reporter),
                    config=FocusConfig(
                        frames_per_call=args.frames_per_call,
                        decode_long_edge=decode_long_edge,
                    ),
                ),
                question=question,
            )
        except Exception as exc:  # one bad sample must not end the batch
            print(
                f"[{position}/{len(plan_paths)}][{sample_id}] ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        serialized = bundle.to_dict()
        serialized["run"] = {
            "sample_id": sample_id,
            "task": task,
            "plan_source": str(plan_path),
            "plan_sha256": plan_digest,
            "plan_identity": plan_payload["run"].get("planner_identity"),
            "report_source": str(report_path),
            "report_sha256": report_digest,
            "reporter_identity": reporter_identity,
            "observation_reporter_identity": report_payload["run"].get("reporter_identity"),
            "decode_long_edge": decode_long_edge,
            "frames_per_call": args.frames_per_call,
            "overlay_timestamps": args.overlay_timestamps,
            "stages_executed": ["focus_execution"],
            "merged_into_state": False,
        }
        atomic_write_result(final_path, serialized)

        executed += 1
        totals["looks"] += len(bundle.results)
        totals["frames"] += sum(len(result.frame_manifest) for result in bundle.results)
        totals["events"] += len(bundle.events)
        totals["gaps"] += sum(
            len(result.observation.observation_gaps) for result in bundle.results
        )
        totals["partial"] += int(bundle.partial)
        totals["clipped"] += sum(
            1 for look in plan.looks if look.window_clipped_to_medium
        )
        per_look = " ".join(
            f"{result.video_id}[{len(result.frame_manifest)}f/"
            f"{len(result.observation.events)}e/"
            f"{len(result.observation.observation_gaps)}g]"
            for result in bundle.results
        )
        print(
            f"[{position}/{len(plan_paths)}][{sample_id}] "
            f"issue={bundle.issue_id} looks={len(bundle.results)} "
            f"partial={bundle.partial} {per_look} -> {final_path}",
            flush=True,
        )
        if bundle.repeated_event_ids:
            print(
                f"    repeated event ids across looks: {list(bundle.repeated_event_ids)}; "
                "the state's event-id keyed merge must namespace these by video first",
                flush=True,
            )

    print(
        f"[done] executed={executed} skipped={skipped} looks={totals['looks']} "
        f"frames={totals['frames']} events={totals['events']} gaps={totals['gaps']} "
        f"partial={totals['partial']} clipped={totals['clipped']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
