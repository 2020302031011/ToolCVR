#!/usr/bin/env python3
"""Turn a built initial state into one round of per-video focus looks.

This is stage 3, the minimal version: pick one unresolved issue, derive one
time window per video it covers mechanically, and let the model write exactly
one sentence for the whole round.  It consumes ``initial_state.json`` plus the
``report.json`` the state was built from -- the reports are needed because the
state records which events were cited, never when they happened.

It never decodes video and never calls a visual model.  The only optional model
call is the ``evidence_need`` sentence, and ``--mock`` removes even that.

The output is a plan, not an answer, and a plan is a bundle: one look per video
the issue covers, each with its own inherited window, because a visual-model
request may never mix videos.  ``refusals`` lists every issue that was
considered and could not be planned, so a planner that can only serve part of
the state cannot be mistaken for one that had nothing to do.  ``deferred``
names the rest, so the ledger balances: considered == refusals + deferred +
the winner.

The sentence itself normally comes from ``candidate_requests[].observable``,
which state formation already wrote and validated.  ``--mock`` therefore
removes the model call entirely on most samples; it is the fallback path, not
the main one.

Every window is a window of its video: it contains the anchors it names and it
lies inside the length the report records, in that order of priority -- where
the configuration asks for more padding than the file has, the file wins and
the plan says so, and where an *anchor* reaches past the file the report is
corrupt and the sample is refused rather than clipped around.  See
:func:`toolcvr.focus_planner._place_window` for the contract.

Every plan records the digest of the state it was planned from, of the report
that state came from, and of the rules that produced it; the state's own record
of its report is checked before this run uses either.  See
:mod:`toolcvr.provenance`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toolcvr.focus_planner import (  # noqa: E402
    DeterministicMockEvidenceNeed,
    FocusPlannerConfig,
    OpenAICompatibleEvidenceNeed,
    plan_focus,
)
from toolcvr.initial_state import normalize_initial_state  # noqa: E402
from toolcvr.provenance import (  # noqa: E402
    file_digest,
    reuse_decision,
    verify_recorded_input,
)
from toolcvr.report_generation import atomic_write_result  # noqa: E402
from toolcvr.schemas import EventRecord, VideoEvidenceReport, WorkingState  # noqa: E402

# Bump whenever the planning rules change in a way that alters the output.
# It is written into every plan and checked on resume, so a code change forces
# recomputation instead of silently serving plans the new rules would not make.
PLANNER_IDENTITY = "focus-planner-v5-window-placement-contract"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="root containing TASK/SAMPLE/initial_state.json",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="root containing TASK/SAMPLE/report.json, used for the event catalogue",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", help="comma-separated task filter, e.g. PSS,CC,FSA")
    parser.add_argument("--sample-ids", help="comma-separated exact ids, e.g. CC-117")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-builder-downgrade",
        action="store_true",
        help=(
            "let --resume overwrite plans a model planned with plans this run "
            "renders synthetically; without it such a resume stops, because a "
            "mock corpus that replaced a model's is indistinguishable "
            "afterwards from one no model ever touched"
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="render evidence_need deterministically instead of calling a model",
    )
    parser.add_argument("--api-env-file", type=Path)
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--api-maximum-tokens", type=int, default=256)
    parser.add_argument("--margin-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-span-seconds", type=float, default=2.0)
    parser.add_argument(
        "--maximum-span-seconds",
        type=float,
        default=60.0,
        help=(
            "budget for the unpadded union of a look's anchors, not a ceiling on "
            "the window: a single observation longer than this widens the window "
            "and is recorded as window_budget_exceeded on that look"
        ),
    )
    return parser


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"API environment file does not exist: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid dotenv line {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid dotenv key at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _builder(args: argparse.Namespace):
    if args.mock:
        return DeterministicMockEvidenceNeed(), "deterministic-mock"
    base_url = args.api_base_url or os.environ.get("LLM_BASE_URL", "")
    model = args.api_model or os.environ.get("LLM_MODEL", "")
    if not base_url or not model:
        raise RuntimeError(
            "focus planner requires --mock, or LLM_BASE_URL/LLM_MODEL from the "
            "environment or --api-base-url/--api-model"
        )
    return (
        OpenAICompatibleEvidenceNeed(
            base_url=base_url,
            model=model,
            api_key_env=args.api_key_env,
            timeout_seconds=args.api_timeout_seconds,
            maximum_tokens=args.api_maximum_tokens,
        ),
        f"openai-compatible:{model}",
    )


def _discover(
    root: Path,
    tasks: set[str] | None,
    sample_ids: set[str] | None,
) -> list[Path]:
    paths = sorted(root.glob("*/*/initial_state.json"))
    selected = [
        path
        for path in paths
        if (tasks is None or path.parent.parent.name.upper() in tasks)
        and (sample_ids is None or path.parent.name in sample_ids)
    ]
    if sample_ids is not None:
        missing = sorted(sample_ids - {path.parent.name for path in selected})
        if missing:
            raise FileNotFoundError(f"initial states not found for: {missing}")
    return selected


def _verify_report_inputs(state_payload: dict[str, Any], *, report_path: Path) -> None:
    """The report in hand must be the one the state in hand was built from.

    Planning reads both at once -- events and durations from the report, issues
    and citations from the state -- so a state paired with another
    observation's report yields a plan whose windows cite events the question
    never mentioned, every field of it well formed.  The state records which
    report that was, so the pairing is checked rather than assumed.

    Checked before the per-sample ``try``, and before the resume shortcut: a
    provenance mismatch is a property of how the run was invoked, not of the
    sample, so it is one wiring mistake to fix rather than an error line per
    sample to read -- and a skip must not carry an unverified pair forward.
    """
    verify_recorded_input(state_payload.get("run", {}), prefix="report", path=report_path)


def _reuse_decision(
    run: dict[str, Any],
    state_digest: str,
    builder_identity: str,
    *,
    destination: Path,
    allow_builder_downgrade: bool,
) -> str | None:
    """Whether a completed plan on disk may be reused, and if not, why not.

    The rule itself lives in :mod:`toolcvr.provenance`, next to the check for
    the other direction -- an artifact about to be *used* rather than reused.
    Both are keyed on the inputs and on the producer: a plan is a function of
    the state *and* of these planning rules, so a change to the rules leaves
    every digest intact while every plan on disk becomes wrong.

    Two producers, and both are named here.  ``PLANNER_IDENTITY`` is the rules
    that placed the windows; ``builder_identity`` is the model that wrote the
    ``evidence_need`` sentence, and it is the one that says whether a model was
    involved at all.  Keying on the rules alone would let a mock ``--resume``
    overwrite plans a model planned while every recorded field agreed -- the
    mock and the model spell the rules identically.
    """
    return reuse_decision(
        run,
        subject="plan",
        identities={
            "planner_identity": PLANNER_IDENTITY,
            "builder_identity": builder_identity,
        },
        inputs={"state_sha256": ("the initial state", state_digest)},
        destination=destination,
        allow_downgrade=allow_builder_downgrade,
    )


def _reports_from_payload(payload: dict[str, Any], path: Path) -> tuple[str, list[VideoEvidenceReport]]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError(f"report has no question: {path}")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"report has no videos list: {path}")
    reports = [VideoEvidenceReport.from_mapping(item) for item in videos]
    if len(reports) < 2:
        raise ValueError(f"focus planning requires at least two videos: {path}")
    return question, reports


def _durations_from_payload(payload: dict[str, Any], path: Path) -> dict[str, float]:
    """How long each video of this sample runs, from the frozen observation.

    The planner must not open a file, but it must not hand back a window that
    runs past the end of one either, so the number comes from the report the
    state was built from -- the same artifact the event catalogue comes from.
    """
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"report has no videos list: {path}")
    durations: dict[str, float] = {}
    for item in videos:
        video_id = str(item.get("video_id", "")).strip()
        duration = item.get("duration_seconds")
        if not video_id or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"report has no usable duration for {video_id!r}: {path}")
        durations[video_id] = float(duration)
    return durations


def _restore_work(
    state_payload: dict[str, Any],
    *,
    question: str,
    reports: list[VideoEvidenceReport],
) -> WorkingState:
    """Rebuild the working state from its own serialization.

    ``normalize_initial_state`` re-applies every mechanical check, but it does
    not read ``filled_by_module`` -- that flag is written by the builder, not by
    the model it is meant to parse.  Leaving it out would make a module repair
    look like a model observation, so it is restored here by id.
    """
    state = normalize_initial_state(question=question, reports=reports, payload=state_payload)
    filled = {
        str(item.get("id", ""))
        for item in state_payload.get("unresolved_issues", []) or []
        if item.get("filled_by_module")
    }
    if filled:
        state = replace(
            state,
            unresolved_issues=tuple(
                replace(issue, filled_by_module=True)
                if issue.issue_id in filled and not issue.filled_by_module
                else issue
                for issue in state.unresolved_issues
            ),
        )
    event_videos = tuple(
        (event.event_id, report.video_id) for report in reports for event in report.events
    )
    return WorkingState(
        state=state,
        round_id=0,
        event_videos=event_videos,
        module_filled_pairs=state.coverage_gaps,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.api_env_file:
        _load_dotenv(args.api_env_file.expanduser().resolve())

    config = FocusPlannerConfig(
        margin_seconds=args.margin_seconds,
        minimum_span_seconds=args.minimum_span_seconds,
        maximum_span_seconds=args.maximum_span_seconds,
    )
    builder, builder_identity = _builder(args)
    tasks = {item.strip().upper() for item in args.tasks.split(",")} if args.tasks else None
    sample_ids = {item.strip() for item in args.sample_ids.split(",")} if args.sample_ids else None
    state_paths = _discover(args.state_dir.expanduser().resolve(), tasks, sample_ids)
    if args.limit is not None:
        state_paths = state_paths[: args.limit]
    if not state_paths:
        raise RuntimeError("no initial states matched the requested filters")
    print(f"[plan] {len(state_paths)} states builder={builder_identity}", flush=True)

    reports_root = args.reports_dir.expanduser().resolve()
    totals = {
        "planned": 0,
        "no_plan": 0,
        "refused": 0,
        "all_starved_refused": 0,
        "partial": 0,
        "looks": 0,
        "clipped": 0,
        "deferred": 0,
    }
    for position, state_path in enumerate(state_paths, start=1):
        sample_id = state_path.parent.name
        task = state_path.parent.parent.name
        report_path = reports_root / task / sample_id / "report.json"
        if not report_path.is_file():
            print(f"[{position}/{len(state_paths)}][{sample_id}] SKIP no report at {report_path}", flush=True)
            continue

        output_directory = args.output_dir.expanduser().resolve() / task / sample_id
        final_path = output_directory / "focus_plan.json"
        state_digest = file_digest(state_path)
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        _verify_report_inputs(state_payload, report_path=report_path)
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_digest = file_digest(report_path)
        if args.resume and final_path.is_file():
            existing = json.loads(final_path.read_text(encoding="utf-8"))
            stale = _reuse_decision(
                existing.get("run", {}),
                state_digest,
                builder_identity,
                destination=final_path,
                allow_builder_downgrade=args.allow_builder_downgrade,
            )
            if stale is None:
                print(f"[{position}/{len(state_paths)}][{sample_id}] resume skip complete", flush=True)
                continue
            print(
                f"[{position}/{len(state_paths)}][{sample_id}] regenerating: {stale}",
                flush=True,
            )

        question, reports = _reports_from_payload(report_payload, report_path)
        working = _restore_work(state_payload, question=question, reports=reports)
        catalogue: dict[str, EventRecord] = {
            event.event_id: event for report in reports for event in report.events
        }
        durations = _durations_from_payload(report_payload, report_path)

        try:
            result = plan_focus(
                working,
                events=catalogue,
                video_durations=durations,
                builder=builder,
                config=config,
            )
        except Exception as exc:  # one bad sample must not end the batch
            print(f"[{position}/{len(state_paths)}][{sample_id}] ERROR {type(exc).__name__}: {exc}", flush=True)
            continue

        serialized = result.to_dict()
        serialized["run"] = {
            "sample_id": sample_id,
            "task": task,
            "ground_truth": report_payload.get("run", {}).get("ground_truth"),
            "state_source": str(state_path),
            "report_source": str(report_path),
            "state_sha256": state_digest,
            "report_sha256": report_digest,
            "planner_identity": PLANNER_IDENTITY,
            "builder_identity": builder_identity,
            "focus_config": {
                "margin_seconds": config.margin_seconds,
                "minimum_span_seconds": config.minimum_span_seconds,
                "maximum_span_seconds": config.maximum_span_seconds,
            },
            # The medium each window was clipped to, so a reader can redo the
            # arithmetic from the plan alone.
            "video_durations": durations,
            "stages_executed": ["focus_planning"],
            "executed_focus_request": False,
            "generated_answer": False,
        }
        atomic_write_result(final_path, serialized)

        totals["refused"] += len(serialized["refusals"])
        totals["all_starved_refused"] += sum(
            1 for item in serialized["refusals"] if item["issue_class"] == "starved"
        )
        totals["deferred"] += len(serialized["deferred"])
        if result.plan is None:
            totals["no_plan"] += 1
            print(
                f"[{position}/{len(state_paths)}][{sample_id}] "
                f"NO PLAN issues={len(working.unresolved_issues)} "
                f"refused={len(serialized['refusals'])} "
                f"deferred={len(serialized['deferred'])} -> {final_path}",
                flush=True,
            )
        else:
            totals["planned"] += 1
            plan = result.plan
            totals["looks"] += len(plan.looks)
            if plan.partial:
                totals["partial"] += 1
            print(
                f"[{position}/{len(state_paths)}][{sample_id}] "
                f"plan issue={plan.issue_id} class={plan.issue_class} "
                f"looks={len(plan.looks)} videos={list(plan.looked_videos)} "
                f"bundle={list(plan.bundle_videos)} "
                f"need={plan.evidence_need_source} "
                f"targets={len(plan.request_target_videos)} "
                f"refused={len(serialized['refusals'])} "
                f"deferred={len(serialized['deferred'])} -> {final_path}",
                flush=True,
            )
            for look in plan.looks:
                start, end = look.time_window
                if look.window_clipped_to_medium:
                    totals["clipped"] += 1
                print(
                    f"    look {look.video_id} window=[{start:.1f},{end:.1f}] "
                    f"anchor={look.anchor_source} anchors={len(look.anchor_event_ids)} "
                    f"dropped={len(look.dropped_anchor_event_ids)} "
                    f"over_budget={look.window_budget_exceeded} "
                    f"clipped={look.window_clipped_to_medium}",
                    flush=True,
                )
            print(f"    evidence_need: {plan.evidence_need}", flush=True)
            if plan.evidence_need_note:
                print(f"    evidence_need_note: {plan.evidence_need_note}", flush=True)
            if plan.unanchorable_videos:
                print(
                    f"    unanchorable {list(plan.unanchorable_videos)}: their reports "
                    f"hold no event, so no window can be inherited; the issue stays "
                    f"unresolved and the rest of the bundle is still looked at",
                    flush=True,
                )
            if plan.uncovered_request_targets:
                print(
                    f"    request also names {list(plan.uncovered_request_targets)}, "
                    f"which this issue's scope does not cover",
                    flush=True,
                )
        for refusal in serialized["refusals"]:
            print(
                f"    refused {refusal['issue_id']} ({refusal['issue_class']}): {refusal['reason']}",
                flush=True,
            )
        for issue_id in serialized["deferred"]:
            print(
                f"    deferred {issue_id}: sorted behind the winner and never "
                f"attempted -- one request per round is the whole scope of this round",
                flush=True,
            )

    print(
        f"[done] planned={totals['planned']} no_plan={totals['no_plan']} "
        f"looks={totals['looks']} partial={totals['partial']} "
        f"clipped={totals['clipped']} refused={totals['refused']} "
        f"of_which_all_starved={totals['all_starved_refused']} "
        f"deferred={totals['deferred']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
