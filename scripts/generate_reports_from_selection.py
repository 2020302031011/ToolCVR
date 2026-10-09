#!/usr/bin/env python3
"""Generate resumable Observer report text from frozen CLIP selections.

This command never runs query rewriting or CLIP. It consumes only the exact
frame manifest recorded by a previous selection-only audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toolcvr.agents import ObserverAgent  # noqa: E402
from toolcvr.crossvid import CrossVidAuditAdapter, SUPPORTED_CROSSVID_TASKS  # noqa: E402
from toolcvr.pipeline import WarmStartConfig  # noqa: E402
from toolcvr.protocol import build_inference_protocol  # noqa: E402
from toolcvr.provenance import (  # noqa: E402
    MOCK_REPORTER_COMPATIBLE_IDENTITIES,
    MOCK_REPORTER_IDENTITY,
    local_qwen3vl_identity_suffix,
)
from toolcvr.report_generation import (  # noqa: E402
    SelectedReportGenerator,
    atomic_write_result,
    load_frozen_selection,
    remove_completed_batch_checkpoints,
)
from toolcvr.reporting import (  # noqa: E402
    DeterministicMockEventReporter,
    LocalQwen3VLReporter,
    OpenAICompatibleMultimodalReporter,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-dir",
        type=Path,
        required=True,
        help="root containing TASK/SAMPLE/result.json selection-only outputs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--crossvid-root",
        type=Path,
        default=Path("/media/data6/xuejj/CrossVid"),
    )
    parser.add_argument("--tasks", default=",".join(SUPPORTED_CROSSVID_TASKS))
    parser.add_argument(
        "--sample-ids",
        help="optional comma-separated exact IDs such as CC-1,NC-6",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "warm_start.yaml",
    )
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--api-reporter", action="store_true")
    parser.add_argument("--vlm-api-base-url")
    parser.add_argument("--vlm-api-model")
    parser.add_argument("--vlm-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--vlm-api-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=1,
        help="maximum concurrent visual API batch requests (API reporter only)",
    )
    parser.add_argument(
        "--vlm-api-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-vlm-tokens", type=int)
    parser.add_argument("--maximum-events-per-batch", type=int)
    parser.add_argument("--decode-long-edge", type=int)
    parser.add_argument("--mock-reporter", action="store_true")
    parser.add_argument(
        "--keep-batch-checkpoints",
        action="store_true",
        help="retain per-call JSON checkpoints after the final report is committed",
    )
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "crossvid-audit-proxies",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


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


def _reporter_identity_suffix(
    *,
    model: Path,
    adapter: Path | None,
    maximum_new_tokens: int,
    maximum_events: int,
    overlay_timestamps: bool,
) -> str:
    """How this run's reporter is named, by the one recipe both stages share.

    Still a named function rather than a direct call, because what has to hold
    is not that this script calls the shared recipe -- it is that this script
    and the focus executor feed the *same parameters* into it.  That is what a
    test can check without loading a model on either side; see
    ``tests/test_entry_points.py``.
    """
    return local_qwen3vl_identity_suffix(
        model=model,
        adapter=adapter,
        maximum_new_tokens=maximum_new_tokens,
        maximum_events=maximum_events,
        overlay_timestamps=overlay_timestamps,
    )


def _reporter(args: argparse.Namespace, config: WarmStartConfig):
    api_reporter = bool(getattr(args, "api_reporter", False))
    if args.mock_reporter:
        if api_reporter or args.vlm_model is not None or args.vlm_adapter is not None:
            raise ValueError("--mock-reporter cannot be combined with another reporter backend")
        return (
            DeterministicMockEventReporter(),
            MOCK_REPORTER_IDENTITY,
            MOCK_REPORTER_COMPATIBLE_IDENTITIES,
        )
    maximum_tokens = (
        args.max_vlm_tokens
        if args.max_vlm_tokens is not None
        else config.report.maximum_new_tokens
    )
    maximum_events = (
        args.maximum_events_per_batch
        if args.maximum_events_per_batch is not None
        else config.report.maximum_events_per_batch
    )
    if maximum_tokens <= 0 or maximum_events <= 0:
        raise ValueError("VLM token/event limits must be positive")
    if api_reporter:
        if args.vlm_model is not None or args.vlm_adapter is not None:
            raise ValueError("--api-reporter cannot be combined with local VLM paths")
        if not args.vlm_api_base_url or not args.vlm_api_model:
            raise ValueError("--api-reporter requires VLM API base URL and model")
        reporter = OpenAICompatibleMultimodalReporter(
            base_url=args.vlm_api_base_url,
            model=args.vlm_api_model,
            api_key_env=args.vlm_api_key_env,
            maximum_new_tokens=maximum_tokens,
            maximum_events=maximum_events,
            overlay_timestamps=config.report.overlay_timestamps,
            timeout_seconds=args.vlm_api_timeout_seconds,
            enable_thinking=args.vlm_api_thinking,
        )
        identity = (
            "openai-compatible-multimodal-v1:"
            f"endpoint={args.vlm_api_base_url.rstrip('/')}:"
            f"model={args.vlm_api_model}:tokens={maximum_tokens}:"
            f"events={maximum_events}:overlay={config.report.overlay_timestamps}:"
            f"thinking={args.vlm_api_thinking}"
        )
        return reporter, identity, ()
    if args.vlm_model is None:
        raise ValueError("--vlm-model is required for the local reporter")
    model = args.vlm_model.expanduser().resolve()
    if not model.exists():
        raise FileNotFoundError(f"VLM model does not exist: {model}")
    adapter: Path | None = None
    if args.vlm_adapter is not None:
        adapter = args.vlm_adapter.expanduser().resolve()
        if not adapter.exists():
            raise FileNotFoundError(f"VLM adapter does not exist: {adapter}")
    reporter = LocalQwen3VLReporter(
        str(model),
        device=args.vlm_device,
        adapter_path=str(adapter) if adapter is not None else None,
        maximum_new_tokens=maximum_tokens,
        maximum_events=maximum_events,
        overlay_timestamps=config.report.overlay_timestamps,
    )
    identity_suffix = _reporter_identity_suffix(
        model=model,
        adapter=adapter,
        maximum_new_tokens=maximum_tokens,
        maximum_events=maximum_events,
        overlay_timestamps=config.report.overlay_timestamps,
    )
    identity = (
        f"local-qwen3vl-v6-frame-manifest:{identity_suffix}"
    )
    compatible = (
        f"local-qwen3vl-v4-assessment:{identity_suffix}",
        # v3 predates per-batch assessments and observation gaps in the report
        # entirely, so loading one yields empty tuples for both.  That used to
        # be unsafe: _local_evidence_summary derived a count from them, and an
        # unmeasured zero reads exactly like a measured one.  It derives no
        # count any more, so what v3 omits now surfaces as absent rather than as
        # a finding about the video.
        f"local-qwen3vl-v3-temporal:{identity_suffix}",
    )
    return reporter, identity, compatible


def _selection_paths(
    root: Path,
    *,
    tasks: set[str],
    sample_ids: set[str] | None,
    start: int,
    limit: int | None,
) -> list[Path]:
    if start < 0:
        raise ValueError("--start must be nonnegative")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    paths = [
        path
        for path in sorted(root.glob("*/*/result.json"))
        if path.parent.parent.name.upper() in tasks
        and (sample_ids is None or path.parent.name in sample_ids)
    ]
    if sample_ids is not None:
        found = {path.parent.name for path in paths}
        missing = sorted(sample_ids - found)
        if missing:
            raise FileNotFoundError(f"selection results not found for: {missing}")
    stop = None if limit is None else start + limit
    return paths[start:stop]


def _validate_completed_result(
    path: Path,
    *,
    selection_sha256: str,
    reporter_identity: str,
    compatible_reporter_identities: tuple[str, ...] = (),
) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("run"), dict):
        raise ValueError(f"completed report has invalid metadata: {path}")
    run = raw["run"]
    if run.get("selection_sha256") != selection_sha256:
        raise ValueError(f"completed report belongs to a different selection: {path}")
    if run.get("reporter_identity") not in {
        reporter_identity,
        *compatible_reporter_identities,
    }:
        raise ValueError(f"completed report belongs to a different reporter: {path}")


def main() -> int:
    args = _parser().parse_args()
    if args.api_concurrency <= 0:
        raise ValueError("--api-concurrency must be positive")
    if args.api_concurrency != 1 and not args.api_reporter:
        raise ValueError("--api-concurrency greater than one requires --api-reporter")
    selection_root = args.selection_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    crossvid_root = args.crossvid_root.expanduser().resolve()
    if not selection_root.is_dir():
        raise FileNotFoundError(f"selection directory does not exist: {selection_root}")
    if output_root == selection_root or selection_root in output_root.parents:
        raise ValueError("report output must not be inside the frozen selection directory")
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise RuntimeError(
            f"refusing to write a fresh run into nonempty output directory: {output_root}"
        )

    tasks = {item.strip().upper() for item in args.tasks.split(",") if item.strip()}
    unsupported = sorted(tasks - set(SUPPORTED_CROSSVID_TASKS))
    if unsupported:
        raise ValueError(f"unsupported CrossVid tasks: {unsupported}")
    sample_ids = (
        {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        if args.sample_ids
        else None
    )
    paths = _selection_paths(
        selection_root,
        tasks=tasks,
        sample_ids=sample_ids,
        start=args.start,
        limit=args.limit,
    )
    if not paths:
        raise FileNotFoundError("no matching TASK/SAMPLE/result.json selections were found")

    config = WarmStartConfig.from_yaml(args.config.expanduser().resolve())
    reporter, reporter_identity, compatible_reporter_identities = _reporter(args, config)
    decode_long_edge = args.decode_long_edge or config.report.decode_long_edge
    generator = SelectedReportGenerator(
        observer=ObserverAgent(reporter),
        reporter_identity=reporter_identity,
        compatible_reporter_identities=compatible_reporter_identities,
        decode_long_edge=decode_long_edge,
        maximum_frames_per_batch=config.selection.frames_per_vlm_call,
        batch_workers=args.api_concurrency,
        progress=lambda message: print(message, flush=True),
    )
    adapter = CrossVidAuditAdapter(
        crossvid_root,
        proxy_directory=args.proxy_dir.expanduser().resolve(),
    )
    rows_by_task = {task: _load_rows(crossvid_root, task) for task in tasks}
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[reports] samples={len(paths)} reporter={reporter_identity} "
        f"decode_long_edge={decode_long_edge} api_concurrency={args.api_concurrency}",
        flush=True,
    )
    completed = 0
    failures: list[str] = []
    for position, source_path in enumerate(paths, start=1):
        selection = load_frozen_selection(
            source_path,
            maximum_frames_per_batch=config.selection.frames_per_vlm_call,
        )
        if selection.task not in tasks:
            raise ValueError(f"selection task/path mismatch: {source_path}")
        row_match = rows_by_task[selection.task].get(selection.sample_id)
        if row_match is None:
            raise KeyError(f"CrossVid source row not found for {selection.sample_id}")
        source_index, row = row_match
        sample = adapter.normalize(selection.task, row, source_index)
        if sample.question != selection.question or sample.options != selection.options:
            raise ValueError(
                f"CrossVid question/options changed since selection: {selection.sample_id}"
            )
        # Validate before the report-level resume fast path.  Otherwise a
        # legacy absolute-time PSS report could be silently reused merely
        # because its old selection digest still matches.
        selection.validate_video_inputs(sample.videos)

        sample_root = output_root / selection.task / selection.sample_id
        final_path = sample_root / "report.json"
        if args.resume and final_path.is_file():
            try:
                _validate_completed_result(
                    final_path,
                    selection_sha256=selection.source_sha256,
                    reporter_identity=reporter_identity,
                    compatible_reporter_identities=compatible_reporter_identities,
                )
            except ValueError as exc:
                digest = hashlib.sha256(final_path.read_bytes()).hexdigest()[:12]
                legacy_path = final_path.with_name(f"report.legacy-{digest}.json")
                if legacy_path.exists():
                    raise FileExistsError(
                        f"refusing to replace existing legacy report: {legacy_path}"
                    )
                final_path.replace(legacy_path)
                print(
                    f"[{position}/{len(paths)}][{selection.sample_id}] "
                    f"stale report preserved at {legacy_path}; regenerating: {exc}",
                    flush=True,
                )
            else:
                print(
                    f"[{position}/{len(paths)}][{selection.sample_id}] resume skip complete",
                    flush=True,
                )
                if not args.keep_batch_checkpoints:
                    remove_completed_batch_checkpoints(
                        sample_root / "batches",
                        sample_directory=sample_root,
                    )
                completed += 1
                continue

        batch_count = sum(video.report_batches for video in selection.videos)
        frame_count = sum(video.selected_frames for video in selection.videos)
        print(
            f"[{position}/{len(paths)}][{selection.sample_id}] "
            f"videos={len(selection.videos)} frames={frame_count} batches={batch_count}",
            flush=True,
        )
        try:
            result = generator.generate(
                selection=selection,
                videos=sample.videos,
                checkpoint_directory=sample_root / "batches",
                resume=args.resume,
            )
            payload = result.to_dict()
            payload["run"] = {
                "sample_id": selection.sample_id,
                "task": selection.task,
                "ground_truth": sample.answer,
                "selection_source": str(selection.source_path),
                "selection_sha256": selection.source_sha256,
                "reporter_identity": reporter_identity,
                "decode_long_edge": decode_long_edge,
                "structured_events_compiled_to_report_text": True,
                "free_form_summary_generation": False,
                "batch_checkpoints_retained": args.keep_batch_checkpoints,
                "inference_protocol": list(build_inference_protocol(
                    question=sample.question, options=sample.options,
                    videos=sample.videos, adapter_notes=sample.adaptation_notes,
                    answer_kind=sample.answer_kind,
                    required_answer_ids=sample.required_answer_ids,
                )),
            }
            atomic_write_result(final_path, payload)
            if not args.keep_batch_checkpoints:
                remove_completed_batch_checkpoints(
                    sample_root / "batches",
                    sample_directory=sample_root,
                )
            event_count = sum(len(video.events) for video in result.videos)
            gap_count = sum(len(video.observation_gaps) for video in result.videos)
            print(
                f"[{position}/{len(paths)}][{selection.sample_id}] "
                f"done events={event_count} gaps={gap_count} output={final_path} "
                f"batch_checkpoints={'kept' if args.keep_batch_checkpoints else 'removed'}",
                flush=True,
            )
            completed += 1
        except Exception as exc:
            # Isolate a failed sample so later samples can still commit their
            # artifacts, but never hide the failure from the stage scheduler.
            # Batch checkpoints remain on disk so a retry can resume from the
            # last committed batch rather than re-decoding the entire video.
            failures.append(f"{selection.sample_id}: {type(exc).__name__}: {exc}")
            print(
                f"[{position}/{len(paths)}][{selection.sample_id}] "
                f"FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()

    print(f"[done] completed={completed}/{len(paths)} output={output_root}", flush=True)
    for failure in failures:
        print(f"[failed] {failure}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
