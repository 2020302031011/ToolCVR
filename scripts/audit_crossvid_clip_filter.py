#!/usr/bin/env python3
"""Batch-audit the universal CLIP filter on all nine CrossVid tasks.

The task name is used only to locate data and stratify the audit report.  Every
sample runs the same rewrite, CLIP timeline, temporal selector, and audit path.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.test_real_warm_start import (  # noqa: E402
    _PreparedQueryRewriter,
    _export_timeline_audit,
    _load_dotenv,
    _thinking_parameter,
)
from toolcvr.clip import TransformersClipEncoder  # noqa: E402
from toolcvr.crossvid import (  # noqa: E402
    CrossVidAuditAdapter,
    SUPPORTED_CROSSVID_TASKS,
    crossvid_sample_id,
)
from toolcvr.pipeline import WarmStartConfig, WarmStartPipeline  # noqa: E402
from toolcvr.query import OpenAICompatibleQueryRewriter  # noqa: E402
from toolcvr.reporting import SelectionAuditReporter  # noqa: E402
from toolcvr.schemas import QueryPlan, RetrievalPhrase, SparseFrame, VideoInput  # noqa: E402
from toolcvr.video import OpenCVVideoReader  # noqa: E402


_SELECTION_PROFILES = {
    "speed": 0.50,
    "balanced": 0.65,
    "recall": 0.80,
}


def _selection_clock_matches(
    path: Path, videos: tuple[VideoInput, ...],
) -> bool:
    """Whether a result can be reused under the current model-time contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"selection result is invalid JSON: {path}") from exc
    raw_videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(raw_videos, list):
        raise ValueError(f"selection result has no videos list: {path}")
    clocks: dict[str, str] = {}
    for item in raw_videos:
        if not isinstance(item, dict) or not isinstance(item.get("video_id"), str):
            raise ValueError(f"selection result has invalid video metadata: {path}")
        clocks[item["video_id"]] = str(item.get("model_clock", "source"))
    expected = {video.video_id: video.model_clock for video in videos}
    return clocks == expected


class _SampleDecodeCache(OpenCVVideoReader):
    """Reuse decoded RGB frames across profile variants for one QA sample."""

    def __init__(self) -> None:
        self._extracted: dict[tuple[str, int, int], SparseFrame] = {}
        self._timelines: dict[
            tuple[str, float, int, tuple[tuple[float, float], ...]],
            tuple[SparseFrame, ...],
        ] = {}
        self.extracted_hits = 0
        self.extracted_misses = 0
        self.timeline_hits = 0
        self.timeline_misses = 0

    def extract(
        self,
        path: Path,
        timestamps: list[float],
        *,
        long_edge: int,
        expected_frame_indices: list[int] | None = None,
    ) -> list[SparseFrame]:
        if expected_frame_indices is None:
            return super().extract(
                path,
                timestamps,
                long_edge=long_edge,
                expected_frame_indices=None,
            )
        if len(expected_frame_indices) != len(timestamps):
            raise ValueError("expected frame indices must align with timestamps")
        resolved = str(path.expanduser().resolve())
        keys = [
            (resolved, long_edge, frame_index)
            for frame_index in expected_frame_indices
        ]
        missing_positions = [
            position for position, key in enumerate(keys) if key not in self._extracted
        ]
        self.extracted_hits += len(keys) - len(missing_positions)
        self.extracted_misses += len(missing_positions)
        if missing_positions:
            decoded = super().extract(
                path,
                [timestamps[position] for position in missing_positions],
                long_edge=long_edge,
                expected_frame_indices=[
                    expected_frame_indices[position] for position in missing_positions
                ],
            )
            for position, frame in zip(missing_positions, decoded):
                self._extracted[keys[position]] = frame
        return [self._extracted[key] for key in keys]

    def iter_sparse(
        self,
        path: Path,
        *,
        stride_seconds: float,
        long_edge: int,
        time_spans: tuple[tuple[float, float], ...] = (),
    ) -> Iterator[SparseFrame]:
        key = (
            str(path.expanduser().resolve()),
            stride_seconds,
            long_edge,
            time_spans,
        )
        frames = self._timelines.get(key)
        if frames is None:
            frames = tuple(
                OpenCVVideoReader.iter_sparse(
                    self,
                    path,
                    stride_seconds=stride_seconds,
                    long_edge=long_edge,
                    time_spans=time_spans,
                )
            )
            self._timelines[key] = frames
            self.timeline_misses += len(frames)
        else:
            self.timeline_hits += len(frames)
        yield from frames


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=",".join(SUPPORTED_CROSSVID_TASKS))
    parser.add_argument(
        "--sample-ids",
        help="comma-separated exact sample ids; bypasses duration stratification",
    )
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--crossvid-root", type=Path, default=Path("/media/data6/xuejj/CrossVid"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/warm_start.yaml")
    parser.add_argument(
        "--api-env-file",
        type=Path,
        help=(
            "optional dotenv file for the query-rewrite endpoint; local OpenAI-compatible "
            "servers may instead use an existing environment key or a local placeholder"
        ),
    )
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument(
        "--selection-profile",
        choices=tuple(_SELECTION_PROFILES),
        help="rank frames by raw CLIP cosine and retain 50%%, 65%%, or 80%% per video",
    )
    parser.add_argument(
        "--selection-profiles",
        help=(
            "comma-separated profiles to run in one process, for example "
            "speed,balanced,recall; outputs are written below PROFILE/ and "
            "all profiles reuse one frozen query plan per sample"
        ),
    )
    parser.add_argument(
        "--retention-ratio",
        type=float,
        help=(
            "custom ranked-retention fraction in (0,1], for example 0.60; "
            "mutually exclusive with --selection-profile(s)"
        ),
    )
    parser.add_argument(
        "--query-plan-dir",
        type=Path,
        help="reuse frozen TASK/SAMPLE/query_plan.json files from another audit output",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache/crossvid-audit-proxies",
        help="independent cache for annotated MOC/MSR dual-view proxy videos",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _requested_profiles(args: argparse.Namespace) -> tuple[str | None, ...]:
    retention_ratio = getattr(args, "retention_ratio", None)
    if args.selection_profile and args.selection_profiles:
        raise ValueError(
            "--selection-profile and --selection-profiles are mutually exclusive"
        )
    if retention_ratio is not None and (
        args.selection_profile or args.selection_profiles
    ):
        raise ValueError(
            "--retention-ratio and --selection-profile(s) are mutually exclusive"
        )
    if retention_ratio is not None and not 0.0 < retention_ratio <= 1.0:
        raise ValueError("--retention-ratio must lie in (0,1]")
    if not args.selection_profiles:
        return (args.selection_profile,)
    profiles = tuple(
        item.strip().lower()
        for item in args.selection_profiles.split(",")
        if item.strip()
    )
    if not profiles:
        raise ValueError("--selection-profiles must contain at least one profile")
    unsupported = sorted(set(profiles) - set(_SELECTION_PROFILES))
    if unsupported:
        raise ValueError(
            f"selection profiles must be drawn from {tuple(_SELECTION_PROFILES)}; "
            f"unsupported: {unsupported}"
        )
    if len(set(profiles)) != len(profiles):
        raise ValueError("--selection-profiles must not contain duplicates")
    return profiles


def _rows(crossvid_root: Path, task: str) -> list[dict[str, object]]:
    path = crossvid_root / "QA" / f"{task}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"CrossVid QA file must contain an object list: {path}")
    return payload


def _duration(row: dict[str, object]) -> float:
    value = row.get("duration", 0.0)
    if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
        return float(sum(value))
    if isinstance(value, (int, float)):
        return float(value)
    # UAV rows do not carry seconds. Stable source order is sufficient for
    # broad ID coverage because their proxies all retain the full frame set.
    return 0.0


def _duration_stratified(
    rows: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("--samples-per-task must be positive")
    if count == 1:
        return [sorted(rows, key=_duration)[len(rows) // 2]]
    if count >= len(rows):
        return sorted(rows, key=_duration)
    ordered = sorted(rows, key=_duration)
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[position] for position in positions]


def _save_plan(path: Path, plan: QueryPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phrases": [
                    {
                        "text": phrase.text,
                        "source": phrase.source,
                        "weight": phrase.weight,
                        "target_video_ids": list(phrase.target_video_ids),
                    }
                    for phrase in plan.phrases
                ],
                "report_hints": list(plan.report_hints),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_plan(path: Path) -> QueryPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"query plan must be an object: {path}")
    phrases = raw.get("phrases")
    hints = raw.get("report_hints")
    if not isinstance(phrases, list) or not isinstance(hints, list):
        raise ValueError(f"query plan is missing phrases/report_hints: {path}")
    if not all(isinstance(item, dict) for item in phrases):
        raise ValueError(f"query plan phrases must all be objects: {path}")
    if not all(isinstance(item, str) and item.strip() for item in hints):
        raise ValueError(f"query plan report_hints must be nonempty strings: {path}")
    for item in phrases:
        targets = item.get("target_video_ids", [])
        if not isinstance(targets, list) or not all(isinstance(value, str) for value in targets):
            raise ValueError(f"query plan target_video_ids must be string lists: {path}")
    return QueryPlan(
        phrases=tuple(
            RetrievalPhrase(
                text=str(item["text"]),
                source=str(item["source"]),
                weight=float(item.get("weight", 1.0)),
                target_video_ids=tuple(str(value) for value in item.get("target_video_ids", [])),
            )
            for item in phrases
        ),
        report_hints=tuple(str(item) for item in hints),
    )


def _summarize(output_root: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(output_root.glob("*/*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        videos = payload.get("videos", [])
        candidates = sum(int(video["clip_candidate_count"]) for video in videos)
        selected = sum(int(video["selected_frames"]) for video in videos)
        rows.append(
            {
                "sample_id": payload["run"]["sample_id"],
                "task": payload["run"]["task"],
                "ground_truth": payload["run"]["ground_truth"],
                "retrieval_phrase_count": len(payload["query_plan"]["phrases"]),
                "candidate_frames": candidates,
                "selected_frames": selected,
                "retention_ratio": selected / candidates if candidates else 0.0,
                "visual_batches": sum(int(video["report_batches"]) for video in videos),
                "selection_modes": sorted({str(video["selection_mode"]) for video in videos}),
                "selection_profile": payload["run"].get("selection_profile"),
                "configured_retention_ratio": payload["run"].get(
                    "configured_retention_ratio"
                ),
                "per_video": [
                    {
                        "video_id": video["video_id"],
                        "candidate_frames": int(video["clip_candidate_count"]),
                        "selected_frames": int(video["selected_frames"]),
                        "retention_ratio": (
                            int(video["selected_frames"]) / int(video["clip_candidate_count"])
                            if int(video["clip_candidate_count"])
                            else 0.0
                        ),
                        "selection_mode": video["selection_mode"],
                    }
                    for video in videos
                ],
                "result": str(path.relative_to(output_root)),
            }
        )
    (output_root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        total_candidates = sum(int(row["candidate_frames"]) for row in rows)
        total_selected = sum(int(row["selected_frames"]) for row in rows)
        print(
            f"[summary] samples={len(rows)} selected={total_selected}/{total_candidates} "
            f"retention={total_selected / total_candidates:.1%}",
            flush=True,
        )


def main() -> int:
    args = _parser().parse_args()
    try:
        profiles = _requested_profiles(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    multi_profile = args.selection_profiles is not None
    tasks = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    unsupported = sorted(set(tasks) - set(SUPPORTED_CROSSVID_TASKS))
    if unsupported:
        raise SystemExit(f"supported tasks are {SUPPORTED_CROSSVID_TASKS}; unsupported: {unsupported}")
    if args.api_env_file is not None:
        _load_dotenv(args.api_env_file.expanduser().resolve())
    base_url = args.api_base_url or os.environ.get("LLM_BASE_URL", "")
    model_name = args.api_model or os.environ.get("LLM_MODEL", "")
    if not base_url or not model_name:
        raise RuntimeError("rewrite API base URL/model is unset")
    # A locally served OpenAI-compatible model normally ignores Authorization,
    # but the common rewriter interface consistently requires a nonempty key.
    # Do not manufacture one for a remote endpoint.
    if not os.environ.get(args.api_key_env):
        local = base_url.lower().startswith(("http://127.0.0.1", "http://localhost"))
        if not local:
            raise RuntimeError(
                f"query rewrite API key is unset: {args.api_key_env}; provide --api-env-file"
            )
        os.environ[args.api_key_env] = "local-no-auth"
    thinking_parameter = _thinking_parameter(base_url)
    rewriter = OpenAICompatibleQueryRewriter(
        base_url=base_url,
        model=model_name,
        api_key_env=args.api_key_env,
        enable_thinking=False if thinking_parameter else None,
        thinking_parameter=thinking_parameter,
    )
    base_config = WarmStartConfig.from_yaml(args.config)
    clip_encoder = TransformersClipEncoder(
        args.clip_model,
        device=args.clip_device,
        batch_size=args.clip_batch_size,
    )
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    external_query_plan_root = (
        args.query_plan_dir.expanduser().resolve()
        if args.query_plan_dir is not None
        else None
    )
    if external_query_plan_root is not None and not external_query_plan_root.is_dir():
        raise FileNotFoundError(
            f"query plan directory does not exist: {external_query_plan_root}"
        )
    shared_query_plan_root = output_root / "query_plans" if multi_profile else None
    print(
        f"[selection] profiles={','.join(profile or 'custom' if args.retention_ratio is not None else profile or 'config-default' for profile in profiles)} "
        f"retention_ratio={args.retention_ratio} "
        f"query_plan_dir={external_query_plan_root or shared_query_plan_root}",
        flush=True,
    )
    adapter = CrossVidAuditAdapter(
        args.crossvid_root,
        proxy_directory=args.proxy_dir,
    )

    requested_sample_ids = (
        {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        if args.sample_ids
        else None
    )
    found_sample_ids: set[str] = set()

    for task in tasks:
        task_rows = _rows(args.crossvid_root, task)
        indexed_rows = list(enumerate(task_rows))
        if requested_sample_ids is None:
            chosen = _duration_stratified(task_rows, args.samples_per_task)
            chosen_identities = {id(row) for row in chosen}
            selected_rows = [
                (source_index, row)
                for source_index, row in indexed_rows
                if id(row) in chosen_identities
            ]
        else:
            selected_rows = []
            for source_index, row in indexed_rows:
                sample_id = crossvid_sample_id(task, row, source_index)
                if sample_id in requested_sample_ids:
                    selected_rows.append((source_index, row))
                    found_sample_ids.add(sample_id)
        print(
            f"[{task}] {'requested' if requested_sample_ids is not None else 'stratified'} "
            "sample ids="
            + ",".join(str(row.get("id")) for _, row in selected_rows),
            flush=True,
        )
        for source_index, row in selected_rows:
            sample = adapter.normalize(task, row, source_index)
            sample_id = sample.sample_id
            for note in sample.adaptation_notes:
                print(f"[{sample_id}] data repair: {note}", flush=True)

            profile_roots = {
                profile: (output_root / profile if multi_profile else output_root)
                for profile in profiles
            }
            pending_profiles: list[str | None] = []
            for profile in profiles:
                existing = profile_roots[profile] / task / sample_id / "result.json"
                reusable = (
                    args.resume
                    and existing.is_file()
                    and _selection_clock_matches(existing, sample.videos)
                )
                if not reusable:
                    if args.resume and existing.is_file():
                        print(
                            f"[{sample_id}] stale selection clock contract; recomputing",
                            flush=True,
                        )
                    pending_profiles.append(profile)
            if not pending_profiles:
                print(f"[{sample_id}] resume skip all profiles", flush=True)
                continue

            question, options = sample.question, sample.options
            if multi_profile:
                plan_path = shared_query_plan_root / task / sample_id / "query_plan.json"
            else:
                plan_path = output_root / task / sample_id / "query_plan.json"
            if plan_path.is_file():
                plan = _load_plan(plan_path)
            elif external_query_plan_root is not None:
                source_plan_path = (
                    external_query_plan_root / task / sample_id / "query_plan.json"
                )
                if not source_plan_path.is_file():
                    raise FileNotFoundError(
                        f"frozen query plan is missing for {sample_id}: {source_plan_path}"
                    )
                plan = _load_plan(source_plan_path)
                _save_plan(plan_path, plan)
            else:
                plan = rewriter.rewrite(question, options)
                _save_plan(plan_path, plan)

            videos = sample.videos
            sample_reader = _SampleDecodeCache()
            for profile in pending_profiles:
                configured_retention_ratio = (
                    _SELECTION_PROFILES[profile]
                    if profile is not None
                    else args.retention_ratio
                )
                config = base_config
                if configured_retention_ratio is not None:
                    config = replace(
                        base_config,
                        selection=replace(
                            base_config.selection,
                            retention_ratio=configured_retention_ratio,
                        ),
                    )
                profile_root = profile_roots[profile]
                sample_root = profile_root / task / sample_id
                result_path = sample_root / "result.json"
                sample_plan_path = sample_root / "query_plan.json"
                if sample_plan_path != plan_path and not sample_plan_path.is_file():
                    _save_plan(sample_plan_path, plan)
                reporter = SelectionAuditReporter(sample_root / "selected")
                pipeline = WarmStartPipeline(
                    config=config,
                    rewriter=_PreparedQueryRewriter(question, options, plan),
                    clip_encoder=clip_encoder,
                    reporter=reporter,
                    reader=sample_reader,
                )
                result = pipeline.run(videos=videos, question=question, options=options)
                _export_timeline_audit(
                    pipeline=pipeline,
                    videos=videos,
                    result=result,
                    output_directory=sample_root / "timeline",
                )
                payload = result.to_dict()
                payload["run"] = {
                    "sample_id": sample_id,
                    "task": task,
                    "ground_truth": sample.answer,
                    "rewrite_model": model_name,
                    "clip_model": args.clip_model,
                    "selection_only": True,
                    "duration_stratified_audit": requested_sample_ids is None,
                    "selection_profile": (
                        profile
                        if profile is not None
                        else "custom" if args.retention_ratio is not None else None
                    ),
                    "configured_retention_ratio": configured_retention_ratio,
                    "query_plan_source": str(plan_path),
                    "adaptation_notes": list(sample.adaptation_notes),
                    "inputs": [
                        {
                            "video_id": video.video_id,
                            "source_time_spans": [list(span) for span in video.time_spans],
                            "scan_stride_seconds": video.scan_stride_seconds,
                            "synchronization_group": video.synchronization_group,
                        }
                        for video in videos
                    ],
                }
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                candidate_count = sum(
                    video.clip_candidate_count for video in result.videos
                )
                selected_count = sum(video.selected_frames for video in result.videos)
                selection_label = profile or (
                    f"ratio-{configured_retention_ratio:.0%}"
                    if configured_retention_ratio is not None
                    else "config-default"
                )
                print(
                    f"[{sample_id}][{selection_label}] "
                    f"selected={selected_count}/{candidate_count} "
                    f"retention={selected_count / candidate_count:.1%}",
                    flush=True,
                )
                _summarize(profile_root)
            print(
                f"[{sample_id}][decode-cache] selected_hits={sample_reader.extracted_hits} "
                f"selected_misses={sample_reader.extracted_misses} "
                f"timeline_hits={sample_reader.timeline_hits} "
                f"timeline_misses={sample_reader.timeline_misses}",
                flush=True,
            )

    if requested_sample_ids is not None:
        missing = sorted(requested_sample_ids - found_sample_ids)
        if missing:
            raise FileNotFoundError(f"requested CrossVid sample ids were not found: {missing}")

    for profile in profiles:
        _summarize(output_root / profile if multi_profile else output_root)
    print(f"[done] {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
