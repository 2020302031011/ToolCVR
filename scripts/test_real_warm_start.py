#!/usr/bin/env python3
"""Run one real ToolCVR warm-start sample with API query rewriting.

The task name is used only to locate a CrossVid QA file. It never selects a
different inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toolcvr.clip import TransformersClipEncoder  # noqa: E402
from toolcvr.pipeline import WarmStartConfig, WarmStartPipeline  # noqa: E402
from toolcvr.query import OpenAICompatibleQueryRewriter  # noqa: E402
from toolcvr.reporting import LocalQwen3VLReporter, SelectionAuditReporter  # noqa: E402
from toolcvr.schemas import QueryPlan, VideoInput, WarmStartResult  # noqa: E402


DEFAULT_CROSSVID_ROOT = Path("/media/data6/xuejj/CrossVid")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/warm_start.yaml"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14-336"
DEFAULT_VLM_MODEL = Path("/media/data6/xuejj/Qwen-VL")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load one CrossVid item, rewrite its question/options with an "
            "OpenAI-compatible API, then run real CLIP selection and real Qwen-VL reporting."
        )
    )
    parser.add_argument("--task", default="CC", help="CrossVid QA filename stem, e.g. CC")
    parser.add_argument("--sample-id", default="1", help="record id, e.g. 1 or CC-1")
    parser.add_argument(
        "--video-index",
        type=int,
        action="append",
        help="1-based video index to run; repeat as needed, omit to run every video",
    )
    parser.add_argument("--crossvid-root", type=Path, default=DEFAULT_CROSSVID_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--api-env-file",
        type=Path,
        help="optional dotenv file containing LLM_BASE_URL, LLM_API_KEY and LLM_MODEL",
    )
    parser.add_argument("--api-base-url", help="defaults to LLM_BASE_URL")
    parser.add_argument("--api-model", help="defaults to LLM_MODEL")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--rewrite-only", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="run real CLIP selection and export contact sheets without loading a VLM",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="contact-sheet directory for --selection-only",
    )
    parser.add_argument(
        "--timeline-audit-dir",
        type=Path,
        help=(
            "export every sparse CLIP candidate with SELECTED/DROPPED markers; "
            "use this to audit false-negative filtering"
        ),
    )
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--vlm-model", type=Path, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument(
        "--max-vlm-tokens",
        type=int,
        help="override report.maximum_new_tokens from the YAML config",
    )
    parser.add_argument("--output", type=Path)
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


def _load_sample(crossvid_root: Path, task: str, sample_id: str) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ValueError("task may contain only letters, numbers, '_' and '-'")
    qa_path = crossvid_root / "QA" / f"{task.upper()}.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"CrossVid QA file does not exist: {qa_path}")
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"CrossVid QA file must contain a list: {qa_path}")
    normalized_id = sample_id.rsplit("-", 1)[-1]
    for row in payload:
        if isinstance(row, dict) and str(row.get("id")) == normalized_id:
            return row
    raise KeyError(f"sample {task}-{normalized_id} was not found in {qa_path}")


def _sample_text(sample: dict[str, object]) -> tuple[str, tuple[str, ...]]:
    question = sample.get("question")
    options = sample.get("options", [])
    if not isinstance(question, str) or not question.strip():
        raise ValueError("sample question must be a nonempty string")
    if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
        raise ValueError("sample options must be a string list")
    return question, tuple(options)


def _video_inputs(
    sample: dict[str, object],
    crossvid_root: Path,
    requested_indices: Sequence[int] | None,
) -> tuple[VideoInput, ...]:
    raw_videos = sample.get("videos")
    if raw_videos is None:
        video_a = sample.get("video A")
        video_b = sample.get("video B")
        if isinstance(video_a, str) and isinstance(video_b, str):
            raw_videos = [video_a, video_b]
    if not isinstance(raw_videos, list) or not raw_videos or not all(
        isinstance(item, str) for item in raw_videos
    ):
        raise ValueError("sample videos must be a nonempty string list")
    if requested_indices:
        if len(set(requested_indices)) != len(requested_indices):
            raise ValueError("--video-index values must be unique")
        invalid = [item for item in requested_indices if item < 1 or item > len(raw_videos)]
        if invalid:
            raise ValueError(f"video index outside 1..{len(raw_videos)}: {invalid}")
        selected = set(requested_indices)
    else:
        selected = set(range(1, len(raw_videos) + 1))
    return tuple(
        VideoInput(video_id=f"v{index}", path=(crossvid_root / "videos" / relative).resolve())
        for index, relative in enumerate(raw_videos, start=1)
        if index in selected
    )


def _export_timeline_audit(
    *,
    pipeline: WarmStartPipeline,
    videos: Sequence[VideoInput],
    result: WarmStartResult,
    output_directory: Path,
) -> None:
    sketches = result.videos
    if len(sketches) != len(videos):
        raise RuntimeError("timeline audit videos do not align with warm-start result")
    columns = 6
    rows_per_page = 8
    page_size = columns * rows_per_page
    cell_width = 224
    cell_height = 126
    label_height = 32
    for video, sketch in zip(videos, sketches):
        timeline, _ = pipeline._timeline(video)
        selected = {item.timeline_index for item in sketch.frame_manifest}
        frames = list(
            pipeline.reader.iter_sparse(
                video.path,
                stride_seconds=timeline.stride_seconds,
                long_edge=cell_width,
                time_spans=video.time_spans,
            )
        )
        if len(frames) != len(timeline.timestamps):
            raise RuntimeError("timeline audit decode count differs from cached CLIP timeline")
        for timeline_index, frame in enumerate(frames):
            if frame.frame_index != int(timeline.frame_indices[timeline_index]):
                raise RuntimeError("timeline audit frame identity differs from CLIP cache")
            if abs(
                frame.timestamp_seconds
                - float(timeline.source_timestamps[timeline_index])
            ) > 0.1:
                raise RuntimeError("timeline audit PTS differs from CLIP cache")
        video_directory = output_directory.expanduser().resolve() / video.video_id
        video_directory.mkdir(parents=True, exist_ok=True)
        for page_index, offset in enumerate(range(0, len(frames), page_size), start=1):
            page_frames = frames[offset : offset + page_size]
            rows = math.ceil(len(page_frames) / columns)
            sheet = Image.new(
                "RGB",
                (columns * cell_width, rows * (cell_height + label_height)),
                color=(18, 18, 18),
            )
            draw = ImageDraw.Draw(sheet)
            for local_index, frame in enumerate(page_frames):
                timeline_index = offset + local_index
                row, column = divmod(local_index, columns)
                x = column * cell_width
                y = row * (cell_height + label_height)
                image = Image.fromarray(frame.rgb, mode="RGB")
                image.thumbnail((cell_width - 6, cell_height - 6))
                paste_x = x + (cell_width - image.width) // 2
                paste_y = y + (cell_height - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))
                retained = timeline_index in selected
                color = (30, 220, 70) if retained else (120, 120, 120)
                draw.rectangle(
                    (x + 1, y + 1, x + cell_width - 2, y + cell_height - 2),
                    outline=color,
                    width=4 if retained else 1,
                )
                status = "SELECTED" if retained else "DROPPED"
                draw.text(
                    (x + 4, y + cell_height + 3),
                    f"{status}  t={frame.timestamp_seconds:.1f}s  frame={frame.frame_index}",
                    fill=color,
                )
            sheet.save(video_directory / f"timeline-{page_index:03d}.jpg", quality=90)


@dataclass(frozen=True)
class _PreparedQueryRewriter:
    question: str
    options: tuple[str, ...]
    plan: QueryPlan

    def rewrite(self, question: str, options: Sequence[str]) -> QueryPlan:
        if question != self.question or tuple(options) != self.options:
            raise ValueError("prepared query plan was requested for a different sample")
        return self.plan


def _print_plan(plan: QueryPlan) -> None:
    print("[rewrite] retrieval phrases:", flush=True)
    for index, phrase in enumerate(plan.phrases, start=1):
        targets = ",".join(phrase.target_video_ids) if phrase.target_video_ids else "all"
        print(
            f"  {index}. {phrase.text} "
            f"(source={phrase.source}, weight={phrase.weight:.2f}, targets={targets})",
            flush=True,
        )
    if not plan.phrases:
        print("  <none: selector will use semantic-novelty coverage>", flush=True)
    print("[rewrite] report hints:", flush=True)
    for index, hint in enumerate(plan.report_hints, start=1):
        print(f"  {index}. {hint}", flush=True)


def _thinking_parameter(base_url: str) -> str | None:
    lowered = base_url.casefold()
    if "api.deepseek.com" in lowered:
        return "thinking"
    if "dashscope.aliyuncs.com" in lowered:
        return "enable_thinking"
    return None


def main() -> int:
    args = _parser().parse_args()
    if args.api_env_file:
        _load_dotenv(args.api_env_file.expanduser().resolve())

    sample = _load_sample(args.crossvid_root, args.task, args.sample_id)
    question, options = _sample_text(sample)
    base_url = args.api_base_url or os.environ.get("LLM_BASE_URL", "")
    model = args.api_model or os.environ.get("LLM_MODEL", "")
    if not base_url or not model:
        raise RuntimeError(
            "Qwen rewrite requires LLM_BASE_URL and LLM_MODEL in the environment/"
            "--api-env-file, or --api-base-url and --api-model"
        )

    print(
        f"[sample] {args.task.upper()}-{sample.get('id')} question={question!r}",
        flush=True,
    )
    thinking_parameter = _thinking_parameter(base_url)
    thinking_label = "disabled" if thinking_parameter else "unspecified"
    print(
        f"[rewrite] model={model} base_url={base_url} thinking={thinking_label}",
        flush=True,
    )
    api_rewriter = OpenAICompatibleQueryRewriter(
        base_url=base_url,
        model=model,
        api_key_env=args.api_key_env,
        enable_thinking=False if thinking_parameter else None,
        thinking_parameter=thinking_parameter,
    )
    plan = api_rewriter.rewrite(question, options)
    _print_plan(plan)
    if args.rewrite_only:
        return 0

    videos = _video_inputs(sample, args.crossvid_root, args.video_index)
    normalized_id = str(sample.get("id"))
    output = args.output or (
        PROJECT_ROOT / "outputs" / f"{args.task.lower()}-{normalized_id}-real-warm-start.json"
    )
    output = output.expanduser().resolve()
    config = WarmStartConfig.from_yaml(args.config)
    clip_encoder = TransformersClipEncoder(
        args.clip_model,
        device=args.clip_device,
        batch_size=args.clip_batch_size,
    )
    if args.selection_only:
        audit_directory = args.audit_dir or output.parent / f"{output.stem}-selected"
        reporter = SelectionAuditReporter(audit_directory)
        print(f"[selection-only] contact sheets={audit_directory.resolve()}", flush=True)
    else:
        if args.audit_dir:
            raise ValueError("--audit-dir requires --selection-only")
        reporter = LocalQwen3VLReporter(
            args.vlm_model,
            device=args.vlm_device,
            adapter_path=args.vlm_adapter,
            maximum_new_tokens=args.max_vlm_tokens or config.report.maximum_new_tokens,
            maximum_events=config.report.maximum_events_per_batch,
            overlay_timestamps=config.report.overlay_timestamps,
        )
    pipeline = WarmStartPipeline(
        config=config,
        rewriter=_PreparedQueryRewriter(question, options, plan),
        clip_encoder=clip_encoder,
        reporter=reporter,
    )
    result = pipeline.run(videos=videos, question=question, options=options)

    if args.timeline_audit_dir:
        _export_timeline_audit(
            pipeline=pipeline,
            videos=videos,
            result=result,
            output_directory=args.timeline_audit_dir,
        )
        print(
            f"[timeline-audit] full sparse timeline={args.timeline_audit_dir.resolve()}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["run"] = {
        "sample_id": f"{args.task.upper()}-{normalized_id}",
        "ground_truth": sample.get("answer"),
        "rewrite_model": model,
        "clip_model": args.clip_model,
        "vlm_model": None if args.selection_only else str(args.vlm_model),
        "max_vlm_tokens": (
            None
            if args.selection_only
            else args.max_vlm_tokens or config.report.maximum_new_tokens
        ),
        "selection_only": args.selection_only,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for video in result.videos:
        print(
            f"[{video.video_id}] stride={video.clip_scan_stride_seconds:.2f}s "
            f"candidates={video.clip_candidate_count} cache_hit={video.cache_hit} "
            f"mode={video.selection_mode} selected={video.selected_frames} "
            f"vlm_batches={video.report_batches} events={len(video.events)}",
            flush=True,
        )
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
