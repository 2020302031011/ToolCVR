from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolcvr.clip import DeterministicMockClipEncoder, TransformersClipEncoder
from toolcvr.pipeline import WarmStartConfig, WarmStartPipeline
from toolcvr.query import ConservativeQueryRewriter, OpenAICompatibleQueryRewriter
from toolcvr.reporting import DeterministicMockEventReporter, LocalQwen3VLReporter
from toolcvr.schemas import VideoInput


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolcvr")
    subparsers = parser.add_subparsers(dest="command", required=True)
    warm = subparsers.add_parser("warm-start", help="run query-conditioned sparse warm scan")
    warm.add_argument("--config", type=Path)
    warm.add_argument("--video", action="append", required=True, metavar="ID=PATH")
    warm.add_argument("--question", required=True)
    warm.add_argument("--option", action="append", default=[])
    warm.add_argument("--output", type=Path, required=True)
    warm.add_argument("--mock-models", action="store_true")
    warm.add_argument("--mock-reporter", action="store_true")
    warm.add_argument("--clip-model")
    warm.add_argument("--clip-device", default="cuda:0")
    warm.add_argument("--clip-batch-size", type=int, default=64)
    warm.add_argument("--vlm-model")
    warm.add_argument("--vlm-adapter")
    warm.add_argument("--vlm-device", default="cuda:0")
    warm.add_argument("--rewrite-base-url")
    warm.add_argument("--rewrite-model")
    warm.add_argument("--rewrite-api-key-env", default="LLM_API_KEY")
    return parser


def _video(value: str) -> VideoInput:
    if "=" not in value:
        raise ValueError(f"--video must use ID=PATH syntax: {value}")
    video_id, raw_path = value.split("=", 1)
    return VideoInput(video_id=video_id.strip(), path=Path(raw_path).expanduser().resolve())


def _run_warm_start(args: argparse.Namespace) -> int:
    config = WarmStartConfig.from_yaml(args.config) if args.config else WarmStartConfig()
    if args.mock_models:
        clip_encoder = DeterministicMockClipEncoder()
    else:
        if not args.clip_model:
            raise ValueError("real execution requires --clip-model or --mock-models")
        clip_encoder = TransformersClipEncoder(
            args.clip_model,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
        )

    if args.rewrite_base_url or args.rewrite_model:
        if not args.rewrite_base_url or not args.rewrite_model:
            raise ValueError("query API rewrite requires both --rewrite-base-url and --rewrite-model")
        rewriter = OpenAICompatibleQueryRewriter(
            base_url=args.rewrite_base_url,
            model=args.rewrite_model,
            api_key_env=args.rewrite_api_key_env,
        )
    else:
        rewriter = ConservativeQueryRewriter()

    if args.mock_models or args.mock_reporter:
        reporter = DeterministicMockEventReporter()
    else:
        if not args.vlm_model:
            raise ValueError("real reporting requires --vlm-model or --mock-reporter")
        reporter = LocalQwen3VLReporter(
            args.vlm_model,
            device=args.vlm_device,
            adapter_path=args.vlm_adapter,
            maximum_new_tokens=config.report.maximum_new_tokens,
            maximum_events=config.report.maximum_events_per_batch,
            overlay_timestamps=config.report.overlay_timestamps,
        )

    pipeline = WarmStartPipeline(
        config=config,
        rewriter=rewriter,
        clip_encoder=clip_encoder,
        reporter=reporter,
    )
    result = pipeline.run(
        videos=[_video(value) for value in args.video],
        question=args.question,
        options=args.option,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for video in result.videos:
        print(
            f"[{video.video_id}] stride={video.clip_scan_stride_seconds:.2f}s "
            f"candidates={video.clip_candidate_count} cache_hit={video.cache_hit} "
            f"mode={video.selection_mode} selected={video.selected_frames} "
            f"vlm_batches={video.report_batches} events={len(video.events)}",
            flush=True,
        )
    print(f"[done] {args.output.resolve()}", flush=True)
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "warm-start":
        return _run_warm_start(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
