#!/usr/bin/env python3
"""Run Observer VLM work across local GPUs or a bounded remote API pool.

Local models shard samples across GPUs.  Remote report generation uses one
process with a bounded batch-level thread pool, so independent temporal batches
can fill the API concurrency budget without stranding a large sample on one
serial worker.  Aggregation remains source-preserving and chronological.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ids(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("sample IDs must be nonempty and unique")
    return values


def _devices(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values) or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("--cuda-devices must be unique physical GPU indices")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reports", "focus"))
    parser.add_argument("--cuda-devices", type=_devices)
    parser.add_argument(
        "--worker-count",
        type=int,
        help="number of logical workers for an API reporter; does not reserve GPUs",
    )
    parser.add_argument("--sample-ids", type=_ids, required=True)
    parser.add_argument("--crossvid-root", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--api-reporter", action="store_true")
    parser.add_argument("--vlm-api-base-url")
    parser.add_argument("--vlm-api-model")
    parser.add_argument("--vlm-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--vlm-api-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--vlm-api-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-vlm-tokens", type=int, default=2048)
    parser.add_argument("--maximum-events-per-batch", type=int, default=8)
    parser.add_argument("--decode-long-edge", type=int)
    parser.add_argument("--frames-per-call", type=int, default=16)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-batch-checkpoints", action="store_true")
    return parser


def _worker_command(args: argparse.Namespace, sample_ids: tuple[str, ...]) -> list[str]:
    command = [sys.executable]
    if args.stage == "reports":
        if args.selection_dir is None:
            raise ValueError("reports stage requires --selection-dir")
        command += [
            str(PROJECT_ROOT / "scripts" / "generate_reports_from_selection.py"),
            "--selection-dir", str(args.selection_dir),
            "--output-dir", str(args.output_dir),
            "--crossvid-root", str(args.crossvid_root),
            "--sample-ids", ",".join(sample_ids),
            "--max-vlm-tokens", str(args.max_vlm_tokens),
            "--maximum-events-per-batch", str(args.maximum_events_per_batch),
            "--resume",
        ]
        if args.api_reporter:
            command += [
                "--api-reporter",
                "--vlm-api-base-url", args.vlm_api_base_url,
                "--vlm-api-model", args.vlm_api_model,
                "--vlm-api-key-env", args.vlm_api_key_env,
                "--vlm-api-timeout-seconds", str(args.vlm_api_timeout_seconds),
                "--api-concurrency", str(args.worker_count or 1),
                "--vlm-api-thinking" if args.vlm_api_thinking else "--no-vlm-api-thinking",
            ]
        else:
            command += ["--vlm-model", str(args.vlm_model), "--vlm-device", "cuda:0"]
        if args.keep_batch_checkpoints:
            command.append("--keep-batch-checkpoints")
    else:
        if args.plan_dir is None or args.reports_dir is None:
            raise ValueError("focus stage requires --plan-dir and --reports-dir")
        command += [
            str(PROJECT_ROOT / "scripts" / "execute_focus_plans.py"),
            "--plan-dir", str(args.plan_dir),
            "--reports-dir", str(args.reports_dir),
            "--output-dir", str(args.output_dir),
            "--crossvid-root", str(args.crossvid_root),
            "--sample-ids", ",".join(sample_ids),
            "--vlm-model", str(args.vlm_model),
            "--vlm-device", "cuda:0",
            "--max-vlm-tokens", str(args.max_vlm_tokens),
            "--maximum-events-per-batch", str(args.maximum_events_per_batch),
            "--frames-per-call", str(args.frames_per_call),
            "--resume",
        ]
    if args.vlm_adapter is not None:
        command += ["--vlm-adapter", str(args.vlm_adapter)]
    if args.decode_long_edge is not None:
        command += ["--decode-long-edge", str(args.decode_long_edge)]
    return command


def _artifact_path(stage: str, output: Path, sample_id: str) -> Path:
    task = sample_id.split("-", 1)[0]
    filename = "report.json" if stage == "reports" else "focus_evidence.json"
    return output / task / sample_id / filename


def main() -> int:
    args = _parser().parse_args()
    if args.max_vlm_tokens <= 0 or args.maximum_events_per_batch <= 0:
        raise ValueError("VLM limits must be positive")
    if args.api_reporter:
        if args.stage != "reports":
            raise ValueError("API reporter is currently supported by the reports stage and reasoning-cycle focus")
        if args.vlm_model is not None or args.vlm_adapter is not None:
            raise ValueError("API reporter cannot be combined with local VLM paths")
        if not args.vlm_api_base_url or not args.vlm_api_model:
            raise ValueError("API reporter requires VLM API base URL and model")
        if args.worker_count is not None and args.worker_count <= 0:
            raise ValueError("--worker-count must be positive")
        api_concurrency = args.worker_count or (
            len(args.cuda_devices) if args.cuda_devices else 1
        )
        # A single process owns the global request pool.  Multiple processes
        # would each apply their own limit and silently multiply concurrency.
        worker_labels = (f"api-pool{api_concurrency}",)
    else:
        if args.worker_count is not None:
            raise ValueError("--worker-count is only valid with --api-reporter")
        if not args.cuda_devices:
            raise ValueError("local reporter requires --cuda-devices")
        if args.vlm_model is None or not args.vlm_model.is_dir():
            raise FileNotFoundError(f"VLM model directory does not exist: {args.vlm_model}")
        if args.vlm_adapter is not None and not args.vlm_adapter.exists():
            raise FileNotFoundError(f"VLM adapter does not exist: {args.vlm_adapter}")
        worker_labels = args.cuda_devices

    # Resume means *artifact-aware* resume.  Filtering before sharding avoids
    # the pathological case where all failed work belonged to one old worker:
    # blindly reusing its old shard would leave N-1 GPUs idle on the retry.
    pending = tuple(
        sample_id
        for sample_id in args.sample_ids
        if not _artifact_path(args.stage, args.output_dir, sample_id).is_file()
    )
    skipped = len(args.sample_ids) - len(pending)
    if not pending:
        print(f"[done] {args.stage}: all {len(args.sample_ids)} requested artifacts already exist")
        return 0
    print(
        f"[schedule] {args.stage}: pending={len(pending)} completed={skipped} "
        + (
            f"with API batch concurrency={args.worker_count or 1}"
            if args.api_reporter
            else f"across {len(worker_labels)} GPU worker(s)"
        ),
        flush=True,
    )
    shards = [tuple(pending[index::len(worker_labels)]) for index in range(len(worker_labels))]
    workers: list[tuple[str, tuple[str, ...], subprocess.Popen[str], object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for worker_index, (worker_label, shard) in enumerate(zip(worker_labels, shards)):
        if not shard:
            continue
        log_path = args.output_dir / f"{args.stage}-worker{worker_index}-{worker_label}.log"
        environment = os.environ.copy()
        if not args.api_reporter:
            environment["CUDA_VISIBLE_DEVICES"] = worker_label
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            _worker_command(args, shard),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append((worker_label, shard, process, stream))
        kind = "API" if args.api_reporter else f"GPU {worker_label}"
        print(f"[worker {worker_index}] {kind}: {','.join(shard)} -> {log_path}", flush=True)

    failures: list[str] = []
    for worker_label, shard, process, stream in workers:
        code = process.wait()
        stream.close()
        if code:
            failures.append(f"worker {worker_label} ({','.join(shard)}): exit {code}")
    missing_artifacts = tuple(
        sample_id
        for sample_id in pending
        if not _artifact_path(args.stage, args.output_dir, sample_id).is_file()
    )
    if missing_artifacts:
        failures.append(
            "missing committed artifact(s): " + ",".join(missing_artifacts)
        )
    if failures:
        raise RuntimeError("parallel VLM worker failure(s): " + "; ".join(failures))
    print(
        f"[done] {args.stage}: processed={len(pending)} skipped={skipped} "
        f"across {len(workers)} worker(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
