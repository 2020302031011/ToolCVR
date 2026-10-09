#!/usr/bin/env python3
"""Run independent real reasoning cycles concurrently, one VLM worker per GPU.

The local text server is deliberately external to this launcher: one shared
Reasoner backend serves all workers, while every worker owns exactly one visual
model replica and never shares CUDA state with another process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-devices", help="physical VLM GPUs for the local reporter")
    parser.add_argument(
        "--worker-count",
        type=int,
        help="number of logical workers for an API reporter; does not reserve GPUs",
    )
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--crossvid-root", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--visual-call-budget", type=int, required=True)
    parser.add_argument("--minimum-focus-stages", type=int, default=2)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--reasoner-max-tokens", type=int, default=768)
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
    parser.add_argument("--decode-long-edge", type=int, default=1024)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _command(args: argparse.Namespace, sample_id: str) -> list[str]:
    task = sample_id.split("-", 1)[0]
    output = args.output_root.resolve() / task / sample_id
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_real_reasoning_cycle.py"),
        "--task", task, "--sample-id", sample_id,
        "--reports-dir", str(args.reports_dir.resolve()), "--state-dir", str(args.state_dir.resolve()),
        "--evidence-dir", str(args.evidence_dir.resolve()), "--crossvid-root", str(args.crossvid_root.resolve()),
        "--api-base-url", args.api_base_url, "--api-model", args.api_model,
        "--api-key-env", args.api_key_env,
        "--api-timeout-seconds", str(args.api_timeout_seconds),
        "--visual-call-budget", str(args.visual_call_budget),
        "--minimum-focus-stages", str(args.minimum_focus_stages),
        "--context-token-budget", str(args.context_token_budget), "--reasoner-max-tokens", str(args.reasoner_max_tokens),
        "--max-vlm-tokens", str(args.max_vlm_tokens),
        "--maximum-events-per-batch", str(args.maximum_events_per_batch), "--decode-long-edge", str(args.decode_long_edge),
        "--output-dir", str(output),
    ]
    if args.api_reporter:
        command += [
            "--api-reporter",
            "--vlm-api-base-url", args.vlm_api_base_url,
            "--vlm-api-model", args.vlm_api_model,
            "--vlm-api-key-env", args.vlm_api_key_env,
            "--vlm-api-timeout-seconds", str(args.vlm_api_timeout_seconds),
            "--vlm-api-thinking" if args.vlm_api_thinking else "--no-vlm-api-thinking",
        ]
    elif args.vlm_model is not None:
        command += ["--vlm-model", str(args.vlm_model.resolve())]
    if not args.api_reporter and args.vlm_adapter is not None:
        command += ["--vlm-adapter", str(args.vlm_adapter.resolve())]
    return command


def main() -> int:
    args = _parser().parse_args()
    devices = tuple(item.strip() for item in (args.cuda_devices or "").split(",") if item.strip())
    samples = tuple(item.strip() for item in args.sample_ids.split(",") if item.strip())
    if not samples or any("-" not in item for item in samples):
        raise ValueError("--sample-ids must contain TASK-ID values")
    if args.api_reporter:
        if args.vlm_model is not None or args.vlm_adapter is not None:
            raise ValueError("API reporter cannot be combined with local VLM paths")
        if not args.vlm_api_base_url or not args.vlm_api_model:
            raise ValueError("API reporter requires VLM API base URL and model")
        if args.worker_count is not None and args.worker_count <= 0:
            raise ValueError("--worker-count must be positive")
        worker_labels = tuple(
            f"api{index}" for index in range(args.worker_count or len(devices) or 1)
        )
    elif args.vlm_model is None:
        raise ValueError("local reporter requires --vlm-model")
    else:
        if args.worker_count is not None:
            raise ValueError("--worker-count is only valid with --api-reporter")
        if not devices or any(not item.isdigit() for item in devices):
            raise ValueError("local reporter requires physical --cuda-devices")
        worker_labels = devices
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    completed = [item for item in samples if (args.output_root.resolve() / item.split("-", 1)[0] / item / "cycle_result.json").exists()]
    pending = tuple(item for item in samples if item not in completed)
    if completed:
        print(f"[resume] preserving {len(completed)} completed cycle result(s)", flush=True)
    if not pending:
        print("[done] all requested cycles already have results", flush=True)
        return 0
    queues = [list(pending[index::len(worker_labels)]) for index in range(len(worker_labels))]
    processes: list[tuple[str, str, subprocess.Popen[object], object]] = []
    for worker, (worker_label, queue) in enumerate(zip(worker_labels, queues)):
        if not queue:
            continue
        sample_id = queue.pop(0)
        log_path = args.output_root.resolve() / f"worker{worker}-{worker_label}.log"
        log = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        if not args.api_reporter:
            env["CUDA_VISIBLE_DEVICES"] = worker_label
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        command = _command(args, sample_id)
        if not args.api_reporter:
            command += ["--vlm-device", "cuda:0"]
        kind = "API" if args.api_reporter else f"GPU {worker_label}"
        print(f"[worker {worker}] {kind}: {sample_id} -> {log_path}", flush=True)
        processes.append((worker_label, sample_id, subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failures: list[str] = []
    while processes:
        next_processes: list[tuple[str, str, subprocess.Popen[object], object]] = []
        for worker_label, sample_id, process, log in processes:
            status = process.poll()
            if status is None:
                next_processes.append((worker_label, sample_id, process, log)); continue
            log.close()
            if status:
                failures.append(f"{sample_id} (worker {worker_label}, exit {status})")
                print(f"[failed] {failures[-1]}", flush=True)
            else:
                print(f"[done] worker {worker_label}: {sample_id}", flush=True)
            worker = worker_labels.index(worker_label)
            if queues[worker]:
                successor = queues[worker].pop(0)
                log_path = args.output_root.resolve() / f"worker{worker}-{worker_label}.log"
                log = log_path.open("a", encoding="utf-8")
                env = os.environ.copy()
                if not args.api_reporter:
                    env["CUDA_VISIBLE_DEVICES"] = worker_label
                env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
                command = _command(args, successor)
                if not args.api_reporter:
                    command += ["--vlm-device", "cuda:0"]
                print(f"[worker {worker}] {worker_label}: {successor}", flush=True)
                next_processes.append((worker_label, successor, subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        processes = next_processes
        if processes:
            import time; time.sleep(2)
    if failures:
        raise RuntimeError(f"reasoning cycle failures: {', '.join(failures)}")
    print(f"[done] newly_completed={len(pending)} preserved={len(completed)} workers={len(worker_labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
