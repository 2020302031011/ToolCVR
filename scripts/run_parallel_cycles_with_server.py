#!/usr/bin/env python3
"""Start one local Reasoner server and run parallel visual reasoning cycles."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-gpu", required=True)
    parser.add_argument("--vlm-gpus", required=True)
    parser.add_argument("--llm-model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--crossvid-root", type=Path, required=True)
    parser.add_argument("--visual-call-budget", type=int, required=True)
    parser.add_argument("--minimum-focus-stages", type=int, default=2)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--reasoner-max-tokens", type=int, default=768)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--max-vlm-tokens", type=int, default=2048)
    parser.add_argument("--maximum-events-per-batch", type=int, default=8)
    parser.add_argument("--decode-long-edge", type=int, default=1024)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.llm_gpu.isdigit() or any(not item.strip().isdigit() for item in args.vlm_gpus.split(",")):
        raise ValueError("GPU arguments must be physical numeric CUDA indices")
    if args.llm_gpu in {item.strip() for item in args.vlm_gpus.split(",")}:
        raise ValueError("Reasoner GPU must not also be a VLM worker GPU")
    env = os.environ.copy()
    env.setdefault("LLM_API_KEY", "local-no-auth")
    server_env = env.copy(); server_env["CUDA_VISIBLE_DEVICES"] = args.llm_gpu
    server_command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "local_llm_server.py"),
        "--model", str(args.llm_model.resolve()), "--host", "127.0.0.1", "--port", str(args.port),
        "--device", "cuda:0", "--default-max-tokens", "2048",
    ]
    print(f"[server] starting Reasoner on GPU {args.llm_gpu}", flush=True)
    server = subprocess.Popen(server_command, cwd=PROJECT_ROOT, env=server_env, start_new_session=True)
    try:
        deadline = time.monotonic() + 900
        url = f"http://127.0.0.1:{args.port}/v1/models"
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Reasoner server exited with {server.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("Reasoner server did not become ready within 900 seconds")
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_parallel_reasoning_cycles.py"),
            "--cuda-devices", args.vlm_gpus, "--sample-ids", args.sample_ids,
            "--reports-dir", str(args.reports_dir), "--state-dir", str(args.state_dir),
            "--evidence-dir", str(args.evidence_dir), "--crossvid-root", str(args.crossvid_root),
            "--api-base-url", f"http://127.0.0.1:{args.port}/v1", "--api-model", args.llm_model.name,
            "--api-key-env", "LLM_API_KEY", "--visual-call-budget", str(args.visual_call_budget),
            "--minimum-focus-stages", str(args.minimum_focus_stages),
            "--context-token-budget", str(args.context_token_budget), "--reasoner-max-tokens", str(args.reasoner_max_tokens),
            "--vlm-model", str(args.vlm_model), "--max-vlm-tokens", str(args.max_vlm_tokens),
            "--maximum-events-per-batch", str(args.maximum_events_per_batch), "--decode-long-edge", str(args.decode_long_edge),
            "--output-root", str(args.output_root),
        ]
        if args.vlm_adapter is not None:
            command += ["--vlm-adapter", str(args.vlm_adapter)]
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    finally:
        print("[server] stopping Reasoner", flush=True)
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
