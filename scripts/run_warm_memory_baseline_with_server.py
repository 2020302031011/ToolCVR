#!/usr/bin/env python3
"""Start one local Reasoner and run the direct warm-evidence baseline."""
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
    parser.add_argument("--gpu", required=True, help="physical GPU for the local Reasoner")
    parser.add_argument("--llm-model", type=Path, required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-token-budget", type=int, default=12_000)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.gpu.isdigit():
        raise ValueError("--gpu must be one physical CUDA index")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env.setdefault("LLM_API_KEY", "local-no-auth")
    server_env = env.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    server_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "local_llm_server.py"),
        "--model", str(args.llm_model.expanduser().resolve()),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--device", "cuda:0",
        "--default-max-tokens", str(args.max_tokens),
    ]
    print(f"[server] starting direct-evidence Reasoner on GPU {args.gpu}", flush=True)
    server = subprocess.Popen(server_command, cwd=PROJECT_ROOT, env=server_env, start_new_session=True)
    try:
        deadline = time.monotonic() + 900
        url = f"http://127.0.0.1:{args.port}/v1/models"
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"local Reasoner server exited with {server.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        print("[server] ready", flush=True)
                        break
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("local Reasoner server did not become ready within 900 seconds")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_warm_memory_baseline.py"),
            "--sample-ids", args.sample_ids,
            "--reports-dir", str(args.reports_dir),
            "--output-dir", str(args.output_dir),
            "--api-base-url", f"http://127.0.0.1:{args.port}/v1",
            "--api-model", args.llm_model.name,
            "--api-key-env", "LLM_API_KEY",
            "--context-token-budget", str(args.context_token_budget),
            "--max-tokens", str(args.max_tokens),
        ]
        if args.resume:
            command.append("--resume")
        print("[cmd] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    finally:
        print("[server] stopping direct-evidence Reasoner", flush=True)
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
