#!/usr/bin/env python3
"""End-to-end run: selection → reports → state → real reasoning loop.

The first stages use the established resumable observation-cycle runner on all
supplied GPUs.  Output roots must be new for a fresh run; ``--resume`` instead
requires every stage to reuse committed artifacts and fill only missing ones.
With a local Reasoner this launcher reserves the first GPU for that model and
uses the remaining GPUs for visual cycles.  With ``--external-llm`` every
supplied GPU remains available to the visual workers.
"""

from __future__ import annotations

import argparse
import os
import re
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
    parser.add_argument("--cuda-devices", required=True, help="two or more physical GPUs, e.g. 1,5")
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cycles-dir", type=Path, required=True)
    parser.add_argument("--crossvid-root", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--llm-model", type=Path, default=Path("/media/data6/xuejj/Qwen3-14B"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--external-llm",
        action="store_true",
        help="use an external OpenAI-compatible Reasoner instead of loading a local LLM",
    )
    parser.add_argument("--api-env-file", type=Path)
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-maximum-tokens", type=int, default=8192)
    parser.add_argument("--visual-call-budget", type=int, required=True)
    parser.add_argument("--minimum-focus-stages", type=int, default=2)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--reasoner-max-tokens", type=int, default=768)
    parser.add_argument("--max-vlm-tokens", type=int, default=2048)
    parser.add_argument("--maximum-events-per-batch", type=int, default=8)
    parser.add_argument("--decode-long-edge", type=int, default=1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse committed artifacts and run only missing stages/samples",
    )
    return parser


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"API environment file does not exist: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid dotenv line {path}:{line_number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid dotenv key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("[cmd] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


class _Server:
    def __init__(self, model: Path, gpu: str, port: int, environment: dict[str, str]) -> None:
        self.model, self.gpu, self.port, self.environment = model, gpu, port, environment
        self.process: subprocess.Popen[object] | None = None

    def start(self) -> None:
        env = self.environment.copy(); env["CUDA_VISIBLE_DEVICES"] = self.gpu; env.setdefault("LLM_API_KEY", "local-no-auth")
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "local_llm_server.py"), "--model", str(self.model.resolve()), "--host", "127.0.0.1", "--port", str(self.port), "--device", "cuda:0", "--default-max-tokens", "2048"]
        self.process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, start_new_session=True)
        deadline = time.monotonic() + 900
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"local Reasoner server exited with {self.process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        print(f"[server] ready on GPU {self.gpu}", flush=True); return
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(2)
        raise RuntimeError("local Reasoner server did not become ready")

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try: os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError: pass
        self.process = None


def main() -> int:
    args = _parser().parse_args()
    devices = tuple(item.strip() for item in args.cuda_devices.split(",") if item.strip())
    if len(devices) < 2 or any(not item.isdigit() for item in devices):
        raise ValueError("--cuda-devices needs at least two physical GPU indices")
    if not args.resume:
        for directory in (args.selection_dir, args.output_root, args.cycles_dir):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(
                    f"fresh full run refuses nonempty directory: {directory}; "
                    "pass --resume to reuse committed artifacts"
                )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    if args.api_env_file is not None:
        for key, value in _dotenv_values(args.api_env_file.expanduser().resolve()).items():
            # An explicitly supplied dotenv is the run configuration and must
            # not be shadowed by a stale key inherited from the login shell.
            env[key] = value
    if args.external_llm:
        args.api_base_url = args.api_base_url or env.get("LLM_BASE_URL")
        args.api_model = args.api_model or env.get("LLM_MODEL")
        if not args.api_base_url or not args.api_model:
            raise ValueError(
                "--external-llm requires --api-base-url/--api-model or corresponding dotenv values"
            )
        if not env.get(args.api_key_env):
            raise RuntimeError(f"external LLM API key is unset: {args.api_key_env}")
    else:
        env.setdefault("LLM_API_KEY", "local-no-auth")
    base = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_parallel_observation_cycle.py"),
        "--cuda-devices", ",".join(devices), "--sample-ids", args.sample_ids,
        "--selection-dir", str(args.selection_dir), "--output-root", str(args.output_root),
        "--crossvid-root", str(args.crossvid_root), "--vlm-model", str(args.vlm_model),
        "--llm-model", str(args.llm_model), "--port", str(args.port),
        "--max-vlm-tokens", str(args.max_vlm_tokens), "--maximum-events-per-batch", str(args.maximum_events_per_batch),
        "--decode-long-edge", str(args.decode_long_edge),
        "--api-timeout-seconds", str(args.api_timeout_seconds),
        "--api-maximum-tokens", str(args.api_maximum_tokens),
        "--skip-initial-focus",
    ]
    if args.vlm_adapter is not None: base += ["--vlm-adapter", str(args.vlm_adapter)]
    if args.external_llm:
        base += [
            "--external-llm",
            "--api-base-url", args.api_base_url,
            "--api-model", args.api_model,
            "--api-key-env", args.api_key_env,
        ]
        if args.api_env_file is not None:
            base += ["--api-env-file", str(args.api_env_file.expanduser().resolve())]
    _run(base, env)
    server = None if args.external_llm else _Server(args.llm_model, devices[0], args.port, env)
    try:
        if server is not None:
            server.start()
        cycle_devices = devices if args.external_llm else devices[1:]
        api_base_url = args.api_base_url if args.external_llm else f"http://127.0.0.1:{args.port}/v1"
        api_model = args.api_model if args.external_llm else args.llm_model.name
        cycles = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_parallel_reasoning_cycles.py"),
            "--cuda-devices", ",".join(cycle_devices), "--sample-ids", args.sample_ids,
            "--reports-dir", str(args.output_root / "reports"), "--state-dir", str(args.output_root / "state"),
            "--evidence-dir", str(args.output_root / "evidence"), "--crossvid-root", str(args.crossvid_root),
            "--api-base-url", api_base_url, "--api-model", api_model,
            "--api-key-env", args.api_key_env,
            "--api-timeout-seconds", str(args.api_timeout_seconds),
            "--visual-call-budget", str(args.visual_call_budget), "--context-token-budget", str(args.context_token_budget),
            "--minimum-focus-stages", str(args.minimum_focus_stages),
            "--reasoner-max-tokens", str(args.reasoner_max_tokens), "--vlm-model", str(args.vlm_model),
            "--max-vlm-tokens", str(args.max_vlm_tokens), "--maximum-events-per-batch", str(args.maximum_events_per_batch),
            "--decode-long-edge", str(args.decode_long_edge), "--output-root", str(args.cycles_dir),
        ]
        if args.vlm_adapter is not None: cycles += ["--vlm-adapter", str(args.vlm_adapter)]
        _run(cycles, env)
    finally:
        if server is not None:
            server.stop()
    print(f"[done] complete artifacts={args.output_root} cycles={args.cycles_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
