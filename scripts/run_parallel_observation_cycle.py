#!/usr/bin/env python3
"""Execute selection → report → state → plan → focus as one observation cycle.

Only the two visual stages are sharded across ``--cuda-devices``.  The text
stages deliberately run once after all report artifacts exist, so their state
and plan each see the complete multi-video sample rather than worker shards.
"""

from __future__ import annotations

import argparse
import os
import signal
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_MODEL = Path("/media/data6/xuejj/Qwen3-14B")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-devices", required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crossvid-root", type=Path, required=True)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--decode-long-edge", type=int, default=1024)
    parser.add_argument("--max-vlm-tokens", type=int, default=2048)
    parser.add_argument("--maximum-events-per-batch", type=int, default=8)
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--api-env-file", type=Path)
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-maximum-tokens", type=int, default=8192)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-device", help="physical GPU for the temporary local LLM server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--external-llm", action="store_true", help="do not launch/stop a local LLM server")
    parser.add_argument("--skip-initial-focus", action="store_true", help="stop after initial state; the first visual zoom is chosen by ReasonerAgent")
    return parser


def _run(command: list[str], environment: dict[str, str]) -> None:
    print(f"[cmd] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _completed_ids(root: Path, artifact: str, requested: str) -> str:
    """Validate that a stage committed every requested artifact.

    A partially successful worker may preserve useful completed samples for a
    later resume, but no downstream stage may consume an incomplete sample set.
    This makes artifact completeness, rather than process exit alone, the stage
    commit contract.
    """
    values = [item.strip() for item in requested.split(",") if item.strip()]
    completed = [
        sample_id
        for sample_id in values
        if any(root.glob(f"*/{sample_id}/{artifact}"))
    ]
    if not completed:
        raise RuntimeError(f"no completed {artifact} artifacts under {root}")
    missing = sorted(set(values) - set(completed))
    if missing:
        raise RuntimeError(
            f"incomplete {artifact} stage: committed={len(completed)}/{len(values)}; "
            f"missing={','.join(missing)}"
        )
    return ",".join(completed)


def _text_backend(args: argparse.Namespace) -> list[str]:
    if args.mock_llm:
        return ["--mock"]
    if not args.api_base_url or not args.api_model:
        raise ValueError("text stages require --api-base-url and --api-model, or --mock-llm")
    backend = [
        "--api-base-url", args.api_base_url,
        "--api-model", args.api_model,
        "--api-key-env", args.api_key_env,
    ]
    if args.api_env_file is not None:
        backend += ["--api-env-file", str(args.api_env_file)]
    return backend


class _LocalServer:
    def __init__(self, *, model: Path, physical_gpu: str, port: int, environment: dict[str, str]) -> None:
        self.model = model
        self.physical_gpu = physical_gpu
        self.port = port
        self.environment = environment
        self.process: subprocess.Popen[object] | None = None

    def start(self) -> None:
        if self.process is not None:
            return
        env = self.environment.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.physical_gpu
        env.setdefault("LLM_API_KEY", "local-no-auth")
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "local_llm_server.py"),
            "--model", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
            "--device", "cuda:0", "--default-max-tokens", "2048",
        ]
        print(f"[server] starting local LLM on physical GPU {self.physical_gpu}", flush=True)
        self.process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, start_new_session=True)
        deadline = time.monotonic() + 900.0
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"local LLM server exited with {self.process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=3.0) as response:
                    if response.status == 200:
                        print(f"[server] ready at {url}", flush=True)
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(2.0)
        raise RuntimeError("local LLM server did not become ready within 900 seconds")

    def stop(self) -> None:
        if self.process is None:
            return
        print("[server] stopping local LLM", flush=True)
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process = None


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.expanduser().resolve()
    reports = output / "reports"
    state = output / "state"
    plan = output / "plan"
    evidence = output / "evidence"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    physical_devices = tuple(item.strip() for item in args.cuda_devices.split(",") if item.strip())
    if not physical_devices or any(not item.isdigit() for item in physical_devices):
        raise ValueError("--cuda-devices must be a comma-separated list of physical GPU indices")
    llm_gpu = args.llm_device or physical_devices[-1]
    if not args.external_llm and args.mock_llm:
        raise ValueError("--mock-llm is only meaningful with --external-llm")
    if not args.external_llm and not args.llm_model.is_dir():
        raise FileNotFoundError(f"LLM model directory does not exist: {args.llm_model}")
    if not args.api_base_url:
        args.api_base_url = f"http://127.0.0.1:{args.port}/v1"
    if not args.api_model:
        args.api_model = args.llm_model.name
    if not args.external_llm:
        # The local server accepts any nonempty bearer token; sharing this with
        # its clients keeps the OpenAI-compatible rewriter/builder contract.
        env.setdefault(args.api_key_env, "local-no-auth")
    server = None if args.external_llm else _LocalServer(
        model=args.llm_model, physical_gpu=llm_gpu, port=args.port, environment=env
    )
    common_vlm = [
        "--cuda-devices", args.cuda_devices,
        "--crossvid-root", str(args.crossvid_root),
        "--vlm-model", str(args.vlm_model),
        "--max-vlm-tokens", str(args.max_vlm_tokens),
        "--maximum-events-per-batch", str(args.maximum_events_per_batch),
        "--decode-long-edge", str(args.decode_long_edge),
    ]
    if args.vlm_adapter is not None:
        common_vlm += ["--vlm-adapter", str(args.vlm_adapter)]

    # CLIP selection is intentionally first and single-worker: it has one
    # shared feature cache and does not need a VLM replica.  Its GPU is the
    # first supplied physical device; all VLM cards are free again afterwards.
    if args.mock_llm:
        raise ValueError("--mock-llm cannot build query-conditioned CLIP selection")
    tasks = ",".join(dict.fromkeys(item.split("-", 1)[0].upper() for item in args.sample_ids.split(",")))
    selection_env = env.copy()
    selection_env["CUDA_VISIBLE_DEVICES"] = physical_devices[0]
    selection_command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "audit_crossvid_clip_filter.py"),
        "--tasks", tasks,
        "--sample-ids", args.sample_ids,
        "--crossvid-root", str(args.crossvid_root),
        "--output-dir", str(args.selection_dir),
        "--api-base-url", args.api_base_url,
        "--api-model", args.api_model,
        "--api-key-env", args.api_key_env,
        "--clip-device", "cuda:0",
        "--clip-batch-size", str(args.clip_batch_size),
    ]
    if args.api_env_file is not None:
        selection_command += ["--api-env-file", str(args.api_env_file)]
    selection_command.append("--resume")
    try:
        if server is not None:
            server.start()
        _run(selection_command, selection_env)
        if server is not None:
            server.stop()

        _run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_parallel_vlm_stage.py"),
         "reports", *common_vlm, "--sample-ids", args.sample_ids,
         "--selection-dir", str(args.selection_dir),
         "--output-dir", str(reports)], env,
        )
        report_ids = _completed_ids(reports, "report.json", args.sample_ids)
        text = _text_backend(args)
        if server is not None:
            server.start()
        _run(
         [sys.executable, str(PROJECT_ROOT / "scripts" / "build_initial_state.py"),
         "--reports-dir", str(reports), "--output-dir", str(state),
         "--sample-ids", report_ids, "--resume",
         "--api-timeout-seconds", str(args.api_timeout_seconds),
         "--api-maximum-tokens", str(args.api_maximum_tokens), *text], env,
        )
        state_ids = _completed_ids(state, "initial_state.json", report_ids)
        if args.skip_initial_focus:
            print(f"[done] reports={reports} state={state}; initial focus intentionally deferred to ReasonerAgent", flush=True)
            return 0
        _run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "plan_focus.py"),
         "--state-dir", str(state), "--reports-dir", str(reports),
         "--output-dir", str(plan), "--sample-ids", state_ids, "--resume",
         "--api-timeout-seconds", str(args.api_timeout_seconds), *text], env,
        )
        plan_ids = _completed_ids(plan, "focus_plan.json", state_ids)
        if server is not None:
            server.stop()
        _run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_parallel_vlm_stage.py"),
         "focus", *common_vlm, "--sample-ids", plan_ids, "--plan-dir", str(plan), "--reports-dir", str(reports),
         "--output-dir", str(evidence)], env,
        )
    finally:
        if server is not None:
            server.stop()
    print(f"[done] reports={reports} state={state} plan={plan} evidence={evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
