#!/usr/bin/env python3
"""Fresh all-API framework test with only CLIP and video decoding kept local.

The same remote multimodal model performs query rewriting, warm observation,
state projection, planning, focused observation, reprojection, and answering.
No local language or vision-language checkpoint is loaded.  ``--clip-gpu``
selects the only CUDA device used, while ``--api-workers`` independently sets
the number of concurrent remote API/decode workers.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-gpu", default="0", help="one physical GPU for local CLIP")
    parser.add_argument(
        "--api-workers",
        type=int,
        default=8,
        help=(
            "maximum concurrent remote API requests; warm observation applies "
            "this limit to independent temporal batches"
        ),
    )
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cycles-dir", type=Path, required=True)
    parser.add_argument("--crossvid-root", type=Path, required=True)
    parser.add_argument("--api-env-file", type=Path, required=True)
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-model", default="qwen3.8-flash")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-maximum-tokens", type=int, default=8192)
    parser.add_argument(
        "--api-visual-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--visual-call-budget", type=int, default=8)
    parser.add_argument("--minimum-focus-stages", type=int, default=2)
    parser.add_argument("--context-token-budget", type=int, default=8192)
    parser.add_argument("--reasoner-max-tokens", type=int, default=2048)
    parser.add_argument("--max-vlm-tokens", type=int, default=2048)
    parser.add_argument("--maximum-events-per-batch", type=int, default=8)
    parser.add_argument("--decode-long-edge", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    return parser


def _dotenv(path: Path) -> dict[str, str]:
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
    print("[cmd] " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _require_artifacts(root: Path, filename: str, sample_ids: tuple[str, ...]) -> None:
    missing = [
        sample_id
        for sample_id in sample_ids
        if not (root / sample_id.split("-", 1)[0] / sample_id / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"stage did not commit {filename} for: {','.join(missing)}"
        )


def main() -> int:
    args = _parser().parse_args()
    samples = tuple(item.strip() for item in args.sample_ids.split(",") if item.strip())
    if not args.clip_gpu.isdigit():
        raise ValueError("--clip-gpu must be one physical CUDA index")
    if args.api_workers <= 0:
        raise ValueError("--api-workers must be positive")
    if not samples or len(set(samples)) != len(samples) or any("-" not in item for item in samples):
        raise ValueError("--sample-ids must contain unique TASK-ID values")
    if any(value <= 0 for value in (
        args.api_timeout_seconds,
        args.api_maximum_tokens,
        args.clip_batch_size,
        args.visual_call_budget,
        args.context_token_budget,
        args.reasoner_max_tokens,
        args.max_vlm_tokens,
        args.maximum_events_per_batch,
        args.decode_long_edge,
    )):
        raise ValueError("all budgets and limits must be positive")
    if args.minimum_focus_stages < 0:
        raise ValueError("minimum focus stages must be nonnegative")
    if not args.resume:
        for directory in (args.selection_dir, args.output_root, args.cycles_dir):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(
                    f"fresh API run refuses nonempty directory: {directory}; pass --resume"
                )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    api_env_file = args.api_env_file.expanduser().resolve()
    for key, value in _dotenv(api_env_file).items():
        env[key] = value
    api_base_url = args.api_base_url or env.get("LLM_BASE_URL")
    if not api_base_url:
        raise ValueError("API base URL is absent from arguments and dotenv")
    if not env.get(args.api_key_env):
        raise RuntimeError(f"API key is unset: {args.api_key_env}")

    selection = args.selection_dir.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    reports = output / "reports"
    state = output / "state"
    evidence = output / "evidence"
    cycles = args.cycles_dir.expanduser().resolve()
    common_ids = ",".join(samples)
    tasks = ",".join(dict.fromkeys(sample.split("-", 1)[0].upper() for sample in samples))

    selection_env = env.copy()
    selection_env["CUDA_VISIBLE_DEVICES"] = args.clip_gpu
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "audit_crossvid_clip_filter.py"),
        "--tasks", tasks,
        "--sample-ids", common_ids,
        "--crossvid-root", str(args.crossvid_root),
        "--output-dir", str(selection),
        "--api-env-file", str(api_env_file),
        "--api-base-url", api_base_url,
        "--api-model", args.api_model,
        "--api-key-env", args.api_key_env,
        "--clip-model", args.clip_model,
        "--clip-device", "cuda:0",
        "--clip-batch-size", str(args.clip_batch_size),
        "--resume",
    ], selection_env)

    visual_thinking_flag = (
        "--vlm-api-thinking" if args.api_visual_thinking else "--no-vlm-api-thinking"
    )
    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_parallel_vlm_stage.py"),
        "reports",
        "--worker-count", str(args.api_workers),
        "--sample-ids", common_ids,
        "--crossvid-root", str(args.crossvid_root),
        "--api-reporter",
        "--vlm-api-base-url", api_base_url,
        "--vlm-api-model", args.api_model,
        "--vlm-api-key-env", args.api_key_env,
        "--vlm-api-timeout-seconds", str(args.api_timeout_seconds),
        visual_thinking_flag,
        "--max-vlm-tokens", str(args.max_vlm_tokens),
        "--maximum-events-per-batch", str(args.maximum_events_per_batch),
        "--decode-long-edge", str(args.decode_long_edge),
        "--selection-dir", str(selection),
        "--output-dir", str(reports),
    ], env)
    _require_artifacts(reports, "report.json", samples)

    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_initial_state.py"),
        "--reports-dir", str(reports),
        "--output-dir", str(state),
        "--sample-ids", common_ids,
        "--api-env-file", str(api_env_file),
        "--api-base-url", api_base_url,
        "--api-model", args.api_model,
        "--api-key-env", args.api_key_env,
        "--api-timeout-seconds", str(args.api_timeout_seconds),
        "--api-maximum-tokens", str(args.api_maximum_tokens),
        "--workers", str(args.api_workers),
        "--resume",
    ], env)
    _require_artifacts(state, "initial_state.json", samples)

    _run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_parallel_reasoning_cycles.py"),
        "--worker-count", str(args.api_workers),
        "--sample-ids", common_ids,
        "--reports-dir", str(reports),
        "--state-dir", str(state),
        "--evidence-dir", str(evidence),
        "--crossvid-root", str(args.crossvid_root),
        "--api-base-url", api_base_url,
        "--api-model", args.api_model,
        "--api-key-env", args.api_key_env,
        "--api-timeout-seconds", str(args.api_timeout_seconds),
        "--visual-call-budget", str(args.visual_call_budget),
        "--minimum-focus-stages", str(args.minimum_focus_stages),
        "--context-token-budget", str(args.context_token_budget),
        "--reasoner-max-tokens", str(args.reasoner_max_tokens),
        "--api-reporter",
        "--vlm-api-base-url", api_base_url,
        "--vlm-api-model", args.api_model,
        "--vlm-api-key-env", args.api_key_env,
        "--vlm-api-timeout-seconds", str(args.api_timeout_seconds),
        visual_thinking_flag,
        "--max-vlm-tokens", str(args.max_vlm_tokens),
        "--maximum-events-per-batch", str(args.maximum_events_per_batch),
        "--decode-long-edge", str(args.decode_long_edge),
        "--output-root", str(cycles),
    ], env)
    _require_artifacts(cycles, "cycle_result.json", samples)
    print(
        f"[done] all-API run complete selection={selection} artifacts={output} cycles={cycles}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
