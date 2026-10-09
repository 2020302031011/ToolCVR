#!/usr/bin/env python3
"""Build the reasoner-side initial state from frozen observer reports.

This stage consumes only ``report.json`` files produced by
``generate_reports_from_selection.py``. It never decodes video, never loads
CLIP, and never calls a visual model.

The output contains no answer, no completeness flag, and no model-reported
confidence.  ``working_relations[].status`` and the per-pair coverage numbers
are derived mechanically from the report event IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toolcvr.initial_state import (  # noqa: E402
    DeterministicMockInitialStateBuilder,
    InitialStateConfig,
    OpenAICompatibleInitialStateBuilder,
)
from toolcvr.provenance import file_digest, reuse_decision  # noqa: E402
from toolcvr.report_generation import atomic_write_result  # noqa: E402
from toolcvr.schemas import VideoEvidenceReport  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="root containing TASK/SAMPLE/report.json observer reports",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", help="comma-separated task filter, e.g. PSS,CC,FSA")
    parser.add_argument("--sample-ids", help="comma-separated exact ids, e.g. CC-117")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-builder-downgrade",
        action="store_true",
        help=(
            "let --resume overwrite states a model built with states this run "
            "builds synthetically; without it such a resume stops, because a "
            "mock corpus that replaced a model's is indistinguishable "
            "afterwards from one no model ever touched"
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use the deterministic CPU builder instead of a language model",
    )
    parser.add_argument("--api-env-file", type=Path)
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--api-maximum-tokens", type=int, default=4096)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="maximum concurrent independent initial-state builds",
    )
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


def _discover(root: Path, tasks: set[str] | None, sample_ids: set[str] | None) -> list[Path]:
    paths = sorted(root.glob("*/*/report.json"))
    selected: list[Path] = []
    for path in paths:
        sample_id, task = path.parent.name, path.parent.parent.name
        if tasks is not None and task.upper() not in tasks:
            continue
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        selected.append(path)
    return selected


def _builder(args: argparse.Namespace, config: InitialStateConfig):
    if args.mock:
        return (
            DeterministicMockInitialStateBuilder(config=config),
            "deterministic-mock-v2-local-assessment",
        )
    base_url = args.api_base_url or os.environ.get("LLM_BASE_URL", "")
    model = args.api_model or os.environ.get("LLM_MODEL", "")
    if not base_url or not model:
        raise RuntimeError(
            "initial state builder requires --mock, or LLM_BASE_URL/LLM_MODEL "
            "from the environment or --api-base-url/--api-model"
        )
    builder = OpenAICompatibleInitialStateBuilder(
        base_url=base_url,
        model=model,
        api_key_env=args.api_key_env,
        timeout_seconds=args.api_timeout_seconds,
        maximum_tokens=args.api_maximum_tokens,
        config=config,
    )
    return builder, f"openai-compatible-v3-anchored-local-summary:{model}"


def _reports_from_payload(payload: dict[str, Any], path: Path) -> tuple[str, tuple[str, ...], tuple[str, ...], list[VideoEvidenceReport]]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError(f"report has no question: {path}")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"report has no videos list: {path}")
    reports = [VideoEvidenceReport.from_mapping(item) for item in videos]
    if len(reports) < 2:
        raise ValueError(f"initial state formation requires at least two videos: {path}")
    raw_options = payload.get("options", [])
    if not isinstance(raw_options, list) or not all(isinstance(item, str) for item in raw_options):
        raise ValueError(f"report options must be a string list: {path}")
    raw_protocol = payload.get("run", {}).get("inference_protocol", [])
    if not isinstance(raw_protocol, list) or not all(isinstance(item, str) and item.strip() for item in raw_protocol):
        raise ValueError(f"report inference protocol must be a nonempty-string list: {path}")
    return question, tuple(raw_options), tuple(raw_protocol), reports


def _build_one(
    *,
    position: int,
    total: int,
    report_path: Path,
    output_root: Path,
    builder: Any,
    builder_identity: str,
    resume: bool,
    allow_builder_downgrade: bool,
) -> dict[str, int]:
    sample_id = report_path.parent.name
    task = report_path.parent.parent.name
    output_directory = output_root / task / sample_id
    final_path = output_directory / "initial_state.json"
    digest = file_digest(report_path)
    if resume and final_path.is_file():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        stale = reuse_decision(
            existing.get("run", {}),
            subject="state",
            identities={"builder_identity": builder_identity},
            inputs={"report_sha256": ("the report", digest)},
            destination=final_path,
            allow_downgrade=allow_builder_downgrade,
        )
        if stale is None:
            print(f"[{position}/{total}][{sample_id}] resume skip complete", flush=True)
            return {"pairs": 0, "module_filled": 0, "unresolved": 0, "relations": 0, "issues": 0}
        print(f"[{position}/{total}][{sample_id}] rebuilding: {stale}", flush=True)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    question, options, protocol_notes, reports = _reports_from_payload(payload, report_path)
    state = builder.build(
        question=question,
        reports=reports,
        options=options,
        protocol_notes=protocol_notes,
    )
    serialized = state.to_dict()
    coverage = serialized["coverage"]
    serialized["run"] = {
        "sample_id": sample_id,
        "task": task,
        "ground_truth": payload.get("run", {}).get("ground_truth"),
        "report_source": str(report_path),
        "report_sha256": digest,
        "builder_identity": builder_identity,
        "stages_executed": ["initial_state_formation"],
        "executed_candidate_requests": False,
        "generated_answer": False,
    }
    atomic_write_result(final_path, serialized)
    print(
        f"[{position}/{total}][{sample_id}] "
        f"pairs={coverage['covered_pair_count']}/{coverage['required_pair_count']} "
        f"module_filled={coverage['module_filled_pair_count']} "
        f"unresolved_pairs={coverage['unresolved_pair_count']} "
        f"relations={len(serialized['working_relations'])} "
        f"requests={len(serialized['candidate_requests'])} -> {final_path}",
        flush=True,
    )
    for note in serialized["validation_notes"]:
        print(f"    [{sample_id}] note: {note}", flush=True)
    return {
        "pairs": coverage["required_pair_count"],
        "module_filled": coverage["module_filled_pair_count"],
        "unresolved": coverage["unresolved_pair_count"],
        "relations": len(serialized["working_relations"]),
        "issues": len(serialized["unresolved_issues"]),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.api_env_file:
        _load_dotenv(args.api_env_file.expanduser().resolve())

    config = InitialStateConfig()
    builder, builder_identity = _builder(args, config)
    tasks = {item.strip().upper() for item in args.tasks.split(",")} if args.tasks else None
    sample_ids = {item.strip() for item in args.sample_ids.split(",")} if args.sample_ids else None
    paths = _discover(args.reports_dir.expanduser().resolve(), tasks, sample_ids)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError("no observer reports matched the requested filters")
    print(
        f"[plan] {len(paths)} reports builder={builder_identity} workers={args.workers}",
        flush=True,
    )

    totals = {"pairs": 0, "module_filled": 0, "unresolved": 0, "relations": 0, "issues": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _build_one,
                position=position,
                total=len(paths),
                report_path=report_path,
                output_root=args.output_dir,
                builder=builder,
                builder_identity=builder_identity,
                resume=args.resume,
                allow_builder_downgrade=args.allow_builder_downgrade,
            )
            for position, report_path in enumerate(paths, start=1)
        ]
        # Resolve in input order for deterministic failure selection. Other
        # independent jobs may finish first and commit their own artifacts.
        for future in futures:
            metrics = future.result()
            for key in totals:
                totals[key] += metrics[key]

    print(
        f"[done] completed={len(paths)}/{len(paths)} pairs={totals['pairs']} "
        f"relations={totals['relations']} issues={totals['issues']} "
        f"module_filled_pairs={totals['module_filled']} "
        f"unresolved_pairs={totals['unresolved']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
