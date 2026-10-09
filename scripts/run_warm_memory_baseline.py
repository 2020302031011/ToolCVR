#!/usr/bin/env python3
"""Answer samples directly from validated warm reports, with no state or FOCUS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolcvr.direct_evidence import (
    OpenAICompatibleDirectEvidenceBackend,
    build_direct_evidence_context,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", required=True, help="comma-separated TASK-ID values")
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--context-token-budget", type=int, default=12_000)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _upgrade_result_schema(path: Path) -> None:
    """Canonicalize pre-audit results without rerunning the model."""
    result = _read_object(path)
    changed = False
    if "status" not in result:
        answer = result.get("answer")
        has_answer = answer is not None
        if isinstance(answer, str):
            has_answer = bool(answer.strip())
        elif isinstance(answer, (list, dict)):
            has_answer = bool(answer)
        result["status"] = "ANSWER" if has_answer else "ABSTAIN"
        changed = True
    if "rejected_evidence_refs" not in result:
        result["rejected_evidence_refs"] = []
        changed = True
    if "citation_status" not in result:
        refs = result.get("evidence_refs", [])
        result["citation_status"] = "grounded" if isinstance(refs, list) and refs else "missing"
        changed = True
    if result.get("mode") == "direct-warm-evidence-v1":
        result["mode"] = "direct-warm-evidence-v2"
        changed = True
    if changed:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    samples = tuple(item.strip() for item in args.sample_ids.split(",") if item.strip())
    if not samples or any("-" not in item for item in samples):
        raise ValueError("--sample-ids must contain TASK-ID values")
    reports_root = args.reports_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    backend = OpenAICompatibleDirectEvidenceBackend(
        base_url=args.api_base_url,
        model=args.api_model,
        api_key_env=args.api_key_env,
        maximum_tokens=args.max_tokens,
    )
    completed = 0
    skipped = 0
    failures: list[str] = []
    for sample_id in samples:
        task = sample_id.split("-", 1)[0].upper()
        source = reports_root / task / sample_id / "report.json"
        target_dir = output_root / task / sample_id
        target = target_dir / "result.json"
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"direct baseline result already exists: {target}")
            _upgrade_result_schema(target)
            print(f"[resume] {sample_id}", flush=True)
            skipped += 1
            continue
        try:
            report = _read_object(source)
            context, known_refs, context_tokens = build_direct_evidence_context(
                report, token_budget=args.context_token_budget,
            )
            options = report.get("options", [])
            assert isinstance(options, list)
            answer = backend.answer(context, known_refs=known_refs, options=options)
            run = report.get("run", {})
            ground_truth = run.get("ground_truth") if isinstance(run, dict) else None
            result = {
                "sample_id": sample_id,
                "task": task,
                "question": report["question"],
                "options": options,
                "status": answer.status,
                "answer": answer.answer,
                "evidence_refs": list(answer.evidence_refs),
                "rejected_evidence_refs": list(answer.rejected_evidence_refs),
                "citation_status": answer.citation_status,
                "ground_truth": ground_truth,
                "warm_event_count": len(known_refs),
                "context_tokens_estimated": context_tokens,
                "mode": "direct-warm-evidence-v2",
                "used_initial_state": False,
                "used_focus": False,
                "used_reprojection": False,
            }
            target_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            completed += 1
            print(
                f"[done] {sample_id} status={answer.status} answer={answer.answer!r} "
                f"citations={answer.citation_status} events={len(known_refs)} "
                f"context_tokens={context_tokens}",
                flush=True,
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            failures.append(f"{sample_id}: {type(error).__name__}: {error}")
            print(f"[failed] {failures[-1]}", flush=True)
    print(
        f"[summary] requested={len(samples)} completed={completed} skipped={skipped} "
        f"failed={len(failures)}",
        flush=True,
    )
    if failures:
        raise RuntimeError("direct warm baseline failures: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
