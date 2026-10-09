#!/usr/bin/env python3
"""Run the real text-only Reasoner ↔ visual Observer loop for one CrossVid sample.

This consumes a completed warm report and initial state. Warm SCAN events
become addressable coarse semantic slots; PLAN selects slots and the Observer
resolves their source video/time provenance into dense FOCUS calls.
There is intentionally no maximum number of rounds; ``--visual-call-budget``
is the hard stop for actual video work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toolcvr.agents import ObserverAgent
from toolcvr.crossvid import CrossVidAuditAdapter, crossvid_sample_id
from toolcvr.focus import FocusConfig, FocusTool
from toolcvr.reasoning_cycle import (
    SlotFocusExecutor,
    ContextBuilder,
    CycleConfig,
    CycleResult,
    CycleRunner,
    DecisionState,
    EvidenceLedger,
    EvidenceRequirement,
    LedgerEvent,
    LedgerGap,
    OpenAICompatibleReasonerBackend,
    OpenAICompatibleStateProjector,
    initialize_semantic_memory,
)
from toolcvr.protocol import build_inference_protocol
from toolcvr.reporting import LocalQwen3VLReporter, OpenAICompatibleMultimodalReporter
from toolcvr.schemas import EventRecord, ObservationBatch, ObservationGap, QueryAssessment
from toolcvr.trajectory import JsonlTrajectoryWriter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True, help="may be empty when first zoom is deferred to ReasonerAgent")
    parser.add_argument("--crossvid-root", type=Path, required=True)
    parser.add_argument("--proxy-dir", type=Path, default=PROJECT_ROOT / ".cache" / "crossvid-audit-proxies")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--reasoner-max-tokens", type=int, default=768)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--visual-call-budget", type=int, required=True)
    parser.add_argument(
        "--minimum-focus-stages", type=int, default=2,
        help="mandatory sequential PLAN→FOCUS stages before ANSWER is available",
    )
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument("--vlm-device", default="cuda:0")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_sample(root: Path, proxy_dir: Path, task: str, sample_id: str):
    path = root / "QA" / f"{task}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected QA list: {path}")
    adapter = CrossVidAuditAdapter(root, proxy_directory=proxy_dir)
    for index, row in enumerate(rows):
        if isinstance(row, dict) and crossvid_sample_id(task, row, index) == sample_id:
            return adapter.normalize(task, row, index)
    raise FileNotFoundError(f"sample not found in {path}: {sample_id}")


def _warm_ledger(report: dict[str, Any]) -> EvidenceLedger:
    events: list[LedgerEvent] = []
    gaps: list[LedgerGap] = []
    for video in report.get("videos", []):
        video_id = str(video["video_id"])
        for raw in video.get("events", []):
            event = EventRecord.from_mapping(raw)
            events.append(LedgerEvent(
                evidence_id=event.event_id, source_event_id=event.event_id, video_id=video_id,
                time_span=(event.start_seconds, event.end_seconds), description=event.description,
                uncertainty=event.uncertainty, frame_ids=event.evidence_frame_ids,
                confidence=event.confidence, action="SCAN", round_id=0,
                interpretation=event.interpretation,
            ))
        for raw in video.get("observation_gaps", []):
            gap = ObservationGap.from_mapping(raw)
            gaps.append(LedgerGap(gap.gap_id, video_id, (gap.start_seconds, gap.end_seconds), gap.frame_ids, gap.reason, "SCAN", 0))
    return EvidenceLedger(tuple(events), tuple(gaps))


def _state(
    payload: dict[str, Any],
    report: dict[str, Any],
    *,
    protocol_notes: tuple[str, ...] = (),
    answer_kind: str | None = None,
    required_answer_ids: tuple[str, ...] = (),
) -> DecisionState:
    unresolved = payload.get("unresolved_issues", [])
    open_questions = tuple(
        EvidenceRequirement(
            requirement_id=str(item["id"]), objective=str(item["issue"]),
            video_ids=tuple(str(value) for value in item["scope"]),
            evidence_refs=tuple(str(value) for value in item.get("related_events", [])),
        ) for item in unresolved
    )
    options = tuple(str(value) for value in report.get("options", []))
    video_ids = tuple(str(item) for item in payload["video_ids"])
    relations = payload.get("working_relations", [])
    relation_findings = tuple(
        EvidenceRequirement(
            requirement_id=str(item["id"]),
            objective=str(item["statement"]),
            video_ids=tuple(str(value) for value in item["scope"]),
            status="supported",
            evidence_refs=tuple(str(value) for value in item.get("supporting_events", [])),
        )
        for item in relations
        if item.get("statement") and item.get("supporting_events")
    )
    observed_findings = tuple(
        EvidenceRequirement(
            requirement_id=str(item["id"]),
            objective=str(item["statement"]),
            video_ids=tuple(str(value) for value in item["videos"]),
            status="supported",
            evidence_refs=tuple(str(value) for value in item.get("event_refs", [])),
        )
        for item in payload.get("observed_information", [])
        if item.get("statement") and item.get("event_refs")
    )
    summary = " ".join(str(item.get("statement", "")) for item in relations if item.get("statement"))[:1600]
    return DecisionState(
        question=str(payload["question"]), video_ids=video_ids,
        requirements=(*observed_findings, *relation_findings, *open_questions),
        options=options, summary=summary,
        protocol_notes=protocol_notes,
        answer_kind=answer_kind,
        required_answer_ids=required_answer_ids,
    )


def _coverage_regions(
    report: dict[str, Any],
    videos,
    *,
    maximum_span_seconds: float = 15.0,
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Address warm-scan omissions without treating them as negative evidence."""
    if maximum_span_seconds <= 0:
        raise ValueError("coverage slot span must be positive")
    report_by_video = {
        str(item["video_id"]): item for item in report.get("videos", [])
        if isinstance(item, dict) and "video_id" in item
    }
    result: dict[str, tuple[tuple[float, float], ...]] = {}
    for video in videos:
        payload = report_by_video.get(video.video_id)
        if payload is None:
            raise ValueError(f"warm report is missing video metadata: {video.video_id}")
        duration = float(payload.get("duration_seconds", 0.0))
        if duration <= 0:
            raise ValueError(f"warm report has invalid duration for {video.video_id}")
        # Coverage and subsequent Focus planning live entirely in the
        # model-visible clock.  In particular, PSS source offsets must never
        # become memory slots: they encode the withheld chronological order.
        domains = video.model_time_domains(duration)
        observed = sorted(
            (max(0.0, float(item[0])), min(duration, float(item[1])))
            for item in payload.get("observed_spans", [])
            if isinstance(item, (list, tuple)) and len(item) == 2
            and float(item[1]) >= float(item[0])
        )
        merged: list[list[float]] = []
        for start, end in observed:
            if not merged or start > merged[-1][1] + 0.25:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        gaps: list[tuple[float, float]] = []
        for domain_start, domain_end in domains:
            cursor = domain_start
            for seen_start, seen_end in merged:
                if seen_end < domain_start or seen_start > domain_end:
                    continue
                clipped_start = max(domain_start, seen_start)
                clipped_end = min(domain_end, seen_end)
                if clipped_start - cursor >= 0.5:
                    gaps.extend(_split_span(cursor, clipped_start, maximum_span_seconds))
                cursor = max(cursor, clipped_end)
            if domain_end - cursor >= 0.5:
                gaps.extend(_split_span(cursor, domain_end, maximum_span_seconds))
        if gaps:
            result[video.video_id] = tuple(gaps)
    return result


def _split_span(start: float, end: float, maximum: float) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    cursor = start
    while end - cursor > maximum:
        regions.append((cursor, cursor + maximum))
        cursor += maximum
    if end > cursor:
        regions.append((cursor, end))
    return regions


def _append_initial_focus(ledger: EvidenceLedger, evidence: dict[str, Any]) -> tuple[EvidenceLedger, tuple[str, ...]]:
    refs: list[str] = []
    next_ledger = ledger
    for index, look in enumerate(evidence.get("looks", []), 1):
        observation = look.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("focus evidence look lacks observation")
        batch = ObservationBatch(
            events=tuple(EventRecord.from_mapping(item) for item in observation.get("events", [])),
            query_assessment=QueryAssessment.from_mapping(observation["query_assessment"]),
            observation_gaps=tuple(ObservationGap.from_mapping(item) for item in observation.get("observation_gaps", [])),
        )
        next_ledger, added, _ = next_ledger.append(
            action="FOCUS", round_id=1, observation_id=f"initial{index}",
            video_id=str(look["video_id"]), batch=batch,
        )
        refs.extend(added)
    if not refs and not next_ledger.gaps:
        raise ValueError("initial focus bundle contains neither events nor explicit gaps")
    return next_ledger, tuple(refs)


def _result_mapping(result: CycleResult, *, visual_call_budget: int) -> dict[str, Any]:
    return {
        "state": {"question": result.state.question, "round_id": result.state.round_id, "summary": result.state.summary,
                  "review": {"outcome": result.state.review_outcome,
                             "reason": result.state.review_reason,
                             "evidence_refs": list(result.state.review_evidence_refs),
                             "decision_eligible": (
                                 result.exploration_agenda.remaining_stages == 0
                                 and result.evidence_revision == result.projected_revision
                             ),
                             "residual_open_question_count": len(result.state.open_requirements)},
                  "inference_protocol": list(result.state.protocol_notes),
                  "answer_contract": {"kind": result.state.answer_kind, "required_video_ids": list(result.state.required_answer_ids)},
                  "findings": [{"id": item.requirement_id,
                                "statement": item.objective,
                                "videos": list(item.video_ids),
                                "evidence_refs": list(item.evidence_refs)}
                               for item in result.state.requirements if item.status != "open"],
                  "open_questions": [{"id": item.requirement_id,
                                      "question": item.objective,
                                      "videos": list(item.video_ids),
                                      "related_evidence_refs": list(item.evidence_refs)}
                                     for item in result.state.open_requirements],
                  "contrastive_frontier": [
                      {"id": item.candidate_id,
                       "question_id": item.related_requirement_id,
                       "videos": list(item.target_video_ids),
                       "observable": item.observable,
                       "purpose": item.purpose}
                      for item in result.state.candidate_observations
                  ]},
        "ledger": {"events": [{"id": item.evidence_id, "video_id": item.video_id, "time_span": item.time_span, "description": item.description, "direct_observation": item.description, "inferred_interpretation": item.interpretation, "uncertainty": item.uncertainty, "action": item.action, "round": item.round_id} for item in result.ledger.events],
                   "gaps": [{"id": item.gap_id, "video_id": item.video_id, "time_span": item.time_span, "reason": item.reason, "action": item.action, "round": item.round_id} for item in result.ledger.gaps]},
        "turns": [
            {"synthesis": [
                 {"id": update.requirement_id,
                  "kind": "open_question" if update.status == "open" else "finding",
                  "videos": list(update.video_ids),
                  "evidence_refs": list(update.evidence_refs),
                  "statement": update.summary}
                 for update in item.updates
             ],
             "summary": item.state_summary, "action": item.action.kind,
             "reason": item.action.reason, "answer": item.action.answer,
             "focus_intent": (
                 {"question_id": item.action.request.requirement_id,
                  "slot_ids": list(item.action.request.slot_ids),
                  "evidence_goal": item.action.request.evidence_goal,
                  "next_probe_goal": item.action.request.next_probe_goal,
                  "mode": item.action.request.mode}
                 if item.action.request is not None else None
             )}
            for item in result.turns
        ],
        "alignment_history": [
            {"round": item.round_id,
             "evidence_revision": item.evidence_revision,
             "requirement_id": item.requirement_id,
             "objective": item.objective,
             "videos": list(item.video_ids),
             "status": item.status,
             "evidence_refs": list(item.evidence_refs),
             "summary": item.summary}
            for item in result.alignment_history
        ],
        "alignment_state": {
            "evidence_revision": result.evidence_revision,
            "projected_revision": result.projected_revision,
            "aligned_revision": result.aligned_revision,
            "current_evidence_is_projected": (
                result.evidence_revision == result.projected_revision
            ),
            "current_evidence_is_aligned": (
                result.evidence_revision == result.aligned_revision
            ),
        },
        "observation_receipts": [
            {"round": item.round_id,
             "request_fingerprint": item.request_fingerprint,
             "requirement_id": item.requirement_id,
             "evidence_goal": item.evidence_goal,
             "targets": [
                 {"video_id": video_id, "time_span": [start, end]}
                 for video_id, start, end in item.targets
             ],
             "outcome": item.outcome,
             "event_refs": list(item.event_refs),
             "gap_refs": list(item.gap_refs)}
            for item in result.observation_receipts
        ],
        "final_action": {"kind": result.final_action.kind, "answer": result.final_action.answer, "evidence_refs": list(result.final_action.evidence_refs), "reason": result.final_action.reason,
                         "termination_origin": result.termination_origin},
        "visual_calls_used": result.visual_calls_used,
        "semantic_zoom_memory": result.memory.render(),
        "exploration_agenda": result.exploration_agenda.to_mapping(
            visual_calls_used=result.visual_calls_used,
            visual_call_budget=visual_call_budget,
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    task = args.task.upper()
    sample_id = args.sample_id
    report = _json(args.reports_dir / task / sample_id / "report.json")
    initial = _json(args.state_dir / task / sample_id / "initial_state.json")
    evidence_path = args.evidence_dir / task / sample_id / "focus_evidence.json"
    sample = _load_sample(args.crossvid_root.resolve(), args.proxy_dir.resolve(), task, sample_id)
    protocol_notes = build_inference_protocol(
        question=sample.question, options=sample.options, videos=sample.videos,
        adapter_notes=sample.adaptation_notes,
        answer_kind=sample.answer_kind,
        required_answer_ids=sample.required_answer_ids,
    )
    ledger = _warm_ledger(report)
    state = _state(
        initial,
        report,
        protocol_notes=protocol_notes,
        answer_kind=sample.answer_kind,
        required_answer_ids=sample.required_answer_ids,
    )
    memory = initialize_semantic_memory(
        ledger,
        coverage_regions=_coverage_regions(report, sample.videos),
    )
    if evidence_path.is_file():
        ledger, initial_refs = _append_initial_focus(ledger, _json(evidence_path))
        # Compatibility path for historical runs with a committed first focus.
        state = replace(state, round_id=1)
    else:
        initial_refs = ()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory = JsonlTrajectoryWriter(output / "trajectory.jsonl")
    trajectory.append("run_start", {
        "task": task,
        "sample_id": sample_id,
        "question": sample.question,
        "options": list(sample.options),
        "ground_truth": sample.answer,
        "answer_kind": sample.answer_kind,
        "required_answer_ids": list(sample.required_answer_ids),
        "inference_protocol": list(protocol_notes),
        "reports_path": str((args.reports_dir / task / sample_id / "report.json").resolve()),
        "reports_sha256": hashlib.sha256(json.dumps(
            report, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest(),
        "initial_state_path": str((args.state_dir / task / sample_id / "initial_state.json").resolve()),
        "initial_state_sha256": hashlib.sha256(json.dumps(
            initial, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest(),
        "configuration": {
            "reasoner_model": args.api_model,
            "observer_backend": "api" if args.api_reporter else "local",
            "observer_model": args.vlm_api_model if args.api_reporter else str(args.vlm_model),
            "visual_call_budget": args.visual_call_budget,
            "minimum_focus_stages": args.minimum_focus_stages,
            "context_token_budget": args.context_token_budget,
            "reasoner_max_tokens": args.reasoner_max_tokens,
            "projector_thinking": False,
            "observer_max_tokens": args.max_vlm_tokens,
            "decode_long_edge": args.decode_long_edge,
        },
    })
    def build_focus_tool() -> FocusTool:
        if args.api_reporter:
            if args.vlm_model is not None or args.vlm_adapter is not None:
                raise ValueError("API reporter cannot be combined with local VLM paths")
            if not args.vlm_api_base_url or not args.vlm_api_model:
                raise ValueError("API reporter requires VLM API base URL and model")
            reporter = OpenAICompatibleMultimodalReporter(
                base_url=args.vlm_api_base_url,
                model=args.vlm_api_model,
                api_key_env=args.vlm_api_key_env,
                maximum_new_tokens=args.max_vlm_tokens,
                maximum_events=args.maximum_events_per_batch,
                timeout_seconds=args.vlm_api_timeout_seconds,
                enable_thinking=args.vlm_api_thinking,
                trace_sink=trajectory.record_api_call,
            )
        else:
            if args.vlm_model is None:
                raise ValueError("local focus requires --vlm-model")
            reporter = LocalQwen3VLReporter(
                str(args.vlm_model.resolve()), device=args.vlm_device,
                adapter_path=str(args.vlm_adapter.resolve()) if args.vlm_adapter else None,
                maximum_new_tokens=args.max_vlm_tokens, maximum_events=args.maximum_events_per_batch,
            )
        return FocusTool(
            observer=ObserverAgent(reporter),
            config=FocusConfig(decode_long_edge=args.decode_long_edge),
        )
    observer = SlotFocusExecutor(
        tool_factory=build_focus_tool,
        videos=sample.videos, question=sample.question,
    )
    reasoner = OpenAICompatibleReasonerBackend(
        base_url=args.api_base_url, model=args.api_model,
        api_key_env=args.api_key_env, maximum_tokens=args.reasoner_max_tokens,
        timeout_seconds=args.api_timeout_seconds,
        trace_sink=trajectory.record_api_call,
    )
    projector = OpenAICompatibleStateProjector(
        base_url=args.api_base_url,
        model=args.api_model,
        api_key_env=args.api_key_env,
        maximum_tokens=args.reasoner_max_tokens,
        timeout_seconds=args.api_timeout_seconds,
        protocol_notes=protocol_notes,
        enable_thinking=False,
        trace_sink=trajectory.record_api_call,
    )
    progress = output / "progress.json"
    def save_progress(current_ledger, current_state, turn, calls, phase) -> None:
        progress.write_text(json.dumps({"phase": phase, "round_id": current_state.round_id, "visual_calls_used": calls, "last_action": turn.action.kind, "last_summary": turn.state_summary, "review_outcome": current_state.review_outcome, "review_reason": current_state.review_reason, "review_evidence_refs": list(current_state.review_evidence_refs), "open_question_count": len(current_state.open_requirements), "event_count": len(current_ledger.events), "gap_count": len(current_ledger.gaps)}, ensure_ascii=False, indent=2), encoding="utf-8")
        trajectory.append("cycle_progress", {
            "phase": phase,
            "round_id": current_state.round_id,
            "visual_calls_used": calls,
            "event_count": len(current_ledger.events),
            "gap_count": len(current_ledger.gaps),
            "state_summary": current_state.summary,
            "review": {"outcome": current_state.review_outcome,
                       "reason": current_state.review_reason,
                       "evidence_refs": list(current_state.review_evidence_refs),
                       "residual_open_question_count": len(current_state.open_requirements)},
            "findings": [
                {"id": item.requirement_id, "statement": item.objective,
                 "videos": list(item.video_ids), "evidence_refs": list(item.evidence_refs)}
                for item in current_state.requirements if item.status != "open"
            ],
            "open_questions": [
                {"id": item.requirement_id, "question": item.objective,
                 "videos": list(item.video_ids),
                 "related_evidence_refs": list(item.evidence_refs)}
                for item in current_state.open_requirements
            ],
            "contrastive_frontier": [
                {"id": item.candidate_id,
                 "question_id": item.related_requirement_id,
                 "videos": list(item.target_video_ids),
                 "observable": item.observable,
                 "purpose": item.purpose}
                for item in current_state.candidate_observations
            ],
            "turn": {
                "synthesis": [
                    {"id": update.requirement_id,
                     "kind": "open_question" if update.status == "open" else "finding",
                     "videos": list(update.video_ids),
                     "evidence_refs": list(update.evidence_refs),
                     "statement": update.summary}
                    for update in turn.updates
                ],
                "state_summary": turn.state_summary,
                "action": turn.action.kind,
                "answer": turn.action.answer,
                "evidence_refs": list(turn.action.evidence_refs),
                "reason": turn.action.reason,
                "focus_intent": (
                    {"question_id": turn.action.request.requirement_id,
                     "slot_ids": list(turn.action.request.slot_ids),
                     "evidence_goal": turn.action.request.evidence_goal,
                     "next_probe_goal": turn.action.request.next_probe_goal,
                     "mode": turn.action.request.mode}
                    if turn.action.request is not None else None
                ),
            },
        })
    try:
        result = CycleRunner(config=CycleConfig(
            visual_call_budget=args.visual_call_budget,
            context_token_budget=args.context_token_budget,
            minimum_focus_stages=args.minimum_focus_stages,
        )).run(
            state=state, ledger=ledger, reasoner=reasoner, observer=observer,
            projector=projector,
            initial_new_refs=initial_refs, on_progress=save_progress,
            memory=memory,
        )
    except Exception as error:
        trajectory.append("run_failure", {
            "error_type": type(error).__name__,
            "error": str(error),
            "progress_path": str(progress),
        })
        raise
    target = output / "cycle_result.json"
    result_payload = _result_mapping(result, visual_call_budget=args.visual_call_budget)
    target.write_text(json.dumps(
        result_payload,
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    trajectory.append("run_end", {
        "cycle_result_path": str(target),
        "cycle_result_sha256": hashlib.sha256(json.dumps(
            result_payload, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest(),
        "final_action": result_payload["final_action"],
        "termination_origin": result.termination_origin,
        "visual_calls_used": result.visual_calls_used,
        "turn_count": len(result.turns),
    })
    print(f"[done] {target} action={result.final_action.kind} visual_calls={result.visual_calls_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
