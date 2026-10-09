from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from toolcvr.reasoning_cycle import (
    ContextBuilder,
    CycleConfig,
    CycleRunner,
    DecisionState,
    EvidenceLedger,
    EvidenceRequirement,
    ExplorationAgenda,
    FocusReviewReceipt,
    FocusIntent,
    LedgerEvent,
    ObservationReceipt,
    ObservedBatch,
    OpenAICompatibleReasonerBackend,
    OpenAICompatibleStateProjector,
    ReasonerAction,
    ReasonerTurn,
    RequirementUpdate,
    SlotFocusExecutor,
    ZoomDraft,
    _parse_projected_state,
    _observation_fingerprint,
    _snapshot_reference_ids,
    initialize_semantic_memory,
    parse_reasoner_turn,
)
from toolcvr.schemas import (
    EventRecord, ObservationBatch, ObservationGap, QueryAssessment, VideoInput,
)


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(events=(
        LedgerEvent(
            evidence_id="v1.scan.r0.b1.e1", source_event_id="e1", video_id="v1",
            time_span=(1.0, 2.0), description="A red car enters the scene.", uncertainty="",
            frame_ids=("S1-F01",), confidence=0.9, action="SCAN", round_id=0,
        ),
        LedgerEvent(
            evidence_id="v2.scan.r0.b1.e1", source_event_id="e1", video_id="v2",
            time_span=(3.0, 4.0), description="A road is visible.", uncertainty="",
            frame_ids=("S2-F01",), confidence=0.8, action="SCAN", round_id=0,
        ),
    ))


def _state() -> DecisionState:
    return DecisionState(
        question="Which video shows the event?", video_ids=("v1", "v2"),
        requirements=(EvidenceRequirement("R01", "Verify whether v2 shows the same event.", ("v2",)),),
        options=("A. v1", "B. v2"), summary="v1 is observed; v2 remains unresolved.",
    )


@dataclass
class _ScriptedReasoner:
    turns: tuple[ReasonerTurn, ...]
    position: int = 0
    contexts: list[str] | None = None

    def _next(self, context: str) -> ReasonerTurn:
        assert "available_focus_slots" in context
        if self.contexts is not None:
            self.contexts.append(context)
        turn = self.turns[self.position]
        self.position += 1
        return turn

    def plan_focus(self, context: str) -> ReasonerTurn:
        return self._next(context)

    def align(self, context: str) -> ReasonerTurn:
        return self._next(context)

    def decide(self, context: str) -> ReasonerTurn:
        return self._next(context)


class _LedgerObserver:
    def observe(self, request: FocusIntent, round_id: int, ledger: EvidenceLedger, memory):
        assert request.requirement_id == "R01" and round_id == 1
        slot = memory.by_ids(request.slot_ids)[0]
        return ((slot.video_id, ObservationBatch(
            events=(EventRecord("e1", 3.0, 4.0, "A red car enters the scene.", ("F1",), 0.9),),
            query_assessment=QueryAssessment("direct", ("e1",)),
        )),)


class _ZoomObserver:
    def observe(self, request: FocusIntent, round_id: int, ledger: EvidenceLedger, memory):
        slot = memory.by_ids(request.slot_ids)[0]
        event = EventRecord("e1", slot.time_span[0], slot.time_span[1],
                            f"direct evidence from {slot.video_id}", ("F1",), 0.9)
        batch = ObservationBatch((event,), QueryAssessment("direct", ("e1",)))
        draft = ZoomDraft(
            f"r{round_id}-{request.requirement_id}-01", slot.video_id,
            slot.time_span, slot.time_span, slot.evidence_refs, (slot.slot_id,),
        )
        return (ObservedBatch(slot.video_id, batch, draft),)


def test_warm_scan_initializes_addressable_coarse_memory() -> None:
    memory = initialize_semantic_memory(_ledger())
    assert len(memory.slots) == 2
    assert all(slot.resolution == "coarse" for slot in memory.slots)
    assert memory.slots[1].slot_id == "v2.c001"
    assert memory.slots[1].evidence_refs == ("v2.scan.r0.b1.e1",)


def test_slot_focus_executor_derives_video_and_span_from_memory(tmp_path: Path) -> None:
    video_path = tmp_path / "v2.mp4"
    video_path.touch()
    batch = ObservationBatch(
        (EventRecord("e1", 3.0, 4.0, "red car enters", ("F1",), 0.9),),
        QueryAssessment("direct", ("e1",)),
    )

    class FakeTool:
        config = SimpleNamespace(frames_per_call=16)

        def run(self, *, video, request, question, report_hints):
            assert video.video_id == "v2"
            assert request.time_span == (3.0, 4.0)
            return SimpleNamespace(observation=batch, resolved_time_span=request.time_span)

    executor = SlotFocusExecutor(
        tool_factory=FakeTool,
        videos=(VideoInput("v2", video_path),), question="Which video?",
    )
    memory = initialize_semantic_memory(_ledger())
    results = executor.observe(
        FocusIntent("R01", ("v2.c001",), "inspect red car"),
        1, _ledger(), memory,
    )
    assert len(results) == 1
    assert results[0].video_id == "v2"
    assert results[0].zoom_draft is not None
    assert results[0].zoom_draft.parent_slot_ids == ("v2.c001",)


def test_one_plan_executes_all_selected_slots_in_declared_order(tmp_path: Path) -> None:
    paths = {video_id: tmp_path / f"{video_id}.mp4" for video_id in ("v1", "v2")}
    for path in paths.values():
        path.touch()
    observed: list[str] = []

    class FakeTool:
        config = SimpleNamespace(frames_per_call=16)

        def run(self, *, video, request, question, report_hints):
            observed.append(video.video_id)
            start, end = request.time_span
            batch = ObservationBatch(
                (EventRecord("e1", start, end, f"fact from {video.video_id}", ("F1",), 0.9),),
                QueryAssessment("direct", ("e1",)),
            )
            return SimpleNamespace(observation=batch, resolved_time_span=request.time_span)

    executor = SlotFocusExecutor(
        tool_factory=FakeTool,
        videos=tuple(VideoInput(video_id, paths[video_id]) for video_id in ("v1", "v2")),
        question="Compare the evidence.",
    )
    results = executor.observe(
        FocusIntent("R01", ("v2.c001", "v1.c001"), "inspect both selected regions"),
        1, _ledger(), initialize_semantic_memory(_ledger()),
    )
    assert observed == ["v2", "v1"]
    assert [item.video_id for item in results] == ["v2", "v1"]


def test_focus_evidence_enters_ledger_then_reasoner_can_answer() -> None:
    first = ReasonerTurn((), "Need direct v2 evidence.", ReasonerAction(
        "PLAN", request=FocusIntent(
            "R01", ("v2.c001",), "Look for a red car entering.",
        ),
    ))
    proposed_answer = ReasonerTurn(
        (), "Evidence appears sufficient.",
        ReasonerAction("ANSWER", answer="B", evidence_refs=("v2.focus.r1.l1.e1",)),
    )
    result = CycleRunner(config=CycleConfig(visual_call_budget=2)).run(
        state=_state(), ledger=_ledger(),
        reasoner=_ScriptedReasoner((first, proposed_answer)),
        observer=_LedgerObserver(),
    )
    assert result.final_action.kind == "ANSWER"
    assert result.visual_calls_used == 1
    assert result.ledger.events[-1].evidence_id == "v2.focus.r1.l1.e1"
    assert result.evidence_revision == result.projected_revision == 2
    assert result.aligned_revision == 0


def test_two_stage_agenda_uses_distinct_coarse_slots_and_then_answers() -> None:
    first = ReasonerTurn((), "inspect v2", ReasonerAction(
        "PLAN", request=FocusIntent(
            "R01", ("v2.c001",), "Check v2 directly.",
            "Check the corresponding v1 event for comparison.",
        ),
    ))
    second = ReasonerTurn((), "inspect v1", ReasonerAction(
        "PLAN", request=FocusIntent(
            "R01", ("v1.c001",), "Check v1 for comparison.",
        ),
    ))
    proposed_answer = ReasonerTurn((), "both inspected", ReasonerAction(
        "ANSWER", answer="B", evidence_refs=("v2.focus.r1.l1.e1",),
    ))
    contexts: list[str] = []
    result = CycleRunner(config=CycleConfig(
        visual_call_budget=2, minimum_focus_stages=2,
    )).run(
        state=_state(), ledger=_ledger(),
        reasoner=_ScriptedReasoner(
            (first, second, proposed_answer), contexts=contexts,
        ),
        observer=_ZoomObserver(),
    )
    assert result.final_action.kind == "ANSWER"
    assert result.visual_calls_used == 2
    assert result.exploration_agenda.completed_focus_stages == 2
    assert '"remaining_focus_stages":2' in contexts[0]
    assert '"remaining_focus_stages":1' in contexts[1]
    current_ids = {slot.slot_id for slot in result.memory.slots}
    assert "v2.c001" not in current_ids
    assert "v1.c001" not in current_ids
    assert all(slot.resolution == "fine" for slot in result.memory.slots)
    assert [item.action.kind for item in result.turns] == ["PLAN", "PLAN", "ANSWER"]


def test_exploration_mode_rejects_answer_without_visual_work() -> None:
    early = ReasonerTurn((), "premature", ReasonerAction(
        "ANSWER", answer="A", evidence_refs=("v1.scan.r0.b1.e1",),
    ))
    try:
        CycleRunner(config=CycleConfig(
            visual_call_budget=2, minimum_focus_stages=2,
        )).run(
            state=_state(), ledger=_ledger(), reasoner=_ScriptedReasoner((early,)),
            observer=_ZoomObserver(),
        )
    except ValueError as exc:
        assert "exploration agenda incomplete" in str(exc)
    else:
        raise AssertionError("invalid policy action was committed as a semantic STOP")


def test_open_requirement_is_residual_uncertainty_not_an_answer_blocker() -> None:
    first = ReasonerTurn((), "inspect v2", ReasonerAction(
        "PLAN", request=FocusIntent("R01", ("v2.c001",), "Inspect v2."),
    ))
    proposed_answer = ReasonerTurn((), "best supported answer", ReasonerAction(
        "ANSWER", answer="B", evidence_refs=("v2.zoom.r1-R01-01.s01",),
    ))
    result = CycleRunner(config=CycleConfig(visual_call_budget=2)).run(
        state=_state(), ledger=_ledger(),
        reasoner=_ScriptedReasoner((first, proposed_answer)),
        observer=_ZoomObserver(),
    )
    assert result.final_action.kind == "ANSWER"
    assert result.final_action.answer == "B"
    assert result.visual_calls_used == 1
    assert result.state.open_requirements
    assert result.evidence_revision == result.projected_revision
    assert result.aligned_revision == 0


def test_runner_rejects_gap_only_repeat_after_slot_identity_changes() -> None:
    goal = "Inspect whether the red car enters."
    first = ReasonerTurn((), "inspect v2", ReasonerAction(
        "PLAN", request=FocusIntent("R01", ("v2.c001",), goal),
    ))
    repeated = ReasonerTurn((), "repeat failed inspection", ReasonerAction(
        "PLAN", request=FocusIntent(
            "R01", ("v2.zoom.r1-R01-01.s01",), goal,
        ),
    ))

    class GapObserver:
        def observe(self, request, round_id, ledger, memory):
            slot = memory.by_ids(request.slot_ids)[0]
            gap = ObservationGap(
                "g1", slot.time_span[0], slot.time_span[1], ("F1",),
                "The visual report could not be validated.",
            )
            batch = ObservationBatch(
                (), QueryAssessment(
                    "insufficient", (), ("No validated event was obtained.",),
                ), (gap,),
            )
            return (ObservedBatch(
                slot.video_id, batch,
                ZoomDraft(
                    f"r{round_id}-{request.requirement_id}-01",
                    slot.video_id, slot.time_span, slot.time_span,
                    slot.evidence_refs, (slot.slot_id,),
                ),
            ),)

    try:
        CycleRunner(config=CycleConfig(visual_call_budget=2)).run(
            state=_state(), ledger=_ledger(),
            reasoner=_ScriptedReasoner((first, repeated)),
            observer=GapObserver(),
        )
    except ValueError as exc:
        assert "repeats a prior gap-only observation request" in str(exc)
    else:
        raise AssertionError("runner accepted a repeated gap-only request under a new slot ID")


def test_replaced_parent_slot_cannot_be_selected_again() -> None:
    request = FocusIntent("R01", ("v2.c001",), "inspect v2")
    turns = (
        ReasonerTurn((), "first", ReasonerAction("PLAN", request=request)),
        ReasonerTurn((), "repeat", ReasonerAction("PLAN", request=request)),
    )
    try:
        CycleRunner(config=CycleConfig(visual_call_budget=3)).run(
            state=_state(), ledger=_ledger(), reasoner=_ScriptedReasoner(turns),
            observer=_ZoomObserver(),
        )
    except ValueError as exc:
        assert "absent from current memory" in str(exc)
    else:
        raise AssertionError("stale provenance was committed as a semantic STOP")


def test_zoom_reprojects_before_optional_alignment() -> None:
    request = FocusIntent("R01", ("v2.c001",), "inspect v2")
    contexts: list[str] = []
    turns = (
        ReasonerTurn((), "old trace", ReasonerAction("PLAN", request=request)),
        ReasonerTurn(
            (RequirementUpdate(
                "R01", "supported", ("v2.zoom.r1-R01-01.s01",),
                "ALIGN confirms the focused v2 event.", ("v2",),
            ),),
            "aligned current evidence", ReasonerAction("ALIGN", reason="audited R01"),
        ),
        ReasonerTurn((), "fresh decision", ReasonerAction(
            "ANSWER", answer="B", evidence_refs=("v2.zoom.r1-R01-01.s01",),
        )),
    )
    result = CycleRunner(config=CycleConfig(visual_call_budget=2)).run(
        state=_state(), ledger=_ledger(), reasoner=_ScriptedReasoner(turns, contexts=contexts),
        observer=_ZoomObserver(),
    )
    assert result.final_action.kind == "ANSWER"
    assert [item.action.kind for item in result.turns] == ["PLAN", "ALIGN", "ANSWER"]
    assert len(result.alignment_history) == 1
    after_focus = json.loads(contexts[1])
    assert after_focus["options"] == ["A. v1", "B. v2"]
    assert any(item[5] == "f" for item in after_focus["memory_map"])
    assert "Validated events were committed" in after_focus["last_focus_review"]["reason"]
    assert after_focus["working_state"]["current_evidence_is_projected"] is True
    assert result.state.requirements[0].objective == "ALIGN confirms the focused v2 event."


def test_context_exposes_slots_instead_of_raw_plan_coordinates() -> None:
    memory = initialize_semantic_memory(_ledger())
    text = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=memory, visual_call_budget=2,
    )
    assert "available_focus_slots" in text
    assert "v2.c001" in text
    assert "zoom_targets" not in text
    payload = json.loads(text)
    assert payload["memory_map_schema"][0:2] == ["index", "slot_id"]
    assert payload["memory_map"][0][0] == 1


def test_api_reasoner_maps_short_slot_indices_and_ignores_incremental_updates() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        agenda=ExplorationAgenda(1), visual_call_budget=2,
    )
    backend = OpenAICompatibleReasonerBackend(
        base_url="https://example.test/v1", model="flash",
    )
    backend._complete = lambda system, user: json.dumps({  # type: ignore[method-assign]
        "updates": [{
            "requirement_id": "R01", "status": "open",
            "evidence_refs": [["malformed-nested-reference"]], "summary": "bad",
        }],
        "state_summary": "inspect v2",
        "action": {"kind": "PLAN", "request": {
            "question_index": 1, "slot_indices": [2],
            "evidence_goal": "inspect v2", "next_probe_goal": "",
            "mode": "refine",
        }},
    })
    turn = backend.plan_focus(context)
    assert turn.updates == ()
    assert turn.action.request is not None
    assert turn.action.request.requirement_id == "R01"
    assert turn.action.request.slot_ids == ("v2.c001",)


def test_api_reasoner_may_request_alignment_from_ordinary_policy() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        visual_call_budget=2, evidence_revision=1, aligned_revision=0,
    )
    turn = OpenAICompatibleReasonerBackend._parse_policy_turn(json.dumps({
        "state_summary": "Cross-video evidence needs reconciliation.",
        "action": {"kind": "ALIGN", "reason": "compare current observations"},
    }), context)
    assert turn.action.kind == "ALIGN"
    assert turn.updates == ()
    working_state = json.loads(context)["working_state"]
    assert working_state["evidence_revision"] == 1
    assert working_state["projected_revision"] == 1
    assert working_state["aligned_revision"] == 0
    assert working_state["current_evidence_is_projected"] is True
    assert working_state["current_evidence_is_explicitly_aligned"] is False


def test_api_reasoner_repairs_id_copying_into_index_selection() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        agenda=ExplorationAgenda(1), visual_call_budget=2,
    )
    backend = OpenAICompatibleReasonerBackend(
        base_url="https://example.test/v1", model="flash",
    )
    candidates = iter((
        json.dumps({
            "state_summary": "copied fragile IDs",
            "action": {"kind": "PLAN", "request": {
                "requirement_id": "R01", "slot_ids": ["v2.c001"],
                "evidence_goal": "inspect v2", "mode": "refine",
            }},
        }),
        json.dumps({
            "state_summary": "selected stable rows",
            "action": {"kind": "PLAN", "request": {
                "question_index": 1, "slot_indices": [2],
                "evidence_goal": "inspect v2", "mode": "refine",
            }},
        }),
    ))
    backend._complete = lambda system, user: next(candidates)  # type: ignore[method-assign]
    turn = backend.plan_focus(context)
    assert turn.action.request is not None
    assert turn.action.request.requirement_id == "R01"
    assert turn.action.request.slot_ids == ("v2.c001",)


def test_api_reasoner_compiles_answer_evidence_indices_to_provenance() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        visual_calls_used=2, visual_call_budget=2,
    )
    payload = json.loads(context)
    selected_index, selected_ref = payload["answer_evidence_map"][0][:2]
    backend = OpenAICompatibleReasonerBackend(
        base_url="https://example.test/v1", model="flash",
    )
    backend._complete = lambda system, user: json.dumps({  # type: ignore[method-assign]
        "state_summary": "best grounded answer",
        "action": {
            "kind": "ANSWER", "answer": "B",
            "evidence_indices": [selected_index],
        },
    })
    turn = backend.decide(context)
    assert turn.action.evidence_refs == (selected_ref,)


def test_api_reasoner_allows_answer_with_residual_questions_and_budget() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        visual_calls_used=0, visual_call_budget=2,
    )
    payload = json.loads(context)
    assert payload["open_questions"]
    assert payload["exploration_agenda"]["visual_calls_remaining"] == 2
    assert payload["last_focus_review"]["decision_eligible"] is True
    selected_index, selected_ref = payload["answer_evidence_map"][0][:2]
    turn = OpenAICompatibleReasonerBackend._parse_policy_turn(json.dumps({
        "state_summary": "Further observation is unlikely to change the best-supported choice.",
        "action": {
            "kind": "ANSWER", "answer": "B",
            "evidence_indices": [selected_index],
        },
    }), context)
    assert turn.action.kind == "ANSWER"
    assert turn.action.evidence_refs == (selected_ref,)


def test_api_policy_repairs_an_exact_repeat_of_gap_only_observation() -> None:
    memory = initialize_semantic_memory(_ledger())
    targets = (("v2", 3.0, 4.0),)
    goal = "Inspect whether the red car enters."
    receipt = ObservationReceipt(
        round_id=1,
        request_fingerprint=_observation_fingerprint(goal, targets),
        requirement_id="R01",
        evidence_goal=goal,
        targets=targets,
        event_refs=(),
        gap_refs=("v2.focus.r1.l1.g1",),
    )
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=memory,
        visual_calls_used=1, visual_call_budget=3,
        observation_receipts=(receipt,),
    )
    payload = json.loads(context)
    v2_slot_index = next(row[0] for row in payload["memory_map"] if row[2] == "v2")
    evidence_index = payload["answer_evidence_map"][0][0]
    candidates = iter((
        json.dumps({
            "state_summary": "repeat the failed request",
            "action": {"kind": "PLAN", "request": {
                "question_index": 1, "slot_indices": [v2_slot_index],
                "evidence_goal": goal, "mode": "refine",
            }},
        }),
        json.dumps({
            "state_summary": "the repeated request has no acquisition value",
            "action": {"kind": "ANSWER", "answer": "B",
                       "evidence_indices": [evidence_index]},
        }),
    ))
    backend = OpenAICompatibleReasonerBackend(
        base_url="https://example.test/v1", model="flash",
    )
    backend._complete = lambda system, user: next(candidates)  # type: ignore[method-assign]
    turn = backend.decide(context)
    assert turn.action.kind == "ANSWER"
    assert payload["observation_receipts"] == [{
        "round": 1,
        "request_fingerprint": receipt.request_fingerprint,
        "requirement": "R01",
        "goal": goal,
        "targets": [{"video": "v2", "time": [3.0, 4.0]}],
        "outcome": "gap_only",
        "event_count": 0,
        "gap_count": 1,
    }]


def test_api_policy_repairs_plan_when_visual_budget_is_exhausted() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        visual_calls_used=2, visual_call_budget=2,
    )
    payload = json.loads(context)
    evidence_index = payload["answer_evidence_map"][0][0]
    candidates = iter((
        json.dumps({
            "state_summary": "try another observation",
            "action": {"kind": "PLAN", "request": {
                "question_index": 1, "slot_indices": [1],
                "evidence_goal": "inspect again", "mode": "refine",
            }},
        }),
        json.dumps({
            "state_summary": "best supported answer at budget limit",
            "action": {"kind": "ANSWER", "answer": "B",
                       "evidence_indices": [evidence_index]},
        }),
    ))
    backend = OpenAICompatibleReasonerBackend(
        base_url="https://example.test/v1", model="flash",
    )
    backend._complete = lambda system, user: next(candidates)  # type: ignore[method-assign]
    turn = backend.decide(context)
    assert turn.action.kind == "ANSWER"


def test_api_reasoner_rejects_answer_provenance_strings() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        visual_calls_used=2, visual_call_budget=2,
    )
    candidate = json.dumps({
        "state_summary": "copied ID",
        "action": {
            "kind": "ANSWER", "answer": "B",
            "evidence_refs": ["v1.c001"],
        },
    })
    try:
        OpenAICompatibleReasonerBackend._parse_policy_turn(candidate, context)
    except ValueError as exc:
        assert "evidence_indices" in str(exc)
    else:
        raise AssertionError("API reasoner accepted a model-copied provenance string")


def test_api_reasoner_rebuilds_generic_synthesis_through_indices() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        visual_call_budget=2,
    )
    payload = json.loads(context)
    v2_index, v2_ref = next(
        (row[0], row[1]) for row in payload["answer_evidence_map"]
        if row[2] == "v2"
    )
    backend = OpenAICompatibleReasonerBackend(
        base_url="https://example.test/v1", model="flash",
    )
    backend._complete = lambda system, user: json.dumps({  # type: ignore[method-assign]
        "synthesis": [{
            "kind": "finding", "videos": ["v2"],
            "evidence_indices": [v2_index],
            "statement": "Current v2 evidence directly shows the event.",
        }],
        "state_summary": "aligned v2 evidence",
        "action": {"kind": "ALIGN", "reason": "audited the current requirement"},
    })
    turn = backend.align(context)
    assert turn.action.kind == "ALIGN"
    assert turn.updates[0].requirement_id == "F01"
    assert turn.updates[0].evidence_refs == (v2_ref,)
    assert turn.updates[0].video_ids == ("v2",)


def test_api_align_repairs_missing_open_question_during_mandatory_exploration() -> None:
    context = ContextBuilder(token_budget=4096).render(
        _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        agenda=ExplorationAgenda(2, completed_focus_stages=1),
        visual_call_budget=2,
    )
    payload = json.loads(context)
    evidence_index = payload["answer_evidence_map"][0][0]
    candidates = iter((
        json.dumps({
            "synthesis": [{
                "kind": "finding", "statement": "A local fact.",
                "videos": ["v1"], "evidence_indices": [evidence_index],
            }],
            "state_summary": "Further exploration is still needed.",
            "action": {"kind": "ALIGN", "reason": "refresh evidence"},
        }),
        json.dumps({
            "synthesis": [
                {"kind": "finding", "statement": "A local fact.",
                 "videos": ["v1"], "evidence_indices": [evidence_index]},
                {"kind": "open_question", "statement": "Which observation settles the decision?",
                 "videos": ["v1", "v2"], "evidence_indices": [evidence_index]},
            ],
            "state_summary": "One decision-critical question remains.",
            "action": {"kind": "ALIGN", "reason": "refresh evidence"},
        }),
    ))
    backend = OpenAICompatibleReasonerBackend(
        base_url="https://example.test/v1", model="flash",
    )
    backend._complete = lambda system, user: next(candidates)  # type: ignore[method-assign]
    turn = backend.align(context)
    assert [item.status for item in turn.updates] == ["supported", "open"]


def test_reasoner_selects_align_and_runtime_invokes_dedicated_audit() -> None:
    calls: list[str] = []
    requested = ReasonerTurn(
        (), "comparison needed", ReasonerAction("ALIGN", reason="compare evidence"),
    )
    audited = ReasonerTurn(
        (RequirementUpdate(
            "R01", "open", ("v2.scan.r0.b1.e1",),
            "The evidence remains inconclusive.", ("v2",),
        ),),
        "audit complete", ReasonerAction("ALIGN", reason="evidence compared"),
    )

    class Reasoner:
        def decide(self, context):
            calls.append("decide")
            return requested

        def align(self, context):
            calls.append("align")
            return audited

        def plan_focus(self, context):
            raise AssertionError("runner must not force the PLAN endpoint")

    runner = CycleRunner(config=CycleConfig(visual_call_budget=2))
    turn = runner._decide_validated(
        ContextBuilder(token_budget=4096).render(
            _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
            visual_call_budget=2, evidence_revision=1, aligned_revision=0,
        ),
        _state(), _ledger(), Reasoner(), initialize_semantic_memory(_ledger()),
        evidence_revision=1, aligned_revision=0,
    )
    assert calls == ["decide", "align"]
    assert turn is audited


def test_current_projected_evidence_can_answer_without_alignment() -> None:
    calls: list[str] = []
    answer = ReasonerTurn(
        (), "ready", ReasonerAction(
            "ANSWER", answer="B", evidence_refs=("v2.scan.r0.b1.e1",),
        ),
    )
    audited = ReasonerTurn(
        (RequirementUpdate(
            "R01", "open", ("v2.scan.r0.b1.e1",),
            "The answer proposal exposed unresolved evidence.", ("v2",),
        ),),
        "audit complete", ReasonerAction("ALIGN", reason="pre-answer audit"),
    )

    class Reasoner:
        def decide(self, context):
            calls.append("decide")
            return answer

        def align(self, context):
            calls.append("align")
            return audited

        def plan_focus(self, context):
            raise AssertionError("runner must not force the PLAN endpoint")

    runner = CycleRunner(config=CycleConfig(visual_call_budget=2))
    turn = runner._decide_validated(
        ContextBuilder(token_budget=4096).render(
            _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
            visual_call_budget=2, evidence_revision=1, aligned_revision=0,
        ),
        _state(), _ledger(), Reasoner(), initialize_semantic_memory(_ledger()),
        evidence_revision=1, aligned_revision=0,
    )
    assert calls == ["decide"]
    assert turn.action.kind == "ANSWER"


def test_align_does_not_impose_all_video_coverage_on_a_local_finding() -> None:
    state = DecisionState(
        question="Which comes first?", video_ids=("v1", "v2"),
        requirements=(EvidenceRequirement(
            "R01", "Compare the two observed states.", ("v1", "v2"),
        ),),
    )
    turn = ReasonerTurn(
        (RequirementUpdate(
            "R01", "supported", ("v1.scan.r0.b1.e1",),
            "A local finding about v1.", ("v1", "v2"),
        ),),
        "insufficient alignment", ReasonerAction("ALIGN", reason="compare videos"),
    )
    CycleRunner(config=CycleConfig(visual_call_budget=2))._preflight_turn(
        state, _ledger(), turn, initialize_semantic_memory(_ledger()),
    )


def test_align_retains_work_for_mandatory_follow_up_stage() -> None:
    turn = ReasonerTurn(
        (RequirementUpdate(
            "R01", "supported", ("v2.scan.r0.b1.e1",),
            "The current requirement was settled.", ("v2",),
        ),),
        "closed too early", ReasonerAction("ALIGN", reason="audited R01"),
    )
    try:
        CycleRunner(config=CycleConfig(
            visual_call_budget=2, minimum_focus_stages=2,
        ))._preflight_turn(
            _state(), _ledger(), turn, initialize_semantic_memory(_ledger()),
            ExplorationAgenda(2, completed_focus_stages=1), visual_calls_used=1,
        )
    except ValueError as exc:
        assert "retain an open question" in str(exc)
    else:
        raise AssertionError("ALIGN bypassed a mandatory follow-up stage")


def test_context_budget_refuses_unbounded_state() -> None:
    try:
        ContextBuilder(token_budget=3).render(
            _state(), _ledger(), (), memory=initialize_semantic_memory(_ledger()),
        )
    except ValueError as exc:
        assert "context budget exceeded" in str(exc)
    else:
        raise AssertionError("oversized reasoner context was silently accepted")


def test_context_packs_large_reference_sets_within_actual_budget() -> None:
    ledger = EvidenceLedger(events=tuple(
        LedgerEvent(
            evidence_id=f"v{video}.scan.r0.b1.e{index}", source_event_id=f"e{index}",
            video_id=f"v{video}", time_span=(float(index), float(index + 1)),
            description=("visible action with deliberately verbose detail " * 8),
            uncertainty="", frame_ids=(f"F{index}",), confidence=0.9,
            action="SCAN", round_id=0,
        )
        for video in (1, 2, 3) for index in range(1, 11)
    ))
    refs = tuple(item.evidence_id for item in ledger.events)
    state = DecisionState(
        question="What differs across the three videos?", video_ids=("v1", "v2", "v3"),
        requirements=(EvidenceRequirement(
            "R01", "Compare the visible actions.", ("v1", "v2", "v3"),
            evidence_refs=refs,
        ),),
        summary="Compare all currently visible actions.",
    )
    builder = ContextBuilder(token_budget=4096)
    text = builder.render(state, ledger, (), memory=initialize_semantic_memory(ledger))
    assert builder.token_counter(text) <= 4096
    assert '"available_focus_slots"' in text
    assert len(json.loads(text)["memory_map"]) == 30


def test_snapshot_reference_set_includes_slot_and_nested_provenance() -> None:
    refs = _snapshot_reference_ids((
        {"id": "v1.zoom.r1.s01", "evidence": ("v1.focus.r1.l1.e1",)},
    ))
    assert refs == {"v1.zoom.r1.s01", "v1.focus.r1.l1.e1"}


def test_projected_review_can_resolve_requirements_with_current_evidence() -> None:
    state = _parse_projected_state(
        json.dumps({
            "requirements": [{
                "id": "R01", "objective": "The observed event is present in v2.",
                "videos": ["v2"], "status": "supported",
                "evidence_indices": [1],
            }],
            "frontier": [],
            "review": {"outcome": "resolved", "reason": "The event is directly visible.",
                       "evidence_indices": [1]},
        }),
        question="Which video?", options=("A. v1", "B. v2"),
        video_ids=("v1", "v2"), evidence_catalogue=("v2.zoom.r1.s01",),
        required_review_refs={"v2.zoom.r1.s01"},
        required_event_review_refs={"v2.zoom.r1.s01"},
    )
    assert not state.open_requirements
    assert state.review_outcome == "resolved"


def test_mandatory_follow_up_projection_cannot_claim_early_readiness() -> None:
    payload = json.dumps({
        "requirements": [{
            "id": "R01", "objective": "The observed event is present in v2.",
            "videos": ["v2"], "status": "supported",
            "evidence_indices": [1],
        }],
        "frontier": [],
        "review": {"outcome": "resolved", "reason": "direct observation",
                   "evidence_indices": [1]},
    })
    try:
        _parse_projected_state(
            payload, question="Which video?", options=(),
            video_ids=("v1", "v2"), evidence_catalogue=("v2.zoom.r1.s01",),
            force_open_requirement=True,
        )
    except ValueError as exc:
        assert "mandatory follow-up" in str(exc)
    else:
        raise AssertionError("mandatory review was bypassed by an early ready projection")


def test_resolved_review_must_use_last_focus_evidence() -> None:
    payload = json.dumps({
        "requirements": [{
            "id": "R01", "objective": "The event is present.",
            "videos": ["v2"], "status": "supported",
            "evidence_indices": [1],
        }],
        "frontier": [],
        "review": {"outcome": "resolved", "reason": "claimed resolution",
                   "evidence_indices": [1]},
    })
    try:
        _parse_projected_state(
            payload, question="Which video?", options=(),
            video_ids=("v1", "v2"),
            evidence_catalogue=("v2.scan.e1", "v2.focus.r1.e1"),
            required_review_refs={"v2.focus.r1.e1"},
            required_event_review_refs={"v2.focus.r1.e1"},
        )
    except ValueError as exc:
        assert "last Focus" in str(exc)
    else:
        raise AssertionError("stale evidence was accepted as resolution from the last Focus")


def test_review_resolution_is_grounded_independently_of_requirement_status() -> None:
    state = _parse_projected_state(
        json.dumps({
            "requirements": [{
                "id": "R01", "objective": "Inspect the newly exposed ambiguity.",
                "videos": ["v2"], "status": "open",
                "evidence_indices": [1],
            }],
            "frontier": [{
                "id": "P01", "requirement_id": "R01", "videos": ["v2"],
                "observable": "Inspect the newly exposed local state.",
                "purpose": "Determine whether the remaining ambiguity changes the decision.",
            }],
            "review": {
                "outcome": "resolved", "reason": "The local state is directly visible.",
                "evidence_indices": [1],
            },
        }),
        question="Which video?", options=(), video_ids=("v1", "v2"),
        evidence_catalogue=("v2.zoom.r1.s01",),
        required_review_refs={"v2.zoom.r1.s01"},
        required_event_review_refs={"v2.zoom.r1.s01"},
    )
    assert state.open_requirements
    assert state.review_evidence_refs == ("v2.zoom.r1.s01",)


def test_gap_only_focus_cannot_make_projection_answer_ready() -> None:
    payload = json.dumps({
        "requirements": [{
            "id": "R01", "objective": "Claim complete despite a gap.",
            "videos": ["v2"], "status": "supported", "evidence_indices": [1],
        }],
        "frontier": [],
        "review": {
            "outcome": "no_gain", "reason": "The observation was invalid.",
            "evidence_indices": [1],
        },
    })
    try:
        _parse_projected_state(
            payload, question="Which video?", options=(), video_ids=("v1", "v2"),
            evidence_catalogue=("v2.zoom.r1.s01",),
            required_review_refs={"v2.zoom.r1.s01"},
            required_event_review_refs=set(),
        )
    except ValueError as exc:
        assert "observation gap" in str(exc)
    else:
        raise AssertionError("a gap-only Focus made the state answer-ready")


def test_nonresolved_review_does_not_mechanically_block_readiness() -> None:
    payload = json.dumps({
        "requirements": [{
            "id": "R01", "objective": "Prematurely closed comparison.",
            "videos": ["v2"], "status": "supported", "evidence_indices": [1],
        }],
        "frontier": [],
        "review": {
            "outcome": "partial", "reason": "One ambiguity remains.",
            "evidence_indices": [1],
        },
    })
    state = _parse_projected_state(
        payload, question="Which video?", options=(), video_ids=("v1", "v2"),
        evidence_catalogue=("v2.focus.r1.e1",),
        visual_calls_remaining=1,
        required_review_refs={"v2.focus.r1.e1"},
        required_event_review_refs={"v2.focus.r1.e1"},
    )
    assert not state.open_requirements
    assert state.review_outcome == "partial"


def test_projector_exposes_only_indices_and_compiles_them_back_to_refs() -> None:
    snapshot = (
        {
            "id": "__memory_index__", "kind": "memory_index",
            "videos": {"v2": [["v2.zoom.r1.s01", 3.0, 4.0, "f"]]},
        },
        {
            "id": "v2.zoom.r1.s01", "kind": "fine", "video": "v2",
            "time": (3.0, 4.0), "content": "A red car enters.",
            "uncertainty": (), "evidence": ("v2.focus.r1.l1.e1",),
        },
    )
    captured = {}

    def complete(*, role, messages, config):
        captured["payload"] = json.loads(messages[-1]["content"])
        return json.dumps({
            "requirements": [{
                "id": "R01", "objective": "The event is visible.",
                "videos": ["v2"], "status": "supported",
                "evidence_indices": [1],
            }],
            "frontier": [],
            "review": {
                "outcome": "resolved", "reason": "The event is visible.",
                "evidence_indices": [2],
            },
        })

    projector = OpenAICompatibleStateProjector(
        base_url="https://example.test/v1", model="flash",
    )
    projector._transport = SimpleNamespace(complete=complete)
    state = projector.project(
        question="Which video?", options=(), video_ids=("v1", "v2"),
        snapshot=snapshot,
        receipt=FocusReviewReceipt(
            requirement_id="R01", evidence_goal="Inspect the event.",
            expected_complementary_probe="", proposed_next_probe="",
            reserved_follow_up_probe="",
            new_event_refs=("v2.focus.r1.l1.e1",), new_gap_refs=(),
            mandatory_stages_remaining=0, visual_calls_remaining_after_focus=0,
            force_open_requirement=False,
        ),
    )
    detail = captured["payload"]["current_evidence"][1]
    assert "id" not in detail and "evidence" not in detail
    assert detail["citation_index"] == 1
    assert detail["source_evidence_indices"] == [2]
    assert state.requirements[0].evidence_refs == ("v2.zoom.r1.s01",)
    assert state.review_evidence_refs == ("v2.focus.r1.l1.e1",)


def test_complementary_probe_reservation_expires_after_follow_up_stage() -> None:
    request = FocusIntent(
        "R01", ("v2.c001",), "Inspect v2.", "Countercheck v1.",
    )
    after_first = ExplorationAgenda(2).reserve_from(request).complete_stage()
    assert after_first.reservation is not None
    after_second = after_first.complete_stage()
    assert after_second.reservation is None


def test_reasoner_turn_parses_slot_level_focus_intent() -> None:
    valid = parse_reasoner_turn(
        '{"updates":[],"state_summary":"need evidence","action":{"kind":"PLAN","request":{"requirement_id":"R01","slot_ids":["v2.c001"],"evidence_goal":"inspect entry","next_probe_goal":"compare v1","mode":"refine"}}}'
    )
    assert valid.action.request is not None
    assert valid.action.request.slot_ids == ("v2.c001",)
    try:
        parse_reasoner_turn('{"updates":[],"state_summary":"bad","action":{"kind":"PLAN"}}')
    except ValueError as exc:
        assert "PLAN requires request" in str(exc)
    else:
        raise AssertionError("incomplete PLAN was accepted")


def test_reasoner_turn_rejects_nested_evidence_references_at_parse_boundary() -> None:
    malformed = json.dumps({
        "updates": [{
            "requirement_id": "R01",
            "status": "open",
            "evidence_refs": [["v1.scan.e1"]],
            "summary": "still open",
        }],
        "state_summary": "need evidence",
        "action": {"kind": "STOP", "reason": "cannot continue"},
    })
    try:
        parse_reasoner_turn(malformed)
    except ValueError as exc:
        assert "evidence_refs must contain only nonempty strings" in str(exc)
    else:
        raise AssertionError("nested evidence reference escaped structural validation")


def test_recover_mode_addresses_only_unobserved_coverage_slots(tmp_path: Path) -> None:
    video_path = tmp_path / "v1.mp4"
    video_path.touch()
    batch = ObservationBatch(
        (EventRecord("e1", 5.0, 10.0, "recovered action", ("F1",), 0.9),),
        QueryAssessment("direct", ("e1",)),
    )

    class FakeTool:
        config = SimpleNamespace(frames_per_call=16)

        def run(self, *, video, request, question, report_hints):
            assert request.time_span == (5.0, 10.0)
            assert "Observe only v1" in question
            return SimpleNamespace(observation=batch, resolved_time_span=request.time_span)

    memory = initialize_semantic_memory(
        _ledger(), coverage_regions={"v1": ((5.0, 10.0),)},
    )
    executor = SlotFocusExecutor(
        tool_factory=FakeTool, videos=(VideoInput("v1", video_path),),
        question="global cross-video question",
    )
    result = executor.observe(
        FocusIntent("R01", ("v1.u001",), "look for omitted action", mode="recover"),
        1, _ledger(), memory,
    )
    assert len(result) == 1
    try:
        executor.observe(
            FocusIntent("R01", ("v1.c001",), "bad recovery", mode="recover"),
            1, _ledger(), memory,
        )
    except ValueError as exc:
        assert "RECOVER" in str(exc)
    else:
        raise AssertionError("recover mode accepted an observed slot")


def test_current_fine_slot_can_be_refined_again() -> None:
    first = ReasonerTurn((), "first", ReasonerAction(
        "PLAN", request=FocusIntent("R01", ("v2.c001",), "inspect v2"),
    ))
    second = ReasonerTurn((), "second", ReasonerAction(
        "PLAN", request=FocusIntent(
            "R01", ("v2.zoom.r1-R01-01.s01",), "inspect transition boundary",
        ),
    ))
    proposed_answer = ReasonerTurn(
        (), "ready", ReasonerAction(
            "ANSWER", answer="B", evidence_refs=("v2.zoom.r2-R01-01.s01",),
        ),
    )
    result = CycleRunner(config=CycleConfig(visual_call_budget=3)).run(
        state=_state(), ledger=_ledger(),
        reasoner=_ScriptedReasoner(
            (first, second, proposed_answer),
        ),
        observer=_ZoomObserver(),
    )
    assert result.final_action.kind == "ANSWER"
    assert result.visual_calls_used == 2
    assert any(slot.revision == 2 for slot in result.memory.slots)
    assert [item.action.kind for item in result.turns] == ["PLAN", "PLAN", "ANSWER"]


def test_plan_compiler_reduces_slots_to_declared_stage_budget() -> None:
    memory = initialize_semantic_memory(_ledger())
    turn = ReasonerTurn((), "inspect both", ReasonerAction(
        "PLAN", request=FocusIntent(
            "R01", ("v1.c001", "v2.c001"), "inspect both locally",
        ),
    ))
    runner = CycleRunner(config=CycleConfig(
        visual_call_budget=2, minimum_focus_stages=2,
    ))
    compiled = runner._compile_plan_to_budget(
        turn, memory, ExplorationAgenda(2), 0,
    )
    assert compiled.action.request is not None
    assert compiled.action.request.slot_ids == ("v1.c001",)
    assert "plan_compiler" in compiled.action.reason


def test_plan_compiler_normalizes_refine_recover_from_slot_level() -> None:
    memory = initialize_semantic_memory(
        _ledger(), coverage_regions={"v1": ((5.0, 7.0),)},
    )
    turn = ReasonerTurn((), "recover omitted span", ReasonerAction(
        "PLAN", request=FocusIntent(
            "R01", ("v1.u001",), "inspect omitted span", mode="refine",
        ),
    ))
    compiled = CycleRunner(config=CycleConfig(visual_call_budget=2))._compile_plan_to_budget(
        turn, memory, ExplorationAgenda(0), 0,
    )
    assert compiled.action.request is not None
    assert compiled.action.request.mode == "recover"
    assert "normalized mode refine->recover" in compiled.action.reason


def test_sequence_answer_contract_rejects_slot_ids() -> None:
    state = DecisionState(
        question="Order the clips.", video_ids=("v1", "v2"),
        requirements=(EvidenceRequirement("R01", "inspect", ("v1",)),),
        answer_kind="sequence", required_answer_ids=("v1", "v2"),
    )
    bad = ReasonerAction(
        "ANSWER", answer="v1.c001->v2.c001",
        evidence_refs=("v1.scan.r0.b1.e1",),
    )
    try:
        CycleRunner._validate_answer_contract(state, bad)
    except ValueError as exc:
        assert "every required video ID" in str(exc)
    else:
        raise AssertionError("slot IDs were accepted as a sequence answer")
