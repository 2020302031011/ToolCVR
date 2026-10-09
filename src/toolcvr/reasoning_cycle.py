"""Evidence-led, budgeted Reasoner/Observer execution without a round limit.

This module deliberately has no task branches and no third agent.  The
Reasoner decides whether to ALIGN, PLAN, ANSWER or STOP; the Observer executes
the visual request produced by PLAN.  The runner only validates, persists the
evidence meaning, accounts for budget, and prevents repeated zero-gain loops.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from toolcvr.openai_compatible import (
    APICallRecord,
    APIRequestConfig,
    OpenAICompatibleTransport,
    extract_json_object,
)
from toolcvr.schemas import ObservationBatch
from toolcvr.focus import FocusRequest, FocusResult, FocusTool, split_complete_zoom
from toolcvr.schemas import VideoInput
from toolcvr.semantic_zoom import SemanticZoomMemory, ZoomDraft, ZoomSlot


RequirementStatus = Literal["open", "supported", "contradicted"]
ReviewOutcome = Literal["initial", "resolved", "partial", "no_gain", "conflict"]
ActionKind = Literal["ALIGN", "PLAN", "ANSWER", "STOP"]
AnswerKind = Literal["choice", "sequence", "interval", "free_text"]


def _validate_identifier_tuple(
    values: tuple[str, ...],
    field: str,
    *,
    allow_empty: bool = True,
) -> None:
    """Enforce JSON identifier leaves before they reach set membership.

    A nested array is neither an identifier nor safely hashable.  Keeping this
    invariant on the value objects makes every backend obey the same boundary,
    rather than relying on whichever execution path happens to use the field
    first.
    """
    if not isinstance(values, tuple):
        raise ValueError(f"{field} must be a tuple of strings")
    if not allow_empty and not values:
        raise ValueError(f"{field} must be nonempty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field} must contain only nonempty strings")


@dataclass(frozen=True)
class LedgerEvent:
    evidence_id: str
    source_event_id: str
    video_id: str
    time_span: tuple[float, float]
    description: str
    uncertainty: str
    frame_ids: tuple[str, ...]
    confidence: float
    action: str
    round_id: int
    interpretation: str = ""


@dataclass(frozen=True)
class LedgerGap:
    gap_id: str
    video_id: str
    time_span: tuple[float, float]
    frame_ids: tuple[str, ...]
    reason: str
    action: str
    round_id: int


@dataclass(frozen=True)
class EvidenceLedger:
    """Append-only source of truth; DecisionState stores only references."""

    events: tuple[LedgerEvent, ...] = ()
    gaps: tuple[LedgerGap, ...] = ()

    def __post_init__(self) -> None:
        ids = [item.evidence_id for item in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("ledger evidence IDs must be unique")
        gap_ids = [item.gap_id for item in self.gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("ledger gap IDs must be unique")

    def append(
        self,
        *,
        action: str,
        round_id: int,
        observation_id: str,
        video_id: str,
        batch: ObservationBatch,
    ) -> tuple["EvidenceLedger", tuple[str, ...], tuple[str, ...]]:
        if not action.strip() or not observation_id.strip() or round_id <= 0:
            raise ValueError("ledger append requires action, observation_id and positive round")
        prefix = f"{video_id}.{action.lower()}.r{round_id}.{observation_id}"
        known = {item.evidence_id for item in self.events}
        additions: list[LedgerEvent] = []
        event_ids: list[str] = []
        for position, event in enumerate(batch.events, start=1):
            evidence_id = f"{prefix}.e{position}"
            if evidence_id in known:
                raise ValueError(f"duplicate observation append: {evidence_id}")
            known.add(evidence_id)
            event_ids.append(evidence_id)
            additions.append(LedgerEvent(
                evidence_id=evidence_id, source_event_id=event.event_id, video_id=video_id,
                time_span=(event.start_seconds, event.end_seconds), description=event.description,
                uncertainty=event.uncertainty, frame_ids=event.evidence_frame_ids,
                confidence=event.confidence, action=action, round_id=round_id,
                interpretation=event.interpretation,
            ))
        gap_additions: list[LedgerGap] = []
        gap_ids: list[str] = []
        known_gaps = {item.gap_id for item in self.gaps}
        for position, gap in enumerate(batch.observation_gaps, start=1):
            gap_id = f"{prefix}.g{position}"
            if gap_id in known_gaps:
                raise ValueError(f"duplicate observation gap append: {gap_id}")
            known_gaps.add(gap_id)
            gap_ids.append(gap_id)
            gap_additions.append(LedgerGap(
                gap_id=gap_id, video_id=video_id,
                time_span=(gap.start_seconds, gap.end_seconds), frame_ids=gap.frame_ids,
                reason=gap.reason, action=action, round_id=round_id,
            ))
        return replace(self, events=(*self.events, *additions), gaps=(*self.gaps, *gap_additions)), tuple(event_ids), tuple(gap_ids)

    def by_ids(self, ids: Sequence[str]) -> tuple[LedgerEvent, ...]:
        by_id = {item.evidence_id: item for item in self.events}
        missing = [item for item in ids if item not in by_id]
        if missing:
            raise ValueError(f"state cites evidence absent from ledger: {missing}")
        return tuple(by_id[item] for item in ids)


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    objective: str
    video_ids: tuple[str, ...]
    status: RequirementStatus = "open"
    evidence_refs: tuple[str, ...] = ()
    attempts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.requirement_id.strip() or not self.objective.strip() or not self.video_ids:
            raise ValueError("requirement needs id, objective and video scope")
        _validate_identifier_tuple(self.video_ids, "requirement video_ids", allow_empty=False)
        _validate_identifier_tuple(self.evidence_refs, "requirement evidence_refs")
        if self.status not in ("open", "supported", "contradicted"):
            raise ValueError(f"unknown requirement status: {self.status}")


@dataclass(frozen=True)
class AnswerHypothesis:
    hypothesis_id: str
    text: str
    status: Literal["viable", "eliminated", "selected"] = "viable"
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateObservation:
    """An answer-free, visually testable question emitted by initial formation."""

    candidate_id: str
    target_video_ids: tuple[str, ...]
    observable: str
    purpose: str
    related_requirement_id: str

    def __post_init__(self) -> None:
        if (not self.candidate_id.strip() or not self.target_video_ids
                or not self.observable.strip() or not self.related_requirement_id.strip()):
            raise ValueError("candidate observation needs id, videos, observable and related requirement")


@dataclass(frozen=True)
class DecisionState:
    question: str
    video_ids: tuple[str, ...]
    requirements: tuple[EvidenceRequirement, ...]
    options: tuple[str, ...] = ()
    hypotheses: tuple[AnswerHypothesis, ...] = ()
    candidate_observations: tuple[CandidateObservation, ...] = ()
    summary: str = ""
    round_id: int = 0
    action_history: tuple[str, ...] = ()
    protocol_notes: tuple[str, ...] = ()
    answer_kind: AnswerKind | None = None
    required_answer_ids: tuple[str, ...] = ()
    review_outcome: ReviewOutcome = "initial"
    review_reason: str = "Initial evidence state has not yet been reviewed."
    review_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.video_ids:
            raise ValueError("decision state needs question and videos")
        if len({item.requirement_id for item in self.requirements}) != len(self.requirements):
            raise ValueError("decision-state requirement IDs must be unique")
        if len({item.hypothesis_id for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("decision-state hypothesis IDs must be unique")
        if len({item.candidate_id for item in self.candidate_observations}) != len(self.candidate_observations):
            raise ValueError("decision-state candidate observation IDs must be unique")
        requirement_ids = {item.requirement_id for item in self.requirements}
        if any(item.related_requirement_id not in requirement_ids for item in self.candidate_observations):
            raise ValueError("candidate observation references an unknown requirement")
        if any(video not in self.video_ids for item in self.candidate_observations for video in item.target_video_ids):
            raise ValueError("candidate observation references an unknown video")
        if any(not item.strip() for item in self.protocol_notes):
            raise ValueError("inference protocol notes must be nonempty")
        if self.answer_kind not in (None, "choice", "sequence", "interval", "free_text"):
            raise ValueError(f"unknown answer kind: {self.answer_kind}")
        if len(self.required_answer_ids) != len(set(self.required_answer_ids)):
            raise ValueError("required answer IDs must be unique")
        if any(item not in self.video_ids for item in self.required_answer_ids):
            raise ValueError("required answer IDs must name sample videos")
        if self.review_outcome not in ("initial", "resolved", "partial", "no_gain", "conflict"):
            raise ValueError(f"unknown review outcome: {self.review_outcome}")
        if not self.review_reason.strip():
            raise ValueError("decision state review reason must be nonempty")
        _validate_identifier_tuple(
            self.review_evidence_refs, "decision state review evidence_refs"
        )

    @property
    def open_requirements(self) -> tuple[EvidenceRequirement, ...]:
        return tuple(item for item in self.requirements if item.status == "open")


@dataclass(frozen=True)
class RequirementUpdate:
    """One item in a complete, answer-free evidence synthesis.

    Despite the historical name, ALIGN outputs are not patches to the prior
    state. ``supported``/``contradicted`` items are grounded findings and
    ``open`` items are decision-critical questions. The runner replaces the
    previous synthesis atomically.
    """

    requirement_id: str
    status: RequirementStatus
    evidence_refs: tuple[str, ...]
    summary: str
    video_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.requirement_id, str) or not self.requirement_id.strip():
            raise ValueError("requirement update needs a nonempty requirement_id")
        if self.status not in ("open", "supported", "contradicted"):
            raise ValueError("requirement update has an invalid status")
        _validate_identifier_tuple(self.evidence_refs, "requirement update evidence_refs")
        _validate_identifier_tuple(self.video_ids, "requirement update video_ids")
        if not isinstance(self.summary, str):
            raise ValueError("requirement update summary must be a string")


@dataclass(frozen=True)
class AlignmentRecord:
    """Append-only audit record of one Reasoner-owned ALIGN judgment."""

    round_id: int
    evidence_revision: int
    requirement_id: str
    objective: str
    video_ids: tuple[str, ...]
    status: RequirementStatus
    evidence_refs: tuple[str, ...]
    summary: str

    def __post_init__(self) -> None:
        if self.round_id <= 0 or self.evidence_revision <= 0 or not self.requirement_id.strip():
            raise ValueError(
                "alignment record needs positive round, evidence revision and requirement ID"
            )
        if not self.objective.strip() or not self.video_ids or not self.summary.strip():
            raise ValueError("alignment record needs objective, video scope and summary")
        if self.status not in ("open", "supported", "contradicted"):
            raise ValueError("alignment record has invalid status")
        _validate_identifier_tuple(self.video_ids, "alignment record video_ids", allow_empty=False)
        _validate_identifier_tuple(self.evidence_refs, "alignment record evidence_refs")


@dataclass(frozen=True)
class FocusIntent:
    """A semantic PLAN over current memory slots, never raw coordinates."""

    requirement_id: str
    slot_ids: tuple[str, ...]
    evidence_goal: str
    next_probe_goal: str = ""
    mode: Literal["refine", "recover"] = "refine"

    def __post_init__(self) -> None:
        if not self.requirement_id.strip() or not self.slot_ids or not self.evidence_goal.strip():
            raise ValueError("focus intent needs requirement, slots and evidence goal")
        _validate_identifier_tuple(self.slot_ids, "focus intent slot_ids", allow_empty=False)
        if len(self.slot_ids) != len(set(self.slot_ids)):
            raise ValueError("focus intent slot IDs must not repeat")
        if self.mode not in ("refine", "recover"):
            raise ValueError("focus intent mode must be refine or recover")


@dataclass(frozen=True)
class ReasonerAction:
    kind: ActionKind
    request: FocusIntent | None = None
    answer: str | None = None
    evidence_refs: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("ALIGN", "PLAN", "ANSWER", "STOP"):
            raise ValueError(f"unknown reasoner action: {self.kind}")
        if self.kind == "PLAN" and self.request is None:
            raise ValueError("PLAN requires an observation request")
        if self.kind == "ANSWER" and (not self.answer or not self.evidence_refs):
            raise ValueError("ANSWER requires answer text and supporting evidence")
        if self.kind in ("ALIGN", "STOP") and self.request is not None:
            raise ValueError(f"{self.kind} cannot carry an observation request")
        if self.kind == "STOP" and not self.reason.strip():
            raise ValueError("STOP requires an explicit reason")
        _validate_identifier_tuple(self.evidence_refs, "reasoner action evidence_refs")


@dataclass(frozen=True)
class ReasonerTurn:
    updates: tuple[RequirementUpdate, ...]
    state_summary: str
    action: ReasonerAction


class ReasonerBackend(Protocol):
    def align(self, context: str) -> ReasonerTurn: ...
    def plan_focus(self, context: str) -> ReasonerTurn: ...
    def decide(self, context: str) -> ReasonerTurn: ...


@dataclass(frozen=True)
class FocusReviewReceipt:
    """Framework-owned facts about one committed Focus transaction.

    The projector may interpret these observations, but it cannot redefine
    which evidence was produced by the transaction or turn a gap into an
    observed event.
    """

    requirement_id: str
    evidence_goal: str
    expected_complementary_probe: str
    proposed_next_probe: str
    reserved_follow_up_probe: str
    new_event_refs: tuple[str, ...]
    new_gap_refs: tuple[str, ...]
    mandatory_stages_remaining: int
    visual_calls_remaining_after_focus: int
    force_open_requirement: bool

    def __post_init__(self) -> None:
        if not self.requirement_id.strip() or not self.evidence_goal.strip():
            raise ValueError("focus review receipt requires requirement and evidence goal")
        _validate_identifier_tuple(self.new_event_refs, "focus receipt event refs")
        _validate_identifier_tuple(self.new_gap_refs, "focus receipt gap refs")
        if set(self.new_event_refs).intersection(self.new_gap_refs):
            raise ValueError("focus receipt cannot classify one ref as event and gap")
        if not self.new_event_refs and not self.new_gap_refs:
            raise ValueError("focus review receipt requires committed observation provenance")
        if self.mandatory_stages_remaining < 0 or self.visual_calls_remaining_after_focus < 0:
            raise ValueError("focus receipt budgets must be nonnegative")

    @property
    def all_new_refs(self) -> tuple[str, ...]:
        return (*self.new_event_refs, *self.new_gap_refs)


@dataclass(frozen=True)
class ObservationReceipt:
    """Executor-owned record of one committed observation transaction.

    This is scheduling evidence, not semantic evidence: it records the exact
    region and question sent to ObserverAgent plus whether that transaction
    produced events, gaps, or both.  ReasonerAgent may use it to avoid
    repeating failed acquisition, but it cannot use a gap as answer evidence.
    """

    round_id: int
    request_fingerprint: str
    requirement_id: str
    evidence_goal: str
    targets: tuple[tuple[str, float, float], ...]
    event_refs: tuple[str, ...]
    gap_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.round_id <= 0 or not self.requirement_id.strip() or not self.evidence_goal.strip():
            raise ValueError("observation receipt requires round, requirement and evidence goal")
        if not self.targets:
            raise ValueError("observation receipt requires at least one requested target")
        for video_id, start, end in self.targets:
            if not video_id.strip() or start < 0 or end < start:
                raise ValueError("observation receipt target needs video and ordered time span")
        _validate_identifier_tuple(self.event_refs, "observation receipt event refs")
        _validate_identifier_tuple(self.gap_refs, "observation receipt gap refs")
        if set(self.event_refs).intersection(self.gap_refs):
            raise ValueError("observation receipt cannot classify one ref as event and gap")
        if not self.event_refs and not self.gap_refs:
            raise ValueError("observation receipt requires committed event or gap provenance")
        expected = _observation_fingerprint(self.evidence_goal, self.targets)
        if self.request_fingerprint != expected:
            raise ValueError("observation receipt fingerprint does not match its request")

    @property
    def outcome(self) -> Literal["events", "gap_only", "mixed"]:
        if self.event_refs and self.gap_refs:
            return "mixed"
        return "events" if self.event_refs else "gap_only"


class StateProjector(Protocol):
    """Pure projection owner: it never receives a previous reasoning state."""
    def project(
        self,
        *,
        question: str,
        options: Sequence[str],
        video_ids: Sequence[str],
        snapshot: Sequence[Mapping[str, object]],
        receipt: FocusReviewReceipt,
    ) -> DecisionState: ...


class ReceiptStateProjector:
    """Deterministic CPU fallback for the evidence-reprojection boundary.

    Production runs use :class:`OpenAICompatibleStateProjector` to interpret
    the current evidence.  Tests and mock runs still need the same state
    transition without a model: this projector discards the previous working
    state and carries forward only the committed observation question and its
    provenance.  It deliberately leaves that question open because a
    deterministic module cannot decide what the visual evidence means.
    """

    def project(
        self,
        *,
        question: str,
        options: Sequence[str],
        video_ids: Sequence[str],
        snapshot: Sequence[Mapping[str, object]],
        receipt: FocusReviewReceipt,
    ) -> DecisionState:
        del snapshot
        follow_up = next(
            (
                value.strip()
                for value in (
                    receipt.reserved_follow_up_probe,
                    receipt.proposed_next_probe,
                    receipt.expected_complementary_probe,
                )
                if value.strip()
            ),
            "",
        )
        objective = follow_up or receipt.evidence_goal
        review_refs = receipt.new_event_refs or receipt.new_gap_refs
        outcome: ReviewOutcome = "partial" if receipt.new_event_refs else "no_gain"
        review_reason = (
            "Validated events were committed; their decision meaning requires a fresh "
            "Reasoner decision."
            if receipt.new_event_refs
            else "The observation produced only explicit gaps and did not settle the question."
        )
        candidate = CandidateObservation(
            candidate_id="P01",
            target_video_ids=tuple(video_ids),
            observable=objective,
            purpose="Obtain evidence that could change the current decision.",
            related_requirement_id=receipt.requirement_id,
        )
        return DecisionState(
            question=question,
            video_ids=tuple(video_ids),
            requirements=(EvidenceRequirement(
                receipt.requirement_id,
                objective,
                tuple(video_ids),
                "open",
                tuple(receipt.new_event_refs),
            ),),
            options=tuple(options),
            candidate_observations=(candidate,),
            summary=objective[:1600],
            review_outcome=outcome,
            review_reason=review_reason,
            review_evidence_refs=tuple(review_refs),
        )


class ObserverExecutor(Protocol):
    def observe(
        self,
        request: FocusIntent,
        round_id: int,
        ledger: EvidenceLedger,
        memory: SemanticZoomMemory,
    ) -> Sequence[tuple[str, ObservationBatch]]: ...


@dataclass(frozen=True)
class ObservedBatch:
    """An Observer result plus the memory address it completely refines."""
    video_id: str
    batch: ObservationBatch
    zoom_draft: ZoomDraft | None = None


class SlotFocusExecutor:
    """Resolve semantic slot selections into source-preserving dense Focus calls."""

    def __init__(
        self,
        *,
        tool_factory: Callable[[], FocusTool],
        videos: Sequence[VideoInput],
        question: str,
    ) -> None:
        self._tool_factory = tool_factory
        self._tool: FocusTool | None = None
        self.videos = {item.video_id: item for item in videos}
        if not question.strip():
            raise ValueError("slot focus executor needs the original question")
        self.question = question

    @property
    def tool(self) -> FocusTool:
        """Load the VLM only if ReasonerAgent actually requests visual work."""
        if self._tool is None:
            self._tool = self._tool_factory()
        return self._tool

    def observe(
        self,
        request: FocusIntent,
        round_id: int,
        ledger: EvidenceLedger,
        memory: SemanticZoomMemory,
    ) -> Sequence[tuple[str, ObservationBatch] | ObservedBatch]:
        results: list[ObservedBatch] = []
        slots = memory.by_ids(request.slot_ids)
        if request.mode == "recover" and any(slot.resolution != "coverage" for slot in slots):
            raise ValueError("RECOVER may select only unobserved coverage slots")
        if request.mode == "refine" and any(slot.resolution == "coverage" for slot in slots):
            raise ValueError("REFINE may select only observed coarse/fine slots")
        for slot_position, slot in enumerate(slots, start=1):
            video_id = slot.video_id
            video = self.videos.get(video_id)
            if video is None:
                raise ValueError(f"memory slot references unknown video: {video_id}")
            zoom_id = f"r{round_id}-{request.requirement_id}-{slot_position:02d}"
            start, end = slot.time_span
            if start == end:
                child_requests = (FocusRequest(
                    video_id=video_id, evidence_need=request.evidence_goal,
                    center_timestamp_seconds=start, sampling_fps=1.0,
                ),)
            else:
                child_requests = split_complete_zoom(
                    video_id=video_id, time_span=slot.time_span,
                    evidence_need=request.evidence_goal,
                    frames_per_call=self.tool.config.frames_per_call,
                    sampling_fps=1.0,
                )
            completed: list[FocusResult] = []
            local_question = (
                f"Observe only {video_id} inside the supplied local interval. "
                f"Evidence need: {request.evidence_goal}"
            )
            for focus_request in child_requests:
                result = self.tool.run(
                    video=video, request=focus_request, question=local_question,
                    report_hints=(
                        request.evidence_goal,
                        "Report local visible facts only. Do not infer another video, an unobserved interval, or a whole-video absence.",
                    ),
                )
                completed.append(result)
            if not completed:
                raise RuntimeError("slot focus produced no observation calls")
            region_span = (
                completed[0].resolved_time_span[0],
                completed[-1].resolved_time_span[1],
            )
            for result in completed:
                results.append(ObservedBatch(
                    video_id, result.observation,
                    ZoomDraft(
                        zoom_id, video_id, region_span, result.resolved_time_span,
                        slot.evidence_refs, (slot.slot_id,),
                    ),
                ))
        return tuple(results)


class OpenAICompatibleReasonerBackend:
    """Production backend; policy remains in ReasonerAgent, not the runner."""

    def __init__(self, *, base_url: str, model: str, api_key_env: str = "LLM_API_KEY", maximum_tokens: int = 768,
                 timeout_seconds: float = 600.0,
                 trace_sink: Callable[[APICallRecord], None] | None = None) -> None:
        if maximum_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("reasoner token and timeout limits must be positive")
        self.base_url, self.model, self.api_key_env = base_url, model, api_key_env
        self.maximum_tokens, self.timeout_seconds = maximum_tokens, timeout_seconds
        self._transport = OpenAICompatibleTransport(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            trace_sink=trace_sink,
        )
        self._request_config = APIRequestConfig(
            maximum_tokens=maximum_tokens,
            timeout_seconds=timeout_seconds,
        )

    def align(self, context: str) -> ReasonerTurn:
        """Rebuild one grounded synthesis from the current evidence revision."""
        system = """You are ReasonerAgent in ALIGN mode. Return exactly one compact JSON object and no markdown. Consume only the supplied structured evidence; never request frames and never answer the task.

Return exactly:
{"synthesis":[{"kind":"finding|open_question","statement":"grounded fact or decision-critical unknown","videos":["v1"],"evidence_indices":[1,2]}],"state_summary":"current evidence synthesis","action":{"kind":"ALIGN","reason":"why synthesis was needed"}}

Rebuild the synthesis from the question, answer contract and current evidence, not by editing or preserving old items. Emit only useful grounded findings and decision-critical open questions; do not enumerate candidate answers. A finding needs 1-8 unique evidence_indices from answer_evidence_map. An open question may cite 0-8 related evidence indices. videos is the actual scope of that statement and may contain one or several supplied video IDs; do not force all input videos into every item and do not require evidence from every scoped video when the statement itself is local. direct_observation is citable visual evidence; inferred_interpretation is an explicitly uncertain hypothesis and cannot by itself establish a finding. When exploration_agenda.remaining_focus_stages is positive, retain at least one open question. Observation gaps are context about what was not observed and are never positive or negative evidence. Preserve uncertainty when current evidence does not settle it. Do not use workflow priors as observed facts. ALIGN performs question-conditioned evidence synthesis; it does not PLAN, ANSWER, or STOP."""
        candidate = self._complete(system, context)
        errors: list[str] = []
        for attempt in range(3):
            try:
                return self._parse_alignment_turn(candidate, context)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(str(error))
                if attempt >= 2:
                    raise ValueError(
                        "Reasoner emitted invalid ALIGN after two structural repairs: "
                        + " | ".join(errors)
                    ) from error
                candidate = self._complete(
                    system,
                    "Return only the ALIGN schema. Rebuild synthesis from current evidence. "
                    "Use integer evidence_indices from answer_evidence_map, never provenance strings."
                    + "\nVALIDATION ERROR:\n" + str(error)
                    + "\n\nCONTEXT:\n" + context
                    + "\n\nINVALID CANDIDATE:\n" + candidate,
                )
        raise AssertionError("unreachable ALIGN repair loop")

    def plan_focus(self, context: str) -> ReasonerTurn:
        """Use a planning-only decoding contract during mandatory exploration."""
        system = """You are ReasonerAgent in evidence-acquisition mode. Return exactly one compact JSON object and no markdown.
The only permitted action is PLAN. Do not answer the question, rank options, rewrite the evidence synthesis, emit confidence, ALIGN, or STOP.
Return exactly:
{"state_summary":"short evidence need","action":{"kind":"PLAN","request":{"question_index":1,"slot_indices":[1],"evidence_goal":"one locally visible question","next_probe_goal":"optional complementary question","mode":"refine or recover"}}}
Select one integer question_index from open_questions and integer slot_indices from memory_map; never copy their ID strings. Prefer a matching contrastive_frontier probe when one is supplied, while selecting every slot needed for that one probe within the current budget. Read memory_map rows as [index,slot_id,video,start,end,level,cost,tag]. Use mode=refine only for c/f rows and mode=recover only for u rows. The sum of selected row costs MUST NOT exceed exploration_agenda.maximum_visual_calls_for_current_stage; prefer the smallest sufficient set. Never invent coordinates, identifiers, an answer, or a conclusion. The executor maps indices back to provenance-preserving IDs. One completed cycle is only an evidence update, never proof that the question is solved. Choose a locally observable test that can change the decision: verify a boundary, transition, identity, countable event, or competing interpretation. Do not ask one video to decide facts about another video or prove absence outside the selected slot. Respect every inference_protocol rule, especially clock comparability, alias identity, counting, and answer format."""
        return self._structured_turn(system, context, planning_only=True)

    def decide(self, context: str) -> ReasonerTurn:
        system = """You are ReasonerAgent. Return exactly one JSON object, with no markdown.
Required top-level keys: state_summary (string), action (object). Do not emit semantic updates: only the dedicated ALIGN action may replace the evidence synthesis.
Action kind is exactly PLAN, ALIGN, ANSWER, or STOP. The working state has already been freshly reprojected after every FOCUS. Choose ALIGN only when an explicit comparison or correspondence synthesis across current evidence would materially improve the next decision; it is never a routine post-Focus or pre-answer stage. Do not repeat it when working_state.current_evidence_is_explicitly_aligned is true unless the current synthesis exposes a genuine unresolved conflict. For ALIGN, return only {"kind":"ALIGN","reason":"why explicit alignment is useful now"}; the runtime will invoke the dedicated synthesis contract.
For PLAN, request is {question_index,slot_indices,evidence_goal,next_probe_goal,mode}. Select question_index from open_questions and slot_indices from memory_map; output integers, never their ID strings. refine selects c/f, recover selects u, and selected costs must fit exploration_agenda.maximum_visual_calls_for_current_stage and visual_calls_remaining. Prefer the smallest sufficient set. Never output raw timestamps or copy/invent IDs. Consult observation_receipts: do not repeat the same normalized goal over the same video/time targets after that request produced only observation gaps.
For ANSWER, include a nonempty answer and 1-8 unique integer evidence_indices selected from answer_evidence_map; never copy provenance strings. Open questions are residual uncertainty, not a mechanical ANSWER blocker. After mandatory exploration and current-evidence reprojection, choose ANSWER when no contrastive_frontier probe or other specific executable observation is likely to change the best-supported answer; choose PLAN only when its stated local observation could materially change that answer. ALIGN is optional and is not an answer gate. Obey answer_contract exactly. When visual_calls_remaining is zero, PLAN is forbidden: return the best evidence-supported ANSWER, using ALIGN only if an explicit comparison is genuinely useful; use STOP only when no answer can be represented at all, and always explain why. With choices, uncertainty is not a refusal condition—select the best-supported option label. Follow structural constraints from answer_contract without turning them into observation policy.
When exploration_agenda.remaining_focus_stages is positive, ANSWER is forbidden. Normally select PLAN; ALIGN is permitted only when reconciling existing evidence is genuinely necessary to formulate the next discriminating observation, and it must retain open work for the remaining Focus stages.
Treat a cycle as evidence acquisition, not a confidence milestone. Local FOCUS evidence applies only to its cited video/time span and cannot establish global absence. Same-clock times may be compared; different-clock times may not. Count physical events/entities rather than frames or duplicate synchronized views. Never use an observation gap as negative evidence. Select only supplied integer indices and obey inference_protocol."""
        return self._structured_turn(system, context, planning_only=False)

    def _structured_turn(self, system: str, context: str, *, planning_only: bool) -> ReasonerTurn:
        candidate = self._complete(system, context)
        errors: list[str] = []
        for repair_index in range(3):
            try:
                turn = self._parse_policy_turn(candidate, context)
                if planning_only and turn.action.kind != "PLAN":
                    raise ValueError("evidence-acquisition mode requires PLAN")
                return turn
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(str(error))
                if repair_index >= 2:
                    raise ValueError(
                        "Reasoner emitted an invalid action after two structural repairs: "
                        + " | ".join(errors)
                    ) from error
                repair_rule = (
                    "Return only the PLAN schema. Select one integer question_index and nonempty integer slot_indices from the supplied rows; do not emit updates, ANSWER, or STOP."
                    if planning_only else
                    "Return only the required schema. Do not emit updates. Choose PLAN, ALIGN, ANSWER, or STOP. For PLAN use one integer question_index and nonempty integer slot_indices from supplied rows. ALIGN carries only a reason. For ANSWER use 1-8 integer evidence_indices from answer_evidence_map, never provenance strings."
                )
                candidate = self._complete(
                    system,
                    repair_rule
                    + "\nVALIDATION ERROR:\n" + str(error)
                    + "\n\nCONTEXT:\n" + context
                    + "\n\nINVALID CANDIDATE:\n" + candidate,
                )
        raise AssertionError("unreachable reasoner repair loop")

    @staticmethod
    def _parse_policy_turn(text: str, context: str) -> ReasonerTurn:
        """Compile the compact API schema into provenance-bearing actions.

        Incremental semantic updates are intentionally discarded at this
        boundary: only ALIGN may replace the synthesis. PLAN uses short
        row indices so the language model never has to reproduce long,
        revision-sensitive slot identifiers.
        """
        data = extract_json_object(text)
        data["updates"] = []
        context_value = json.loads(context)
        agenda = context_value.get("exploration_agenda")
        if not isinstance(agenda, dict):
            raise ValueError("reasoner context lacks exploration_agenda")
        visual_calls_remaining = agenda.get("visual_calls_remaining")
        remaining_focus_stages = agenda.get("remaining_focus_stages")
        current_stage_cap = agenda.get("maximum_visual_calls_for_current_stage")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (visual_calls_remaining, remaining_focus_stages, current_stage_cap)
        ):
            raise ValueError("reasoner context has invalid exploration budget")
        action = data.get("action")
        if not isinstance(action, dict):
            raise ValueError("reasoner turn requires action object")
        if action.get("kind") == "ALIGN":
            extra = set(action) - {"kind", "reason"}
            if extra:
                raise ValueError(f"ALIGN request contains unsupported fields: {sorted(extra)}")
            if not isinstance(action.get("reason"), str) or not action["reason"].strip():
                raise ValueError("ALIGN request requires a nonempty reason")
        elif action.get("kind") == "PLAN":
            if visual_calls_remaining == 0 or current_stage_cap == 0:
                raise ValueError("PLAN is unavailable because no visual-call budget remains")
            request = action.get("request")
            if not isinstance(request, dict):
                raise ValueError("PLAN requires request object")
            question_index = request.pop("question_index", None)
            if not isinstance(question_index, int) or isinstance(question_index, bool):
                raise ValueError("PLAN requires one integer question_index")
            requirements = context_value.get("open_questions")
            if not isinstance(requirements, list):
                raise ValueError("reasoner context lacks open_questions")
            requirement_lookup = {
                item.get("index"): item.get("id")
                for item in requirements
                if isinstance(item, dict)
                and isinstance(item.get("index"), int)
                and isinstance(item.get("id"), str)
            }
            requirement_id = requirement_lookup.get(question_index)
            if requirement_id is None:
                raise ValueError(
                    f"PLAN selected unknown question index: {question_index}"
                )
            request["requirement_id"] = requirement_id
            indices = request.pop("slot_indices", None)
            if (
                not isinstance(indices, list)
                or not indices
                or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
                or len(set(indices)) != len(indices)
            ):
                raise ValueError("PLAN slot_indices must be a nonempty unique integer array")
            rows = context_value.get("memory_map")
            if not isinstance(rows, list):
                raise ValueError("reasoner context lacks memory_map")
            lookup: dict[int, str] = {}
            for row in rows:
                if (
                    isinstance(row, list) and len(row) >= 2
                    and isinstance(row[0], int) and isinstance(row[1], str)
                ):
                    lookup[row[0]] = row[1]
            missing = [index for index in indices if index not in lookup]
            if missing:
                raise ValueError(f"PLAN selected unknown slot indices: {missing}")
            row_costs = {
                row[0]: row[6]
                for row in rows
                if isinstance(row, list) and len(row) >= 7
                and isinstance(row[0], int)
                and isinstance(row[6], int) and not isinstance(row[6], bool)
            }
            estimated_cost = sum(row_costs[index] for index in indices)
            if estimated_cost > min(visual_calls_remaining, current_stage_cap):
                raise ValueError(
                    "PLAN slot cost exceeds the currently executable visual-call budget"
                )
            request.pop("slot_ids", None)
            request["slot_ids"] = [lookup[index] for index in indices]
            target_lookup = {
                row[0]: (row[2], float(row[3]), float(row[4]))
                for row in rows
                if (
                    isinstance(row, list) and len(row) >= 5
                    and isinstance(row[0], int) and isinstance(row[2], str)
                    and isinstance(row[3], (int, float)) and not isinstance(row[3], bool)
                    and isinstance(row[4], (int, float)) and not isinstance(row[4], bool)
                )
            }
            if any(index not in target_lookup for index in indices):
                raise ValueError("PLAN selected a memory row without valid target provenance")
            goal = request.get("evidence_goal")
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("PLAN requires a nonempty evidence_goal")
            request_fingerprint = _observation_fingerprint(
                goal, tuple(target_lookup[index] for index in indices),
            )
            receipts = context_value.get("observation_receipts")
            if not isinstance(receipts, list):
                raise ValueError("reasoner context lacks observation_receipts")
            if any(
                isinstance(receipt, dict)
                and receipt.get("outcome") == "gap_only"
                and receipt.get("request_fingerprint") == request_fingerprint
                for receipt in receipts
            ):
                raise ValueError(
                    "PLAN exactly repeats a prior gap-only observation request; "
                    "choose a different target or a materially different local question"
                )
        elif action.get("kind") == "ANSWER":
            if remaining_focus_stages > 0:
                raise ValueError("ANSWER is unavailable while mandatory exploration remains")
            open_questions = context_value.get("open_questions")
            if not isinstance(open_questions, list):
                raise ValueError("reasoner context lacks open_questions")
            if "evidence_refs" in action:
                raise ValueError("ANSWER must use evidence_indices, not provenance IDs")
            indices = action.pop("evidence_indices", None)
            if (
                not isinstance(indices, list)
                or not indices
                or len(indices) > 8
                or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
                or len(indices) != len(set(indices))
            ):
                raise ValueError("ANSWER evidence_indices must be 1-8 unique integers")
            rows = context_value.get("answer_evidence_map")
            if not isinstance(rows, list):
                raise ValueError("reasoner context lacks answer_evidence_map")
            lookup = {
                row[0]: row[1]
                for row in rows
                if isinstance(row, list) and len(row) >= 2
                and isinstance(row[0], int) and isinstance(row[1], str)
            }
            missing = [index for index in indices if index not in lookup]
            if missing:
                raise ValueError(f"ANSWER selected unknown evidence indices: {missing}")
            action["evidence_refs"] = [lookup[index] for index in indices]
        return parse_reasoner_turn(json.dumps(data, ensure_ascii=False))

    @staticmethod
    def _parse_alignment_turn(text: str, context: str) -> ReasonerTurn:
        data = extract_json_object(text)
        action = data.get("action")
        if not isinstance(action, dict) or action.get("kind") != "ALIGN":
            raise ValueError("ALIGN mode requires action.kind=ALIGN")
        context_value = json.loads(context)
        evidence_rows = context_value.get("answer_evidence_map")
        videos = context_value.get("videos")
        if not isinstance(evidence_rows, list) or not isinstance(videos, list):
            raise ValueError("ALIGN context lacks videos or answer_evidence_map")
        known_videos = {item for item in videos if isinstance(item, str)}
        evidence_lookup = {
            row[0]: row[1]
            for row in evidence_rows
            if isinstance(row, list) and len(row) >= 2
            and isinstance(row[0], int) and isinstance(row[1], str)
        }
        raw_updates = data.pop("synthesis", None)
        if not isinstance(raw_updates, list) or not raw_updates:
            raise ValueError("ALIGN synthesis must be a nonempty array")
        compiled: list[dict[str, object]] = []
        counters = {"finding": 0, "open_question": 0}
        for position, item in enumerate(raw_updates):
            if not isinstance(item, dict):
                raise ValueError(f"ALIGN synthesis[{position}] must be an object")
            if set(item) - {"kind", "statement", "videos", "evidence_indices"}:
                raise ValueError("ALIGN synthesis item contains unsupported fields")
            kind = item.get("kind")
            if kind not in counters:
                raise ValueError("ALIGN synthesis kind must be finding or open_question")
            statement = item.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError("ALIGN synthesis statement must be nonempty")
            scope = item.get("videos")
            if (not isinstance(scope, list) or not scope
                    or any(not isinstance(video, str) or video not in known_videos for video in scope)
                    or len(scope) != len(set(scope))):
                raise ValueError("ALIGN synthesis videos must be unique supplied video IDs")
            indices = item.get("evidence_indices", [])
            if (
                not isinstance(indices, list) or len(indices) > 8
                or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
                or len(indices) != len(set(indices))
            ):
                raise ValueError("ALIGN evidence_indices must be 0-8 unique integers")
            if kind == "finding" and not indices:
                raise ValueError("ALIGN finding requires positive evidence")
            missing = [index for index in indices if index not in evidence_lookup]
            if missing:
                raise ValueError(f"ALIGN selected unknown evidence indices: {missing}")
            counters[kind] += 1
            compiled.append({
                "requirement_id": ("F" if kind == "finding" else "U") + f"{counters[kind]:02d}",
                "status": "supported" if kind == "finding" else "open",
                "evidence_refs": [evidence_lookup[index] for index in indices],
                "summary": statement,
                "video_ids": scope,
            })
        agenda = context_value.get("exploration_agenda")
        if not isinstance(agenda, dict):
            raise ValueError("ALIGN context lacks exploration_agenda")
        remaining_stages = agenda.get("remaining_focus_stages")
        if not isinstance(remaining_stages, int) or isinstance(remaining_stages, bool):
            raise ValueError("ALIGN context has invalid remaining_focus_stages")
        if remaining_stages > 0 and counters["open_question"] == 0:
            raise ValueError(
                "ALIGN must retain an open_question while mandatory exploration remains"
            )
        data["updates"] = compiled
        return parse_reasoner_turn(json.dumps(data, ensure_ascii=False))

    def _complete(self, system: str, user: str) -> str:
        return self._transport.complete(
            role="reasoner_action",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            config=self._request_config,
        )


class OpenAICompatibleStateProjector:
    """Rebuild a decision state from current evidence only, never a prior trace."""

    def __init__(self, *, base_url: str, model: str, api_key_env: str = "LLM_API_KEY", maximum_tokens: int = 768,
                 timeout_seconds: float = 600.0, protocol_notes: Sequence[str] = (),
                 enable_thinking: bool | None = None,
                 trace_sink: Callable[[APICallRecord], None] | None = None) -> None:
        if maximum_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("projector token and timeout limits must be positive")
        self.base_url, self.model, self.api_key_env = base_url, model, api_key_env
        self.maximum_tokens, self.timeout_seconds = maximum_tokens, timeout_seconds
        self.protocol_notes = tuple(protocol_notes)
        if any(not item.strip() for item in self.protocol_notes):
            raise ValueError("projector protocol notes must be nonempty")
        self._transport = OpenAICompatibleTransport(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            trace_sink=trace_sink,
        )
        self._request_config = APIRequestConfig(
            maximum_tokens=maximum_tokens,
            timeout_seconds=timeout_seconds,
            enable_thinking=enable_thinking,
        )

    def project(
        self,
        *,
        question: str,
        options: Sequence[str],
        video_ids: Sequence[str],
        snapshot: Sequence[Mapping[str, object]],
        receipt: FocusReviewReceipt,
    ) -> DecisionState:
        system = """You are the evidence-reprojection module owned by ReasonerAgent. Return exactly one compact JSON object and no markdown. Rebuild a fresh answer-free working state from only the question, inference protocol, indexed current evidence, and framework-owned last-focus receipt. Never preserve a prior conclusion. The memory_index is a complete temporal navigation map but is not itself evidence; the remaining items are the bounded evidence-rich page.

Required top-level keys:
{"requirements": [...], "frontier": [...], "review": {"outcome": "resolved|partial|no_gain|conflict", "reason": "...", "evidence_indices": [...]}}

Output 1-3 decision requirements. Each requirement is {id,objective,videos,status,evidence_indices}. status is open, supported, or contradicted. An open requirement is a decision-critical uncertainty that still needs observation. Mark supported or contradicted only when the cited current evidence directly settles that requirement; every non-open requirement needs at least one evidence index. evidence_indices contains at most 2 unique integers selected from the supplied evidence_reference_catalogue. Never copy or invent provenance ID strings. videos must be nonempty and contain only supplied video IDs.

frontier contains 0-3 locally observable discriminating probes. Each probe is {id,requirement_id,videos,observable,purpose}. requirement_id must name an open requirement. observable states what could be seen in the named videos; purpose states which unresolved distinction that observation could change. Do not encode an answer candidate, option ranking, confidence, task-specific procedure, or a generic request for more detail. If mandatory stages remain, emit at least one probe. Otherwise an empty frontier means that no specific executable observation is likely to change the decision; residual uncertainty may remain.

Review the last Focus against its stated evidence goal. outcome=resolved when a newly observed event directly settled the local goal, partial when new evidence narrowed but did not settle it, no_gain when it produced only a gap, repeated known detail, or failed to discriminate competing interpretations, and conflict when newly observed events support incompatible interpretations. If last_focus_receipt.force_open_requirement is true, retain at least one open requirement for the required follow-up stage. Otherwise retain an open requirement whenever decision-relevant ambiguity remains, but do not treat an open requirement as a mechanical ban on answering. A reserved or proposed complementary probe is a review obligation, not a conclusion: carry its unresolved substance into an open requirement when current evidence has not addressed it.

review.evidence_indices must contain at most 2 unique integers from last_focus_receipt.review_evidence_indices. outcome=resolved or conflict must use an index from last_focus_receipt.new_event_evidence_indices; a gap can support only partial or no_gain. This review grounding is independent of requirement status: a Focus may settle its stated local goal while revealing a new open decision requirement.

Preserve competing interpretations when evidence does not distinguish them. Express them as uncertainty, not as a list of final answers. A local observation cannot imply whole-video or cross-video absence. Do not turn a plausible workflow prior into an observed transition. Use only comparable clocks and source-preserving identities. Do not output an answer, completion decision, confidence, action, summary, or prior reasoning."""
        catalogue = _snapshot_reference_catalogue(snapshot)
        reference_indices = {ref: index for index, ref in enumerate(catalogue, 1)}
        all_review_refs, event_review_refs = _review_support_refs(snapshot, receipt)
        review_indices = tuple(
            reference_indices[ref] for ref in catalogue if ref in all_review_refs
        )
        event_indices = tuple(
            reference_indices[ref] for ref in catalogue if ref in event_review_refs
        )
        gap_indices = tuple(
            reference_indices[ref] for ref in receipt.new_gap_refs
            if ref in reference_indices
        )
        if not review_indices:
            raise ValueError("committed Focus provenance is absent from projection snapshot")
        receipt_payload = {
            "focus_requirement_id": receipt.requirement_id,
            "focus_goal": receipt.evidence_goal,
            "expected_complementary_probe": receipt.expected_complementary_probe,
            "proposed_next_probe": receipt.proposed_next_probe,
            "reserved_follow_up_probe": receipt.reserved_follow_up_probe,
            "review_evidence_indices": list(review_indices),
            "new_event_evidence_indices": list(event_indices),
            "new_gap_evidence_indices": list(gap_indices),
            "observation_status": (
                "events_and_gaps" if receipt.new_event_refs and receipt.new_gap_refs
                else "validated_events" if receipt.new_event_refs
                else "observation_gap"
            ),
            "mandatory_stages_remaining": receipt.mandatory_stages_remaining,
            "visual_calls_remaining_after_focus": receipt.visual_calls_remaining_after_focus,
            "force_open_requirement": receipt.force_open_requirement,
        }
        payload = {
            "question": question,
            "options": list(options),
            "videos": list(video_ids),
            "inference_protocol": list(self.protocol_notes),
            "last_focus_receipt": receipt_payload,
            "evidence_reference_catalogue": [
                {"index": index, "last_focus": index in review_indices,
                 "new_event": index in event_indices, "new_gap": index in gap_indices}
                for index in range(1, len(catalogue) + 1)
            ],
            "current_evidence": list(_index_projection_snapshot(snapshot, reference_indices)),
        }
        candidate = self._transport.complete(
            role="state_projection",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            config=self._request_config,
        )
        errors: list[str] = []
        for attempt in range(3):
            try:
                return _parse_projected_state(
                    candidate, question=question, options=tuple(options),
                    video_ids=tuple(video_ids), evidence_catalogue=catalogue,
                    protocol_notes=self.protocol_notes,
                    force_open_requirement=receipt.force_open_requirement,
                    visual_calls_remaining=receipt.visual_calls_remaining_after_focus,
                    required_review_refs=all_review_refs,
                    required_event_review_refs=event_review_refs,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(str(error))
                if attempt >= 2:
                    raise ValueError(
                        "projector stayed invalid after two indexed structural repairs: "
                        + " | ".join(errors)
                    ) from error
                repair_text = (
                    json.dumps(payload, ensure_ascii=False)
                    + "\nVALIDATION ERROR:\n" + str(error)
                    + "\nINVALID CANDIDATE:\n" + candidate
                    + "\nReturn corrected JSON only with requirements, frontier, and review. "
                    "Use integer evidence_indices, never provenance strings. "
                    "Every videos field must be a JSON array containing only these exact "
                    f"string IDs: {json.dumps(list(video_ids), ensure_ascii=False)}."
                )
                candidate = self._transport.complete(
                    role="state_projection.repair",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": repair_text},
                    ],
                    config=self._request_config,
                )
        raise AssertionError("unreachable projector repair loop")


def _parse_projected_state(
    text: str,
    *,
    question: str,
    options: tuple[str, ...],
    video_ids: tuple[str, ...],
    evidence_catalogue: Sequence[str],
    protocol_notes: tuple[str, ...] = (),
    force_open_requirement: bool = False,
    visual_calls_remaining: int = 0,
    required_review_refs: set[str] | None = None,
    required_event_review_refs: set[str] | None = None,
) -> DecisionState:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("projected state must be a JSON object")
    if set(value) != {"requirements", "frontier", "review"}:
        raise ValueError(
            "projected state must contain only requirements, frontier, and review"
        )
    catalogue = tuple(evidence_catalogue)
    _validate_identifier_tuple(catalogue, "projection evidence catalogue")
    if len(catalogue) != len(set(catalogue)):
        raise ValueError("projection evidence catalogue must be unique")
    requirements: list[EvidenceRequirement] = []
    for item in value.get("requirements", [])[:3]:
        if not isinstance(item, dict):
            raise ValueError("projected requirement must be an object")
        if "evidence_refs" in item:
            raise ValueError("projected requirement must use evidence_indices, not provenance IDs")
        refs = _projection_refs_from_indices(
            item.get("evidence_indices", []), catalogue,
            "projected requirement evidence_indices", maximum=2,
        )
        status = str(item.get("status", "open"))
        if status not in ("open", "supported", "contradicted"):
            raise ValueError("projected requirement has an invalid status")
        if status != "open" and not refs:
            raise ValueError("resolved projected requirement needs evidence refs")
        scope = _projection_video_scope(
            item.get("videos"), video_ids, "projected requirement videos",
        )
        requirements.append(EvidenceRequirement(
            str(item["id"]), str(item["objective"])[:240], scope, status, refs,
        ))
    if not requirements:
        raise ValueError("projected state must contain at least one requirement")
    if force_open_requirement and not any(item.status == "open" for item in requirements):
        raise ValueError("mandatory follow-up projection must retain an open requirement")
    requirement_by_id = {item.requirement_id: item for item in requirements}
    raw_frontier = value.get("frontier")
    if not isinstance(raw_frontier, list) or len(raw_frontier) > 3:
        raise ValueError("projected frontier must be an array of at most three probes")
    frontier: list[CandidateObservation] = []
    for item in raw_frontier:
        if not isinstance(item, dict):
            raise ValueError("projected frontier probe must be an object")
        if set(item) != {"id", "requirement_id", "videos", "observable", "purpose"}:
            raise ValueError("projected frontier probe has unsupported or missing fields")
        requirement_id = str(item["requirement_id"])
        requirement = requirement_by_id.get(requirement_id)
        if requirement is None or requirement.status != "open":
            raise ValueError("projected frontier probe must target an open requirement")
        scope = _projection_video_scope(
            item["videos"], video_ids, "projected frontier probe videos",
        )
        if any(video not in requirement.video_ids for video in scope):
            raise ValueError("projected frontier probe exceeds its requirement video scope")
        frontier.append(CandidateObservation(
            candidate_id=str(item["id"]),
            target_video_ids=scope,
            observable=str(item["observable"])[:240],
            purpose=str(item["purpose"])[:240],
            related_requirement_id=requirement_id,
        ))
    if force_open_requirement and not frontier:
        raise ValueError("mandatory follow-up projection must emit a discriminating probe")
    raw_review = value.get("review")
    if not isinstance(raw_review, dict):
        raise ValueError("projected state requires a review object")
    outcome = str(raw_review.get("outcome", ""))
    if outcome not in ("resolved", "partial", "no_gain", "conflict"):
        raise ValueError("projected review has an invalid outcome")
    if visual_calls_remaining < 0:
        raise ValueError("projection visual call budget must be nonnegative")
    review_reason = str(raw_review.get("reason", ""))
    if not review_reason.strip():
        raise ValueError("projected review reason must be nonempty")
    if "evidence_refs" in raw_review:
        raise ValueError("projected review must use evidence_indices, not provenance IDs")
    review_evidence_refs = _projection_refs_from_indices(
        raw_review.get("evidence_indices", []), catalogue,
        "projected review evidence_indices", maximum=2,
    )
    review_refs = required_review_refs or set()
    event_review_refs = required_event_review_refs or set()
    gap_review_refs = review_refs - event_review_refs
    if any(
        item.status != "open" and set(item.evidence_refs).intersection(gap_review_refs)
        for item in requirements
    ):
        raise ValueError("an observation gap cannot support a projected finding")
    if review_refs and not set(review_evidence_refs).intersection(review_refs):
        raise ValueError("review must cite evidence produced by the last Focus")
    if outcome in ("resolved", "conflict") and not set(review_evidence_refs).intersection(event_review_refs):
        raise ValueError(f"{outcome} review must cite a validated event from the last Focus")
    hypotheses = tuple(AnswerHypothesis(f"H{index:02d}", option) for index, option in enumerate(options, 1))
    summary = " ".join(
        f"{item.requirement_id}: {item.objective}" for item in requirements
    )[:1600]
    return DecisionState(
        question=question, video_ids=video_ids, requirements=tuple(requirements),
        options=options, hypotheses=hypotheses,
        candidate_observations=tuple(frontier), summary=summary,
        protocol_notes=protocol_notes,
        review_outcome=outcome,
        review_reason=review_reason[:480],
        review_evidence_refs=review_evidence_refs,
    )


def _projection_video_scope(
    value: object,
    video_ids: tuple[str, ...],
    field: str,
) -> tuple[str, ...]:
    """Canonicalize only an unambiguous wrapper around a supplied video ID."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty array")
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            video_id = item
        elif (
            isinstance(item, dict)
            and set(item) == {"video_id"}
            and isinstance(item["video_id"], str)
        ):
            video_id = item["video_id"]
        else:
            raise ValueError(f"{field} must contain only supplied string video IDs")
        if video_id not in video_ids:
            raise ValueError(f"{field} selected an unknown video ID: {video_id}")
        if video_id not in normalized:
            normalized.append(video_id)
    return tuple(normalized)


def _projection_refs_from_indices(
    value: object,
    catalogue: tuple[str, ...],
    field: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{field} must be a list of at most {maximum} integers")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in value):
        raise ValueError(f"{field} must contain only integers")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must contain unique integers")
    invalid = [index for index in value if index < 1 or index > len(catalogue)]
    if invalid:
        raise ValueError(f"{field} selected unknown indices: {invalid}")
    return tuple(catalogue[index - 1] for index in value)


def _snapshot_reference_ids(snapshot: Sequence[Mapping[str, object]]) -> set[str]:
    """Return every provenance identifier that is visible in the snapshot.

    A current memory item has its own slot identifier and may expose source
    evidence identifiers beneath ``evidence``.  Both are legitimate current
    citations.  Restricting the projector to slot identifiers alone made a
    correctly copied nested evidence ID look fabricated after a zoom.
    """
    known: set[str] = set()
    for item in snapshot:
        if item.get("kind") == "memory_index":
            continue
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier:
            known.add(identifier)
        evidence = item.get("evidence", ())
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            known.update(str(ref) for ref in evidence if isinstance(ref, str) and ref)
    return known


def _snapshot_reference_catalogue(
    snapshot: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Assign stable call-local indices to every citable snapshot reference."""
    ordered: list[str] = []
    seen: set[str] = set()
    for item in snapshot:
        if item.get("kind") == "memory_index":
            continue
        identifier = item.get("id")
        values: list[object] = [identifier]
        evidence = item.get("evidence", ())
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            values.extend(evidence)
        for value in values:
            if isinstance(value, str) and value.strip() and value not in seen:
                ordered.append(value)
                seen.add(value)
    return tuple(ordered)


def _review_support_refs(
    snapshot: Sequence[Mapping[str, object]],
    receipt: FocusReviewReceipt,
) -> tuple[set[str], set[str]]:
    """Return citable refs derived from this Focus, split by event capability."""
    all_new = set(receipt.all_new_refs)
    new_events = set(receipt.new_event_refs)
    all_support = set(all_new)
    event_support = set(new_events)
    for item in snapshot:
        if item.get("kind") == "memory_index":
            continue
        evidence = item.get("evidence", ())
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            continue
        nested = {ref for ref in evidence if isinstance(ref, str)}
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            continue
        if nested.intersection(all_new):
            all_support.add(identifier)
        if nested.intersection(new_events):
            event_support.add(identifier)
    visible = set(_snapshot_reference_catalogue(snapshot))
    return all_support.intersection(visible), event_support.intersection(visible)


def _index_projection_snapshot(
    snapshot: Sequence[Mapping[str, object]],
    reference_indices: Mapping[str, int],
) -> tuple[dict[str, object], ...]:
    """Remove model-copyable citations and expose only call-local integers."""
    indexed: list[dict[str, object]] = []
    for source in snapshot:
        item = dict(source)
        if item.get("kind") == "memory_index":
            indexed.append(item)
            continue
        identifier = item.pop("id", None)
        evidence = item.pop("evidence", ())
        if isinstance(identifier, str) and identifier in reference_indices:
            item["citation_index"] = reference_indices[identifier]
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            item["source_evidence_indices"] = [
                reference_indices[ref] for ref in evidence
                if isinstance(ref, str) and ref in reference_indices
            ]
        indexed.append(item)
    return tuple(indexed)


def parse_reasoner_turn(text: str) -> ReasonerTurn:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("reasoner turn must be a JSON object")
    raw_updates = data.get("updates", [])
    if not isinstance(raw_updates, list):
        raise ValueError("reasoner updates must be an array")
    updates: list[RequirementUpdate] = []
    for position, item in enumerate(raw_updates):
        if not isinstance(item, dict):
            raise ValueError(f"reasoner updates[{position}] must be an object")
        updates.append(RequirementUpdate(
            str(item["requirement_id"]),
            str(item["status"]),
            _json_identifier_tuple(
                item.get("evidence_refs", []),
                f"reasoner updates[{position}].evidence_refs",
                maximum=8,
            ),
            item["summary"],
            _json_identifier_tuple(
                item.get("video_ids", []),
                f"reasoner updates[{position}].video_ids",
                maximum=64,
            ),
        ))
    raw_action = data.get("action")
    if not isinstance(raw_action, dict):
        raise ValueError("reasoner turn requires action object")
    request = None
    if raw_action.get("kind") == "PLAN":
        raw_request = raw_action.get("request")
        if not isinstance(raw_request, dict):
            raise ValueError("PLAN requires request object")
        request = FocusIntent(
            str(raw_request["requirement_id"]),
            _json_identifier_tuple(
                raw_request.get("slot_ids"),
                "reasoner PLAN slot_ids",
                maximum=64,
                allow_empty=False,
            ),
            str(raw_request["evidence_goal"]),
            str(raw_request.get("next_probe_goal", "")),
            str(raw_request.get("mode", "refine")),
        )
    raw_answer = raw_action.get("answer")
    answer = None
    if raw_answer is not None:
        answer = raw_answer if isinstance(raw_answer, str) else json.dumps(raw_answer, ensure_ascii=False)
    action = ReasonerAction(
        str(raw_action["kind"]), request=request, answer=answer,
        evidence_refs=_json_identifier_tuple(
            raw_action.get("evidence_refs", []),
            "reasoner action evidence_refs",
            maximum=8,
        ),
        reason=str(raw_action.get("reason", "")),
    )
    state_summary = data.get("state_summary", "")
    if not isinstance(state_summary, str):
        raise ValueError("reasoner state_summary must be a string")
    return ReasonerTurn(updates=tuple(updates), state_summary=state_summary, action=action)


def _json_identifier_tuple(
    value: object,
    field: str,
    *,
    maximum: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    """Parse one JSON identifier array without coercing nested structures."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array of strings")
    if len(value) > maximum:
        raise ValueError(f"{field} exceeds its {maximum}-item limit")
    result = tuple(value)
    _validate_identifier_tuple(result, field, allow_empty=allow_empty)
    return result


@dataclass(frozen=True)
class CycleConfig:
    visual_call_budget: int
    context_token_budget: int = 4096
    maximum_stagnant_turns: int = 2
    minimum_focus_stages: int = 0

    def __post_init__(self) -> None:
        if (self.visual_call_budget <= 0 or self.context_token_budget <= 0
                or self.maximum_stagnant_turns <= 0 or self.minimum_focus_stages < 0):
            raise ValueError("cycle budgets must be positive")
        if self.visual_call_budget < self.minimum_focus_stages:
            raise ValueError("visual call budget cannot fund the required focus stages")


@dataclass(frozen=True)
class ExplorationReservation:
    """A future *evidence question*, never a retained reasoning conclusion."""

    goal: str
    created_after_stage: int

    def __post_init__(self) -> None:
        if not self.goal.strip() or self.created_after_stage < 0:
            raise ValueError("exploration reservation needs a goal and stage")


@dataclass(frozen=True)
class ExplorationAgenda:
    """Protocol state for mandatory, sequential visual exploration.

    It deliberately contains no answer candidate, confidence, relation, or
    evidence interpretation.  Consequently evidence reprojection may discard
    the complete WorkingState while this scheduling contract remains valid.
    """

    minimum_focus_stages: int
    completed_focus_stages: int = 0
    reservation: ExplorationReservation | None = None

    def __post_init__(self) -> None:
        if self.minimum_focus_stages < 0 or self.completed_focus_stages < 0:
            raise ValueError("exploration stage counts must be non-negative")
        if self.completed_focus_stages > self.minimum_focus_stages:
            raise ValueError("completed focus stages exceed agenda")

    @property
    def remaining_stages(self) -> int:
        return self.minimum_focus_stages - self.completed_focus_stages

    @property
    def exploration_required(self) -> bool:
        return self.remaining_stages > 0

    @property
    def requires_follow_up_reservation(self) -> bool:
        return self.remaining_stages > 1 and self.reservation is None

    def reserve_from(self, request: FocusIntent) -> "ExplorationAgenda":
        if not self.requires_follow_up_reservation or not request.next_probe_goal.strip():
            return self
        return replace(
            self,
            reservation=ExplorationReservation(
                goal=request.next_probe_goal,
                created_after_stage=self.completed_focus_stages,
            ),
        )

    def complete_stage(self) -> "ExplorationAgenda":
        if not self.exploration_required:
            return self
        # The reservation belongs to the immediate following stage.  Keep a
        # reservation created by the stage that just completed, then clear it
        # after the next stage so stale probe text cannot masquerade as an
        # indefinitely unresolved finding.
        reservation = self.reservation
        if (
            reservation is not None
            and self.completed_focus_stages > reservation.created_after_stage
        ):
            reservation = None
        return replace(
            self,
            completed_focus_stages=self.completed_focus_stages + 1,
            reservation=reservation,
        )

    def to_mapping(self, *, visual_calls_used: int, visual_call_budget: int) -> dict[str, object]:
        reservation: dict[str, object] | None = None
        if self.reservation is not None:
            reservation = {
                "goal": self.reservation.goal,
                "created_after_stage": self.reservation.created_after_stage,
            }
        remaining_calls = max(0, visual_call_budget - visual_calls_used)
        # Preserve one call for each mandatory future stage, including this
        # one.  This turns the agenda into a real budget allocation rather
        # than a wish that a large first zoom may consume accidentally.
        stage_call_cap = remaining_calls // max(1, self.remaining_stages) if self.exploration_required else remaining_calls
        return {
            "minimum_focus_stages": self.minimum_focus_stages,
            "completed_focus_stages": self.completed_focus_stages,
            "remaining_focus_stages": self.remaining_stages,
            "phase": "mandatory_exploration" if self.exploration_required else "answer_or_continue",
            "require_follow_up_reservation": self.requires_follow_up_reservation,
            "reserved_complementary_probe": reservation,
            "visual_calls_used": visual_calls_used,
            "visual_calls_remaining": remaining_calls,
            "maximum_visual_calls_for_current_stage": stage_call_cap,
            "instruction": (
                "This is a sequential evidence-collection agenda, not a confidence check. "
                "Before its required stages are complete, select a PLAN rather than ANSWER. "
                "A new Focus result updates evidence memory; Reasoner decides whether a fresh ALIGN is useful."
            ),
        }


@dataclass(frozen=True)
class CycleResult:
    ledger: EvidenceLedger
    state: DecisionState
    turns: tuple[ReasonerTurn, ...]
    final_action: ReasonerAction
    visual_calls_used: int
    memory: SemanticZoomMemory = SemanticZoomMemory()
    exploration_agenda: ExplorationAgenda = ExplorationAgenda(0)
    termination_origin: Literal["reasoner", "runtime"] = "reasoner"
    alignment_history: tuple[AlignmentRecord, ...] = ()
    evidence_revision: int = 0
    projected_revision: int = 0
    aligned_revision: int = 0
    observation_receipts: tuple[ObservationReceipt, ...] = ()


TokenCounter = Callable[[str], int]


def conservative_token_count(text: str) -> int:
    """Conservative fallback when the deployed tokenizer is not injected."""
    return math.ceil(len(text) / 3)


class ContextBuilder:
    def __init__(self, *, token_budget: int, token_counter: TokenCounter = conservative_token_count) -> None:
        self.token_budget = token_budget
        self.token_counter = token_counter

    def render(
        self,
        state: DecisionState,
        ledger: EvidenceLedger,
        new_refs: Sequence[str],
        memory: SemanticZoomMemory = SemanticZoomMemory(),
        agenda: ExplorationAgenda | None = None,
        visual_calls_used: int = 0,
        visual_call_budget: int = 0,
        alignment_history: Sequence[AlignmentRecord] = (),
        evidence_revision: int = 0,
        projected_revision: int | None = None,
        aligned_revision: int = 0,
        observation_receipts: Sequence[ObservationReceipt] = (),
    ) -> str:
        current_projected_revision = (
            evidence_revision if projected_revision is None else projected_revision
        )
        if evidence_revision < 0 or current_projected_revision < 0 or aligned_revision < 0:
            raise ValueError("evidence-state revisions must be nonnegative")
        if current_projected_revision > evidence_revision:
            raise ValueError("projected revision cannot exceed evidence revision")
        if aligned_revision > evidence_revision:
            raise ValueError("aligned revision cannot exceed evidence revision")
        current_agenda = agenda or ExplorationAgenda(0)
        relevant = {ref for item in state.requirements for ref in item.evidence_refs}
        relevant.update(state.review_evidence_refs)
        relevant.update(new_refs)
        ledger_ids = {item.evidence_id for item in ledger.events}
        events = ledger.by_ids(sorted(relevant & ledger_ids)) if relevant & ledger_ids else ()
        events = tuple(item for item in events if item.evidence_id not in memory.superseded_evidence_refs)
        # Memory is the authoritative current evidence view. Do not repeat a
        # slot's provenance event as a second free-standing evidence item.
        represented = {ref for slot in memory.slots for ref in slot.evidence_refs}
        events = tuple(item for item in events if item.evidence_id not in represented)
        displayed_events = _select_context_events(events, state=state, maximum=8)

        # Pack against the actual counter rather than assuming that twelve
        # arbitrary descriptions fit. The selector is rerun at each size so
        # cited evidence and cross-video coverage retain priority.
        maximum_slots = min(12, len(memory.slots))
        ordered_memory = tuple(sorted(
            memory.slots,
            key=lambda value: (value.video_id, value.time_span, value.slot_id),
        ))
        memory_indices = {
            item.slot_id: index for index, item in enumerate(ordered_memory, 1)
        }
        limits = range(maximum_slots, 0, -1) if maximum_slots else (0,)
        smallest_used = 0
        for limit in limits:
            displayed_slots = (
                _select_memory_slots(memory, state=state, maximum=limit)
                if limit else ()
            )
            citable_refs: list[tuple[str, str, str]] = []
            seen_citations: set[str] = set()
            event_video = {item.evidence_id: item.video_id for item in ledger.events}
            for slot in displayed_slots:
                valid_events = [
                    ref for ref in slot.evidence_refs if ref in event_video
                ]
                # A gap-only slot remains focusable but can never support an
                # answer. Mixed slots are citable because they contain at
                # least one validated event.
                if valid_events and slot.slot_id not in seen_citations:
                    citable_refs.append((slot.slot_id, slot.video_id, "memory_slot"))
                    seen_citations.add(slot.slot_id)
                for ref in valid_events:
                    if ref not in seen_citations:
                        citable_refs.append((ref, event_video[ref], "event"))
                        seen_citations.add(ref)
            for event in displayed_events:
                if event.evidence_id not in seen_citations:
                    citable_refs.append((event.evidence_id, event.video_id, "event"))
                    seen_citations.add(event.evidence_id)
            citation_index = {
                ref: index for index, (ref, _, _) in enumerate(citable_refs, 1)
            }
            payload = {
                "question": state.question,
                "options": list(state.options),
                "option_labels": [chr(65 + index) for index in range(len(state.options))],
                "videos": list(state.video_ids),
                "answer_contract": {
                    "kind": _resolved_answer_kind(state),
                    "required_video_ids": list(state.required_answer_ids),
                    "choice_labels": [chr(65 + index) for index in range(len(state.options))],
                },
                "inference_protocol": list(state.protocol_notes),
                "synthesis_summary": state.summary[:800],
                "last_focus_review": {
                    "outcome": state.review_outcome,
                    "reason": state.review_reason[:480],
                    "evidence_indices": [
                        citation_index[ref] for ref in state.review_evidence_refs
                        if ref in citation_index
                    ],
                    "open_question_count": len(state.open_requirements),
                    "decision_eligible": (
                        not current_agenda.exploration_required
                        and current_projected_revision == evidence_revision
                    ),
                },
                "findings": [
                    {"index": index, "id": item.requirement_id,
                     "statement": item.objective[:240],
                     "videos": list(item.video_ids),
                     "evidence_indices": [
                         citation_index[ref] for ref in item.evidence_refs
                         if ref in citation_index
                     ][:4],
                     "evidence_count": len(item.evidence_refs)}
                    for index, item in enumerate(
                        (value for value in state.requirements if value.status != "open"), 1
                    )
                ],
                "open_questions": [
                    {"index": index, "id": item.requirement_id,
                     "question": item.objective[:240],
                     "videos": list(item.video_ids),
                     "related_evidence_indices": [
                         citation_index[ref] for ref in item.evidence_refs
                         if ref in citation_index
                     ][:4]}
                    for index, item in enumerate(state.open_requirements, 1)
                ],
                "contrastive_frontier": [
                    {
                        "id": item.candidate_id,
                        "question_id": item.related_requirement_id,
                        "videos": list(item.target_video_ids),
                        "observable": item.observable[:240],
                        "purpose": item.purpose[:240],
                    }
                    for item in state.candidate_observations
                ],
                "alignment_history": [
                    {
                        "round": item.round_id,
                        "evidence_revision": item.evidence_revision,
                        "requirement": item.requirement_id,
                        "objective": item.objective[:180],
                        "videos": list(item.video_ids),
                        "status": item.status,
                        "evidence_indices": [
                            citation_index[ref] for ref in item.evidence_refs
                            if ref in citation_index
                        ][:8],
                        "summary": item.summary[:240],
                    }
                    for item in alignment_history[-8:]
                ],
                "observation_receipts": [
                    {
                        "round": item.round_id,
                        "request_fingerprint": item.request_fingerprint,
                        "requirement": item.requirement_id,
                        "goal": item.evidence_goal[:240],
                        "targets": [
                            {"video": video_id, "time": [round(start, 3), round(end, 3)]}
                            for video_id, start, end in item.targets
                        ],
                        "outcome": item.outcome,
                        "event_count": len(item.event_refs),
                        "gap_count": len(item.gap_refs),
                    }
                    for item in observation_receipts
                ],
                "working_state": {
                    "evidence_revision": evidence_revision,
                    "projected_revision": current_projected_revision,
                    "aligned_revision": aligned_revision,
                    "current_evidence_is_projected": (
                        current_projected_revision == evidence_revision
                    ),
                    "current_evidence_is_explicitly_aligned": (
                        aligned_revision == evidence_revision
                    ),
                    "instruction": (
                        "Reprojection after FOCUS is automatic. ALIGN remains an optional "
                        "Reasoner action for a useful explicit evidence comparison."
                    ),
                },
                "new_or_relevant_evidence": [
                    {"evidence_index": citation_index.get(item.evidence_id),
                     "video": item.video_id, "time": item.time_span,
                     "direct_observation": item.description[:200],
                     "inferred_interpretation": item.interpretation[:120],
                     "uncertainty": item.uncertainty[:100],
                     "action": item.action, "round": item.round_id}
                    for item in displayed_events
                ],
                "answer_evidence_map_schema": ["index", "provenance_ref", "video", "kind"],
                "answer_evidence_map": [
                    [index, ref, video, kind]
                    for index, (ref, video, kind) in enumerate(citable_refs, 1)
                ],
                "recent_gaps": [
                    {"video": item.video_id, "time": item.time_span, "reason": item.reason[:160]}
                    for item in ledger.gaps if item.round_id == state.round_id
                ][:6],
                # Every current address remains navigable even when only a
                # small evidence-rich page can fit below. Compact rows are
                # [index, slot_id, video_id, start, end, level,
                #  visual_call_cost, semantic_tag].  The short stable index
                # is the model-facing selector; the executor resolves it back
                # to the full provenance identifier. Costs are visible for
                # every address, not
                # only the small detailed page, so PLAN can obey the hard
                # budget without guessing.
                "memory_level_legend": {"c": "coarse observed", "f": "fine observed", "u": "unobserved coverage"},
                "memory_map_schema": ["index", "slot_id", "video", "start", "end", "level", "visual_call_cost", "semantic_tag"],
                "memory_map": [
                    [index, item.slot_id, item.video_id, round(item.time_span[0], 3),
                     round(item.time_span[1], 3), item.resolution[0],
                     _slot_visual_cost(item),
                     item.content[:48] if limit >= 4 else ""]
                    for index, item in enumerate(ordered_memory, 1)
                ],
                # This is the detailed evidence page, not the full address
                # space. PLAN is allowed to choose any current memory_map ID.
                "available_focus_slots": [
                    {"slot_index": memory_indices[item.slot_id],
                     "slot_id": item.slot_id, "video": item.video_id,
                     "time": list(item.time_span), "resolution": item.resolution,
                     "content": item.content[:240],
                     "uncertainty": list(item.uncertainty[:3]),
                     "answer_evidence_index": citation_index.get(item.slot_id),
                     "source_evidence_indices": [
                         citation_index[ref] for ref in item.evidence_refs
                         if ref in citation_index
                     ][:4],
                     "focusable": True,
                     "estimated_visual_calls": max(
                         1, math.ceil((item.time_span[1] - item.time_span[0]) / 15.0),
                     )}
                    for item in displayed_slots
                ],
                "exploration_agenda": current_agenda.to_mapping(
                    visual_calls_used=visual_calls_used,
                    visual_call_budget=visual_call_budget,
                ),
                "instruction": (
                    "PLAN selects one integer question_index and integer slot_indices; the executor resolves provenance IDs. "
                    "ANSWER selects integer evidence_indices from answer_evidence_map. "
                    "Never copy or invent slot IDs, evidence IDs, or coordinates. "
                    "Use refine for coarse/fine and recover for coverage. Gaps are not negative evidence. "
                    "The sum of memory_map visual_call_cost values must fit the current-stage cap. "
                    "Each FOCUS must ask a locally observable question that could change the decision. "
                    "If last_focus_review is no_gain or conflict, test a competing interpretation or a different evidence region rather than repeating the same confirmation. "
                    "Use contrastive_frontier when it contains an executable discriminating probe. "
                    "After mandatory exploration and current-evidence reprojection, Reasoner may ANSWER despite residual open questions when no frontier probe is likely to change the best-supported answer. ALIGN is optional."
                ),
            }
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            smallest_used = self.token_counter(text)
            if smallest_used <= self.token_budget:
                return text
        raise ValueError(
            "reasoner context budget exceeded after deterministic compaction: "
            f"{smallest_used}>{self.token_budget} tokens"
        )


def initialize_semantic_memory(
    ledger: EvidenceLedger,
    *,
    coverage_regions: Mapping[str, Sequence[tuple[float, float]]] | None = None,
) -> SemanticZoomMemory:
    """Materialize warm SCAN evidence as the initial addressable coarse M0."""
    counters: dict[str, int] = {}
    slots: list[ZoomSlot] = []
    for event in sorted(
        (item for item in ledger.events if item.action == "SCAN"),
        key=lambda item: (item.video_id, item.time_span, item.evidence_id),
    ):
        counters[event.video_id] = counters.get(event.video_id, 0) + 1
        slots.append(ZoomSlot(
            slot_id=f"{event.video_id}.c{counters[event.video_id]:03d}",
            video_id=event.video_id, time_span=event.time_span, resolution="coarse",
            content=event.description, evidence_refs=(event.evidence_id,), revision=0,
            uncertainty=tuple(
                item for item in (
                    event.uncertainty,
                    f"Inferred interpretation (not direct evidence): {event.interpretation}"
                    if event.interpretation.strip() else "",
                ) if item.strip()
            ),
        ))
    if not slots:
        raise ValueError("semantic zoom memory requires warm SCAN evidence")
    for video_id, regions in sorted((coverage_regions or {}).items()):
        for index, (start, end) in enumerate(regions, 1):
            slots.append(ZoomSlot(
                slot_id=f"{video_id}.u{index:03d}", video_id=video_id,
                time_span=(start, end), resolution="coverage",
                content="Unobserved warm-scan interval; no positive or negative visual claim is available.",
                evidence_refs=(), revision=0,
                uncertainty=("This interval was not selected for warm VLM observation.",),
            ))
    return SemanticZoomMemory(slots=tuple(slots))


def _select_memory_slots(
    memory: SemanticZoomMemory, *, state: DecisionState, maximum: int,
) -> tuple[ZoomSlot, ...]:
    """Bound text context while retaining cited slots and video/time coverage."""
    if maximum <= 0:
        raise ValueError("memory slot context maximum must be positive")
    relevant_refs = {ref for item in state.requirements for ref in item.evidence_refs}
    relevant_refs.update(state.review_evidence_refs)
    selected: list[ZoomSlot] = []
    seen: set[str] = set()
    for slot in memory.slots:
        if ((slot.slot_id in relevant_refs or relevant_refs.intersection(slot.evidence_refs))
                and slot.slot_id not in seen):
            selected.append(slot); seen.add(slot.slot_id)
            if len(selected) >= maximum:
                return tuple(selected)
    by_video: dict[str, list[ZoomSlot]] = {}
    for slot in memory.slots:
        if slot.slot_id not in seen:
            by_video.setdefault(slot.video_id, []).append(slot)
    while len(selected) < maximum and by_video:
        for video_id in tuple(sorted(by_video)):
            values = by_video[video_id]
            if not values:
                del by_video[video_id]
                continue
            slot = values.pop(len(values) // 2)
            selected.append(slot); seen.add(slot.slot_id)
            if len(selected) >= maximum:
                break
    return tuple(sorted(selected, key=lambda item: (item.video_id, item.time_span, item.slot_id)))


def render_memory_snapshot(ledger: EvidenceLedger, memory: SemanticZoomMemory, *, maximum_items: int = 20) -> tuple[dict[str, object], ...]:
    """Expose a complete compact index plus a bounded evidence-rich page."""
    if maximum_items <= 0:
        raise ValueError("memory snapshot budget must be positive")
    # Projection sees only the current replace-in-place memory, never a stale
    # coarse ledger event alongside its fine replacement.
    ordered = sorted(memory.slots, key=lambda item: (item.video_id, item.time_span, item.slot_id))
    selected = list(ordered)
    if len(ordered) > maximum_items:
        fine = sorted(
            (item for item in ordered if item.resolution == "fine"),
            key=lambda item: (-item.revision, item.video_id, item.time_span),
        )
        selected = fine[: max(1, maximum_items // 2)]
        selected_ids = {item.slot_id for item in selected}
        by_video: dict[str, list[ZoomSlot]] = {}
        for item in ordered:
            if item.slot_id not in selected_ids:
                by_video.setdefault(item.video_id, []).append(item)
        while len(selected) < maximum_items and by_video:
            for video_id in tuple(sorted(by_video)):
                values = by_video[video_id]
                if not values:
                    del by_video[video_id]
                    continue
                selected.append(values.pop(len(values) // 2))
                if len(selected) >= maximum_items:
                    break
        selected = sorted(selected, key=lambda item: (item.video_id, item.time_span, item.slot_id))
    index_by_video: dict[str, list[list[object]]] = {}
    for item in ordered:
        index_by_video.setdefault(item.video_id, []).append([
            item.slot_id, round(item.time_span[0], 3), round(item.time_span[1], 3),
            item.resolution[0],
        ])
    index: dict[str, object] = {
        "id": "__memory_index__",
        "kind": "memory_index",
        "schema": ["slot_id", "start", "end", "level"],
        "level_legend": {"c": "coarse", "f": "fine", "u": "unobserved"},
        "videos": index_by_video,
    }
    details = tuple(
        {"id": item.slot_id, "kind": item.resolution, "video": item.video_id,
         "time": item.time_span, "content": item.content[:240],
         "uncertainty": item.uncertainty, "evidence": item.evidence_refs}
        for item in selected
    )
    return (index, *details)


class CycleRunner:
    """Mechanically executes decisions; it never chooses the next action."""

    def __init__(self, *, config: CycleConfig, context_builder: ContextBuilder | None = None) -> None:
        self.config = config
        self.context_builder = context_builder or ContextBuilder(token_budget=config.context_token_budget)

    def run(
        self,
        *,
        state: DecisionState,
        ledger: EvidenceLedger,
        reasoner: ReasonerBackend,
        observer: ObserverExecutor,
        projector: StateProjector | None = None,
        initial_new_refs: Sequence[str] = (),
        on_progress: Callable[
            [EvidenceLedger, DecisionState, ReasonerTurn, int, str], None
        ] | None = None,
        memory: SemanticZoomMemory = SemanticZoomMemory(),
    ) -> CycleResult:
        turns: list[ReasonerTurn] = []
        alignment_history: list[AlignmentRecord] = []
        observation_receipts: list[ObservationReceipt] = []
        state_projector = projector or ReceiptStateProjector()
        # ``state`` is the answer-free synthesis formed from warm evidence.
        # Every committed FOCUS is followed by a fresh projection from current
        # memory. ALIGN is a separate, optional Reasoner action.
        ledger.by_ids(initial_new_refs)
        if not memory.slots:
            memory = initialize_semantic_memory(ledger)
        pending_refs = tuple(initial_new_refs)
        visual_calls = 0
        stagnant = 0
        # The answer-free initial state is the projection of warm evidence.
        evidence_revision = 1 if (ledger.events or ledger.gaps or memory.slots) else 0
        projected_revision = evidence_revision
        aligned_revision = 0
        agenda = ExplorationAgenda(self.config.minimum_focus_stages)
        while True:
            try:
                context = self.context_builder.render(
                    state, ledger, pending_refs, memory, agenda,
                    visual_calls, self.config.visual_call_budget,
                    alignment_history=alignment_history,
                    evidence_revision=evidence_revision,
                    projected_revision=projected_revision,
                    aligned_revision=aligned_revision,
                    observation_receipts=observation_receipts,
                )
            except ValueError as error:
                raise RuntimeError(
                    f"reasoner context construction failed: {error}"
                ) from error
            turn = self._decide_validated(
                context, state, ledger, reasoner, memory, agenda, visual_calls,
                evidence_revision=evidence_revision,
                projected_revision=projected_revision,
                aligned_revision=aligned_revision,
                observation_receipts=observation_receipts,
            )
            state, changed = self._apply_turn(state, ledger, turn, memory)
            turns.append(turn)
            if on_progress is not None:
                on_progress(ledger, state, turn, visual_calls, "decision_validated")
            if turn.action.kind == "ANSWER":
                self._require_answerable(state, ledger, turn.action, memory)
                return CycleResult(
                    ledger, state, tuple(turns), turn.action, visual_calls,
                    memory, agenda, "reasoner", tuple(alignment_history),
                    evidence_revision, projected_revision, aligned_revision,
                    tuple(observation_receipts),
                )
            if turn.action.kind == "STOP":
                return CycleResult(
                    ledger, state, tuple(turns), turn.action, visual_calls,
                    memory, agenda, "reasoner", tuple(alignment_history),
                    evidence_revision, projected_revision, aligned_revision,
                    tuple(observation_receipts),
                )
            if turn.action.kind == "ALIGN":
                for update in turn.updates:
                    alignment_history.append(AlignmentRecord(
                        round_id=state.round_id,
                        evidence_revision=evidence_revision,
                        requirement_id=update.requirement_id,
                        objective=update.summary,
                        video_ids=update.video_ids,
                        status=update.status,
                        evidence_refs=update.evidence_refs,
                        summary=turn.state_summary,
                    ))
                aligned_revision = evidence_revision
                stagnant = 0 if changed else stagnant + 1
                if stagnant >= self.config.maximum_stagnant_turns:
                    return CycleResult(
                        ledger, state, tuple(turns),
                        ReasonerAction("STOP", reason="repeated no-gain ALIGN"),
                        visual_calls, memory, agenda, "runtime", tuple(alignment_history),
                        evidence_revision, projected_revision, aligned_revision,
                        tuple(observation_receipts),
                    )
                pending_refs = ()
                continue
            if visual_calls >= self.config.visual_call_budget:
                return CycleResult(
                    ledger, state, tuple(turns),
                    ReasonerAction("STOP", reason="visual budget exhausted"),
                    visual_calls, memory, agenda, "runtime", tuple(alignment_history),
                    evidence_revision, projected_revision, aligned_revision,
                    tuple(observation_receipts),
                )
            request = turn.action.request
            assert request is not None
            request_targets = _observation_targets(request, memory)
            fingerprint = _observation_fingerprint(request.evidence_goal, request_targets)
            try:
                results = tuple(observer.observe(request, state.round_id, ledger, memory))
            except (ValueError, RuntimeError) as error:
                raise RuntimeError(
                    f"Observer could not execute the validated visual request: {error}"
                ) from error
            if not results:
                raise RuntimeError("observer returned no observation batches")
            added: list[str] = []
            added_gaps: list[str] = []
            next_ledger = ledger
            zoom_items: list[tuple[ZoomDraft, str, tuple[str, ...], tuple[str, ...]]] = []
            for position, observed in enumerate(results, start=1):
                if isinstance(observed, ObservedBatch):
                    video_id, batch, draft = observed.video_id, observed.batch, observed.zoom_draft
                else:
                    video_id, batch = observed
                    draft = None
                next_ledger, ids, gap_ids = next_ledger.append(action="FOCUS", round_id=state.round_id, observation_id=f"l{position}", video_id=video_id, batch=batch)
                added.extend(ids)
                added_gaps.extend(gap_ids)
                if draft is not None:
                    descriptions = [item.description for item in batch.events]
                    gap_descriptions = [f"Observation gap: {item.reason}" for item in batch.observation_gaps]
                    content = " ".join((*descriptions, *gap_descriptions))
                    uncertainty = tuple(dict.fromkeys(
                        item.uncertainty for item in batch.events if item.uncertainty.strip()
                    )) + tuple(dict.fromkeys(
                        f"Inferred interpretation (not direct evidence): {item.interpretation}"
                        for item in batch.events if item.interpretation.strip()
                    )) + tuple(
                        f"Gap {item.start_seconds:.3f}-{item.end_seconds:.3f}s: {item.reason}"
                        for item in batch.observation_gaps
                    ) + tuple(
                        f"Unresolved: {item}" for item in batch.query_assessment.unresolved_needs
                    )
                    zoom_items.append((draft, content, (*ids, *gap_ids), uncertainty))
            next_agenda = agenda.reserve_from(request).complete_stage()
            if zoom_items:
                # A PLAN may contain several ordered slots. Group only child
                # batches that refine the same source region; the vault can
                # never blend children from distinct regions or videos.
                by_zoom: dict[str, list[tuple[ZoomDraft, str, tuple[str, ...], tuple[str, ...]]]] = {}
                for item in zoom_items:
                    by_zoom.setdefault(item[0].zoom_id, []).append(item)
                for items in by_zoom.values():
                    ordered = sorted(items, key=lambda item: item[0].slot_span)
                    memory = memory.apply_complete_zoom(
                        drafts=tuple(item[0] for item in ordered),
                        contents=tuple(item[1] for item in ordered),
                        evidence_refs=tuple(item[2] for item in ordered),
                        uncertainties=tuple(item[3] for item in ordered),
                    )
            # The transaction receipt is derived from executor inputs and
            # committed ledger output, so no model can manufacture its
            # provenance or reinterpret a gap as an event.
            receipt = ObservationReceipt(
                round_id=state.round_id,
                request_fingerprint=fingerprint,
                requirement_id=request.requirement_id,
                evidence_goal=request.evidence_goal,
                targets=request_targets,
                event_refs=tuple(added),
                gap_refs=tuple(added_gaps),
            )
            observation_receipts.append(receipt)
            visual_calls += len(results)
            evidence_revision += 1
            agenda = next_agenda
            ledger = next_ledger
            reserved_probe = (
                agenda.reservation.goal if agenda.reservation is not None else ""
            )
            review_receipt = FocusReviewReceipt(
                requirement_id=request.requirement_id,
                evidence_goal=request.evidence_goal,
                expected_complementary_probe=reserved_probe,
                proposed_next_probe=request.next_probe_goal,
                reserved_follow_up_probe=reserved_probe,
                new_event_refs=tuple(added),
                new_gap_refs=tuple(added_gaps),
                mandatory_stages_remaining=agenda.remaining_stages,
                visual_calls_remaining_after_focus=max(
                    0, self.config.visual_call_budget - visual_calls,
                ),
                force_open_requirement=agenda.exploration_required,
            )
            previous_state = state
            try:
                projected = state_projector.project(
                    question=state.question,
                    options=state.options,
                    video_ids=state.video_ids,
                    snapshot=render_memory_snapshot(ledger, memory),
                    receipt=review_receipt,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise RuntimeError(
                    f"evidence reprojection failed after committed FOCUS: {error}"
                ) from error
            if (
                projected.question != previous_state.question
                or projected.video_ids != previous_state.video_ids
                or projected.options != previous_state.options
            ):
                raise RuntimeError(
                    "evidence reprojection changed immutable sample identity"
                )
            state = replace(
                projected,
                round_id=previous_state.round_id,
                action_history=(*previous_state.action_history, fingerprint),
                protocol_notes=previous_state.protocol_notes,
                answer_kind=previous_state.answer_kind,
                required_answer_ids=previous_state.required_answer_ids,
            )
            projected_revision = evidence_revision
            pending_refs = tuple(added)
            stagnant = 0 if added else stagnant + 1
            if on_progress is not None:
                on_progress(ledger, state, turn, visual_calls, "evidence_committed")

    def _apply_turn(self, state: DecisionState, ledger: EvidenceLedger, turn: ReasonerTurn, memory: SemanticZoomMemory = SemanticZoomMemory()) -> tuple[DecisionState, bool]:
        known = ({item.evidence_id for item in ledger.events}
                 | {item.gap_id for item in ledger.gaps}
                 | {item.slot_id for item in memory.slots})
        if turn.action.kind != "ALIGN":
            if turn.updates:
                raise ValueError("only ALIGN may replace the evidence synthesis")
            revised = state.requirements
        else:
            revised_items: list[EvidenceRequirement] = []
            for item in turn.updates:
                if not item.summary.strip() or any(ref not in known for ref in item.evidence_refs):
                    raise ValueError("synthesis item must have a statement and current citations")
                if item.status != "open" and not item.evidence_refs:
                    raise ValueError("a finding needs positive evidence citations")
                revised_items.append(EvidenceRequirement(
                    item.requirement_id,
                    item.summary,
                    item.video_ids,
                    item.status,
                    item.evidence_refs,
                ))
            revised = tuple(revised_items)
        next_state = replace(
            state,
            requirements=tuple(revised),
            candidate_observations=(
                () if turn.action.kind == "ALIGN" else state.candidate_observations
            ),
            summary=turn.state_summary[:1600],
            round_id=state.round_id + 1,
        )
        return next_state, next_state.requirements != state.requirements

    def _decide_validated(
        self,
        context: str,
        state: DecisionState,
        ledger: EvidenceLedger,
        reasoner: ReasonerBackend,
        memory: SemanticZoomMemory = SemanticZoomMemory(),
        agenda: ExplorationAgenda = ExplorationAgenda(0),
        visual_calls_used: int = 0,
        *,
        evidence_revision: int = 0,
        projected_revision: int | None = None,
        aligned_revision: int = 0,
        observation_receipts: Sequence[ObservationReceipt] = (),
    ) -> ReasonerTurn:
        """Validate execution invariants without manufacturing semantic STOP.

        A model-selected STOP is a policy decision.  A malformed schema or an
        invalid provenance reference is an execution failure and must remain
        retryable by the outer runner; committing it as STOP previously made
        infrastructure errors look like legitimate no-answer predictions.
        """
        current_projected_revision = (
            evidence_revision if projected_revision is None else projected_revision
        )
        if current_projected_revision > evidence_revision:
            raise ValueError("projected revision cannot exceed evidence revision")
        if aligned_revision > evidence_revision:
            raise ValueError("aligned revision cannot exceed evidence revision")
        # Reprojection is a completed state transition before this decision.
        # ALIGN remains exclusively a Reasoner-selected action.
        turn = reasoner.decide(context)
        requested_alignment = turn.action.kind == "ALIGN" and not turn.updates
        if requested_alignment:
            turn = reasoner.align(context)
        turn = self._compile_plan_to_budget(
            turn, memory, agenda, visual_calls_used,
        )
        self._preflight_turn(
            state, ledger, turn, memory, agenda, visual_calls_used,
            evidence_revision=evidence_revision,
            projected_revision=current_projected_revision,
            observation_receipts=observation_receipts,
        )
        return turn

    def _compile_plan_to_budget(
        self,
        turn: ReasonerTurn,
        memory: SemanticZoomMemory,
        agenda: ExplorationAgenda,
        visual_calls_used: int,
    ) -> ReasonerTurn:
        """Compile a semantic PLAN into one executable acquisition stage.

        The Reasoner chooses relevance.  This generic budget module only
        normalizes the slot level and takes an in-order subset that fits the
        declared hard cap. Every reduction is recorded on the action; the
        next deterministic context render exposes the resulting evidence.
        """
        if turn.action.kind != "PLAN":
            return turn
        request = turn.action.request
        assert request is not None
        slots = memory.by_ids(request.slot_ids)
        requested_coverage = request.mode == "recover"
        matching = tuple(
            slot for slot in slots
            if (slot.resolution == "coverage") == requested_coverage
        )
        notes: list[str] = []
        mode = request.mode
        if not matching:
            matching = slots
            mode = "recover" if all(slot.resolution == "coverage" for slot in slots) else "refine"
            if any((slot.resolution == "coverage") != (mode == "recover") for slot in slots):
                matching = tuple(
                    slot for slot in slots
                    if (slot.resolution == "coverage") == (mode == "recover")
                )
            notes.append(f"normalized mode {request.mode}->{mode}")
        elif len(matching) != len(slots):
            notes.append("removed slots incompatible with requested refine/recover level")

        remaining = max(0, self.config.visual_call_budget - visual_calls_used)
        cap = remaining
        if agenda.exploration_required:
            cap = min(cap, remaining // agenda.remaining_stages)
        selected: list[ZoomSlot] = []
        used = 0
        for slot in matching:
            cost = _slot_visual_cost(slot)
            if used + cost > cap:
                continue
            selected.append(slot)
            used += cost
        if not selected:
            return turn
        selected_ids = tuple(slot.slot_id for slot in selected)
        if selected_ids != request.slot_ids:
            notes.append(
                f"reduced slots {len(request.slot_ids)}->{len(selected_ids)} to fit visual-call cap {cap}"
            )
        if not notes:
            return turn
        compiled_request = replace(request, slot_ids=selected_ids, mode=mode)
        compiled_reason = "; ".join(notes)
        if turn.action.reason.strip():
            compiled_reason = turn.action.reason.strip() + "; " + compiled_reason
        return replace(
            turn,
            action=replace(
                turn.action,
                request=compiled_request,
                reason="plan_compiler: " + compiled_reason,
            ),
        )

    def _preflight_turn(
        self,
        state: DecisionState,
        ledger: EvidenceLedger,
        turn: ReasonerTurn,
        memory: SemanticZoomMemory = SemanticZoomMemory(),
        agenda: ExplorationAgenda = ExplorationAgenda(0),
        visual_calls_used: int = 0,
        *,
        evidence_revision: int = 0,
        projected_revision: int | None = None,
        observation_receipts: Sequence[ObservationReceipt] = (),
    ) -> None:
        current_projected_revision = (
            evidence_revision if projected_revision is None else projected_revision
        )
        updates = {item.requirement_id: item for item in turn.updates}
        if len(updates) != len(turn.updates):
            raise ValueError("synthesis item IDs must be unique")
        if any(item.status not in ("open", "supported", "contradicted") for item in turn.updates):
            raise ValueError("reasoner requirement update has an invalid status")
        known = ({item.evidence_id for item in ledger.events}
                 | {item.gap_id for item in ledger.gaps}
                 | {item.slot_id for item in memory.slots})
        for item in turn.updates:
            if not item.summary.strip() or any(ref not in known for ref in item.evidence_refs):
                raise ValueError("requirement update must have nonempty summary and ledger citations")
            if item.status != "open" and not item.evidence_refs:
                raise ValueError("resolved requirement needs positive evidence citations")
        if turn.action.kind == "ALIGN":
            if not turn.updates:
                raise ValueError("ALIGN must produce a nonempty evidence synthesis")
            if agenda.exploration_required and not any(
                update.status == "open" for update in turn.updates
            ):
                raise ValueError("mandatory exploration ALIGN must retain an open question")
            gap_ids = {item.gap_id for item in ledger.gaps}
            for update in turn.updates:
                if not update.video_ids or any(video not in state.video_ids for video in update.video_ids):
                    raise ValueError("ALIGN synthesis item needs a valid video scope")
                if any(ref in gap_ids for ref in update.evidence_refs):
                    raise ValueError("ALIGN cannot use observation gaps as evidence")
                if update.status != "open" and not update.evidence_refs:
                    raise ValueError("ALIGN finding requires current positive evidence")
        elif turn.action.kind == "PLAN":
            request = turn.action.request
            assert request is not None
            self._validate_plan(request, state, memory)
            request_fingerprint = _observation_fingerprint(
                request.evidence_goal, _observation_targets(request, memory),
            )
            if any(
                receipt.outcome == "gap_only"
                and receipt.request_fingerprint == request_fingerprint
                for receipt in observation_receipts
            ):
                raise ValueError(
                    "PLAN exactly repeats a prior gap-only observation request"
                )
            estimate = _estimated_focus_calls(request, memory)
            remaining_calls = self.config.visual_call_budget - visual_calls_used
            if estimate > remaining_calls:
                raise ValueError(
                    f"PLAN exceeds remaining visual-call budget: estimated={estimate}, remaining={remaining_calls}"
                )
            if agenda.exploration_required:
                if turn.updates:
                    raise ValueError("mandatory exploration PLAN must not update requirement status")
                cap = (self.config.visual_call_budget - visual_calls_used) // agenda.remaining_stages
                if estimate > cap:
                    raise ValueError(
                        "PLAN consumes visual calls reserved for later mandatory exploration "
                        f"stages: estimated={estimate}, current-stage cap={cap}"
                    )
        elif turn.action.kind == "ANSWER":
            if agenda.exploration_required:
                raise ValueError(
                    f"exploration agenda incomplete: {agenda.remaining_stages} required FOCUS stage(s) remain"
                )
            if current_projected_revision != evidence_revision:
                raise ValueError(
                    "ANSWER cannot commit evidence newer than the current working-state projection"
                )
            self._validate_answer_contract(state, turn.action)

    @staticmethod
    def _require_answerable(state: DecisionState, ledger: EvidenceLedger, action: ReasonerAction, memory: SemanticZoomMemory = SemanticZoomMemory()) -> None:
        known = {item.evidence_id for item in ledger.events} | {item.slot_id for item in memory.slots}
        missing = [item for item in action.evidence_refs if item not in known]
        if missing:
            raise ValueError(f"answer cites evidence absent from current memory: {missing}")
        CycleRunner._validate_answer_contract(state, action)

    @staticmethod
    def _validate_answer_contract(state: DecisionState, action: ReasonerAction) -> None:
        answer = action.answer
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be nonempty text")
        kind = _resolved_answer_kind(state)
        if kind == "choice":
            allowed = {chr(65 + index) for index in range(len(state.options))}
            if answer.strip() not in allowed:
                raise ValueError(f"multiple-choice answer must be one option label: {sorted(allowed)}")
            return
        if kind == "sequence":
            values = tuple(item.strip() for item in answer.split("->") if item.strip())
            required = state.required_answer_ids or state.video_ids
            if values != tuple(dict.fromkeys(values)):
                raise ValueError("sequence answer repeats an input video ID")
            if len(values) != len(required) or set(values) != set(required):
                raise ValueError(
                    "sequence answer must contain every required video ID exactly once; "
                    "slot and evidence IDs are invalid"
                )
            return
        if kind == "interval":
            try:
                value = json.loads(answer)
            except json.JSONDecodeError as exc:
                raise ValueError("interval answer must be JSON [start_seconds, end_seconds]") from exc
            if (
                not isinstance(value, list) or len(value) != 2
                or not all(isinstance(item, (int, float)) for item in value)
                or float(value[0]) < 0 or float(value[1]) <= float(value[0])
            ):
                raise ValueError("interval answer must be ordered nonnegative [start_seconds, end_seconds]")
            return
        if re.fullmatch(r"(?:v\d+)(?:->v\d+)+", answer.strip()):
            raise ValueError("free-text answer unexpectedly used sequence syntax")

    @staticmethod
    def _validate_plan(request: FocusIntent, state: DecisionState, memory: SemanticZoomMemory) -> None:
        requirements = {item.requirement_id: item for item in state.requirements}
        requirement = requirements.get(request.requirement_id)
        if requirement is None or requirement.status != "open":
            raise ValueError("PLAN must target one currently open requirement")
        slots = memory.by_ids(request.slot_ids)
        if request.mode == "recover" and any(slot.resolution != "coverage" for slot in slots):
            raise ValueError("RECOVER PLAN may select only coverage slots")
        if request.mode == "refine" and any(slot.resolution == "coverage" for slot in slots):
            raise ValueError("REFINE PLAN may select only observed coarse/fine slots")
        if any(slot.video_id not in state.video_ids for slot in slots):
            raise ValueError("PLAN selected a slot outside the sample videos")


def _estimated_focus_calls(request: FocusIntent, memory: SemanticZoomMemory) -> int:
    """Bound a PLAN before execution while retaining Reasoner-selected scope.

    Focus uses 1 fps and at most 16 frames per Observer call.  This estimate
    is exact for the executor's current splitting policy and lets the agenda
    reserve visual capacity for the next required stage.
    """
    # A closed 1-fps interval with at most 16 frames spans 15 seconds; adjacent
    # calls share the boundary frame (see split_complete_zoom).
    return sum(_slot_visual_cost(slot) for slot in memory.by_ids(request.slot_ids))


def _slot_visual_cost(slot: ZoomSlot) -> int:
    return max(1, math.ceil((slot.time_span[1] - slot.time_span[0]) / 15.0))


def _resolved_answer_kind(state: DecisionState) -> AnswerKind:
    if state.answer_kind is not None:
        return state.answer_kind
    return "choice" if state.options else "free_text"


def _observation_targets(
    request: FocusIntent, memory: SemanticZoomMemory,
) -> tuple[tuple[str, float, float], ...]:
    """Resolve transient slot IDs to stable source/time acquisition targets."""
    return tuple(
        (slot.video_id, slot.time_span[0], slot.time_span[1])
        for slot in memory.by_ids(request.slot_ids)
    )


def _observation_fingerprint(
    evidence_goal: str,
    targets: Sequence[tuple[str, float, float]],
) -> str:
    """Identity of an acquisition request across replace-in-place slot IDs.

    Requirement IDs and memory slot IDs are intentionally excluded: ALIGN and
    semantic zoom may replace those names without changing what ObserverAgent
    would actually be asked to inspect. Target order is also irrelevant.
    """
    normalized_goal = " ".join(evidence_goal.casefold().split())
    if not normalized_goal or not targets:
        raise ValueError("observation fingerprint requires goal and targets")
    normalized_targets = sorted(
        (video_id, round(float(start), 3), round(float(end), 3))
        for video_id, start, end in targets
    )
    value = json.dumps(
        {"goal": normalized_goal, "targets": normalized_targets},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _select_context_events(
    events: Sequence[LedgerEvent], *, state: DecisionState, maximum: int
) -> tuple[LedgerEvent, ...]:
    """Keep decision context bounded while retaining temporal/video coverage.

    The ledger remains the complete provenance store.  This merely limits the
    set of descriptions exposed to a single text decision.  Requirement-cited
    evidence receives priority, then the remaining slots are distributed over
    videos and temporal order rather than taking the first batch wholesale.
    """
    if maximum <= 0:
        raise ValueError("context evidence maximum must be positive")
    if len(events) <= maximum:
        return tuple(events)
    by_id = {item.evidence_id: item for item in events}
    cited = [
        by_id[ref]
        for requirement in state.requirements
        for ref in requirement.evidence_refs
        if ref in by_id
    ]
    selected: list[LedgerEvent] = []
    seen: set[str] = set()
    for item in cited:
        if item.evidence_id not in seen and len(selected) < maximum:
            selected.append(item); seen.add(item.evidence_id)
    remaining = [item for item in events if item.evidence_id not in seen]
    by_video: dict[str, list[LedgerEvent]] = {}
    for item in remaining:
        by_video.setdefault(item.video_id, []).append(item)
    while len(selected) < maximum and by_video:
        for video_id in tuple(sorted(by_video)):
            values = by_video[video_id]
            if not values:
                del by_video[video_id]
                continue
            # Select a middle point first, then retain coverage as the list is
            # consumed; entries arrive from the ledger in temporal order.
            position = len(values) // 2
            item = values.pop(position)
            selected.append(item)
            if len(selected) >= maximum:
                break
    return tuple(sorted(selected, key=lambda item: (item.video_id, item.time_span, item.evidence_id)))
