"""Initial state formation: observer reports in, reasoner working memory out.

This stage replaces the answer-first ``FORM`` step.  It deliberately has no
``best_answer`` field, no ``DONE`` flag, and no model-reported confidence, so
early commitment is impossible by schema rather than discouraged by prompt.

The only thing the language model does is read the per-video reports and
propose entries.  Everything that downstream stages depend on is derived
mechanically here:

* every cited event ID and video ID is checked against the reports;
* ``working_relations[].status`` is derived from which videos actually
  contributed evidence, never reported by the model;
* only question-relevant relations and evidence gaps enter the state; the
  module does not manufacture an all-pairs comparison scaffold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from toolcvr.openai_compatible import APIRequestConfig, OpenAICompatibleTransport
from toolcvr.query import parse_json_object
from toolcvr.schemas import (
    CandidateRequest,
    InitialState,
    ObservedInformation,
    UnresolvedIssue,
    VideoEvidenceReport,
    WorkingRelation,
)


@dataclass(frozen=True)
class InitialStateConfig:
    maximum_information: int = 24
    maximum_relations: int = 24
    maximum_issues: int = 24
    maximum_requests: int = 12
    maximum_attempts: int = 2

    def __post_init__(self) -> None:
        limits = (
            self.maximum_information,
            self.maximum_relations,
            self.maximum_issues,
            self.maximum_requests,
            self.maximum_attempts,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("initial state limits must be positive")


class InitialStateBuilder(Protocol):
    def build(
        self,
        *,
        question: str,
        reports: Sequence[VideoEvidenceReport],
        options: Sequence[str] = (),
        protocol_notes: Sequence[str] = (),
    ) -> InitialState:
        """Return a complete, answer-free initial state."""


def required_pairs(video_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return unordered pairs for an explicitly chosen multi-video scope.

    This utility does not decide that all input videos must be compared. The
    caller has already selected ``video_ids`` as one relevant scope.
    """
    if len(video_ids) < 2:
        raise ValueError("initial state formation requires at least two videos")
    return tuple(
        (left, right)
        for index, left in enumerate(video_ids)
        for right in video_ids[index + 1 :]
    )


def normalize_initial_state(
    *,
    question: str,
    reports: Sequence[VideoEvidenceReport],
    payload: Mapping[str, Any],
    config: InitialStateConfig = InitialStateConfig(),
) -> InitialState:
    """Apply every mechanical check between raw model JSON and ``InitialState``."""
    if not question.strip():
        raise ValueError("initial state question must be nonempty")
    if len(reports) < 2:
        raise ValueError("initial state formation requires at least two video reports")
    video_ids = tuple(report.video_id for report in reports)
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("evidence report video IDs must be unique")

    known_videos = set(video_ids)
    events_by_id: dict[str, str] = {}
    for report in reports:
        for event in report.events:
            if event.event_id in events_by_id:
                raise ValueError(f"event ID is not unique across reports: {event.event_id}")
            events_by_id[event.event_id] = report.video_id

    notes: list[str] = []

    information = _read_information(
        payload.get("observed_information"),
        known_videos=known_videos,
        events_by_id=events_by_id,
        limit=config.maximum_information,
        notes=notes,
    )
    relations = _read_relations(
        payload.get("working_relations"),
        known_videos=known_videos,
        events_by_id=events_by_id,
        limit=config.maximum_relations,
        notes=notes,
    )
    issues = _read_issues(
        payload.get("unresolved_issues"),
        known_videos=known_videos,
        events_by_id=events_by_id,
        limit=config.maximum_issues,
        notes=notes,
    )
    # Relations are question-driven.  The initial state no longer fabricates
    # every N(N-1)/2 pair: a pair exists only when the model proposed a
    # relation or evidence issue that actually matters to the question.
    pairs = tuple(sorted(_covered_pairs(relations, issues)))
    gaps: tuple[tuple[str, str], ...] = ()

    relations, status_notes = derive_relation_statuses(
        relations,
        events_by_id=events_by_id,
        issues=issues,
    )
    notes.extend(status_notes)
    requests = _read_requests(
        payload.get("candidate_requests"),
        known_videos=known_videos,
        issue_ids={issue.issue_id for issue in issues},
        limit=config.maximum_requests,
        notes=notes,
    )

    return InitialState(
        question=question,
        video_ids=video_ids,
        observed_information=tuple(information),
        working_relations=tuple(relations),
        unresolved_issues=tuple(issues),
        candidate_requests=tuple(requests),
        required_pairs=pairs,
        coverage_gaps=gaps,
        validation_notes=tuple(notes),
    )


def render_initial_state(state: InitialState) -> str:
    """Deterministic text view of a validated initial state.

    Free-form model prose never reaches this renderer: every line is compiled
    from a checked field, exactly as the observer report text is.
    """
    coverage = state.to_dict()["coverage"]
    lines = [
        "INITIAL_STATE",
        f"COVERAGE pairs={coverage['covered_pair_count']}/{coverage['required_pair_count']} "
        f"unresolved_pairs={coverage['unresolved_pair_count']} "
        f"module_filled_pairs={coverage['module_filled_pair_count']}",
        "OBSERVED INFORMATION",
    ]
    if state.observed_information:
        lines.extend(
            f"- {item.information_id} [{','.join(item.video_ids)}] "
            f"{item.statement.strip()} | events={','.join(item.event_refs)}"
            for item in state.observed_information
        )
    else:
        lines.append("- none")
    lines.append("WORKING RELATIONS")
    if state.working_relations:
        lines.extend(
            f"- {item.relation_id} [{','.join(item.scope)}] {item.status} "
            f"{item.statement.strip()} | events={','.join(item.supporting_events)}"
            for item in state.working_relations
        )
    else:
        lines.append("- none")
    lines.append("UNRESOLVED ISSUES")
    if state.unresolved_issues:
        lines.extend(
            f"- {item.issue_id} [{','.join(item.scope)}] {item.issue.strip()} "
            f"| reason={item.reason.strip()} "
            f"| events={','.join(item.related_events) or 'none'}"
            for item in state.unresolved_issues
        )
    else:
        lines.append("- none")
    lines.append("CANDIDATE REQUESTS (not executed)")
    if state.candidate_requests:
        lines.extend(
            f"- {item.request_id} -> [{','.join(item.target_video_ids)}] "
            f"{item.observable.strip()} | purpose={item.purpose.strip()} "
            f"| issue={item.related_issue}"
            for item in state.candidate_requests
        )
    else:
        lines.append("- none")
    lines.append(
        "STATE_CAVEAT this state contains no answer and no completion decision; "
        "relation evidence status records citation coverage only and does not "
        "assert that a statement is true; relations marked evidence_partial or "
        "unresolved are open"
    )
    return "\n".join(lines)


# ── payload readers ──────────────────────────────────────────────────


def _read_information(
    raw: object,
    *,
    known_videos: set[str],
    events_by_id: Mapping[str, str],
    limit: int,
    notes: list[str],
) -> list[ObservedInformation]:
    result: list[ObservedInformation] = []
    seen: set[str] = set()
    for position, item in enumerate(_mapping_list(raw, "observed_information", notes)):
        if len(result) >= limit:
            notes.append(f"observed_information truncated at {limit}")
            break
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in seen:
            notes.append(f"observed_information[{position}] dropped: missing or duplicate id")
            continue
        videos = _known(item.get("videos"), known_videos, notes, f"information {identifier}.videos")
        refs = _known(item.get("event_refs"), set(events_by_id), notes, f"information {identifier}.event_refs")
        statement = str(item.get("statement", "")).strip()
        foreign = [ref for ref in refs if events_by_id[ref] not in set(videos)]
        if foreign:
            notes.append(
                f"information {identifier} dropped: event references outside videos {foreign}"
            )
            continue
        if not videos or not refs or not statement:
            notes.append(f"information {identifier} dropped: incomplete after validation")
            continue
        seen.add(identifier)
        result.append(
            ObservedInformation(
                information_id=identifier,
                video_ids=tuple(videos),
                event_refs=tuple(refs),
                statement=statement,
            )
        )
    return result


def _read_relations(
    raw: object,
    *,
    known_videos: set[str],
    events_by_id: Mapping[str, str],
    limit: int,
    notes: list[str],
) -> list[WorkingRelation]:
    result: list[WorkingRelation] = []
    seen: set[str] = set()
    for position, item in enumerate(_mapping_list(raw, "working_relations", notes)):
        if len(result) >= limit:
            notes.append(f"working_relations truncated at {limit}")
            break
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in seen:
            notes.append(f"working_relations[{position}] dropped: missing or duplicate id")
            continue
        scope = _known(item.get("scope"), known_videos, notes, f"relation {identifier}.scope")
        evidence = _known(
            item.get("supporting_events"),
            set(events_by_id),
            notes,
            f"relation {identifier}.supporting_events",
        )
        statement = str(item.get("statement", "")).strip()
        foreign = [ref for ref in evidence if events_by_id[ref] not in set(scope)]
        if foreign:
            notes.append(
                f"relation {identifier} dropped: supporting events outside scope {foreign}"
            )
            continue
        if len(scope) < 2 or not evidence or not statement:
            # A relation with no surviving evidence is not a relation; the pair
            # it would have covered is re-filled as an unresolved issue instead.
            notes.append(f"relation {identifier} dropped: incomplete after validation")
            continue
        seen.add(identifier)
        result.append(
            WorkingRelation(
                relation_id=identifier,
                scope=tuple(scope),
                statement=statement,
                supporting_events=tuple(evidence),
                # Placeholder only: ``_derive_status`` overwrites this for every
                # relation before the state is constructed.
                status="evidence_partial",
            )
        )
    return result


def _read_issues(
    raw: object,
    *,
    known_videos: set[str],
    events_by_id: Mapping[str, str],
    limit: int,
    notes: list[str],
) -> list[UnresolvedIssue]:
    result: list[UnresolvedIssue] = []
    seen: set[str] = set()
    for position, item in enumerate(_mapping_list(raw, "unresolved_issues", notes)):
        if len(result) >= limit:
            notes.append(f"unresolved_issues truncated at {limit}")
            break
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in seen:
            notes.append(f"unresolved_issues[{position}] dropped: missing or duplicate id")
            continue
        raw_scope = item.get("scope")
        scope = _known(raw_scope, known_videos, notes, f"issue {identifier}.scope")
        if isinstance(raw_scope, list) and any(
            isinstance(video_id, str) and video_id not in known_videos
            for video_id in raw_scope
        ):
            notes.append(f"issue {identifier} dropped: scope contained an unknown video")
            continue
        events = _known(
            item.get("related_events") or [],
            set(events_by_id),
            notes,
            f"issue {identifier}.related_events",
        )
        foreign = [ref for ref in events if events_by_id[ref] not in set(scope)]
        if foreign:
            notes.append(
                f"issue {identifier}.related_events dropped outside scope {foreign}"
            )
            events = [ref for ref in events if ref not in foreign]
        issue = str(item.get("issue", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not scope or not issue or not reason:
            notes.append(f"issue {identifier} dropped: incomplete after validation")
            continue
        seen.add(identifier)
        result.append(
            UnresolvedIssue(
                issue_id=identifier,
                scope=tuple(scope),
                issue=issue,
                reason=reason,
                related_events=tuple(events),
            )
        )
    return result


def _read_requests(
    raw: object,
    *,
    known_videos: set[str],
    issue_ids: set[str],
    limit: int,
    notes: list[str],
) -> list[CandidateRequest]:
    result: list[CandidateRequest] = []
    seen: set[str] = set()
    for position, item in enumerate(_mapping_list(raw, "candidate_requests", notes)):
        if len(result) >= limit:
            notes.append(f"candidate_requests truncated at {limit}")
            break
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in seen:
            notes.append(f"candidate_requests[{position}] dropped: missing or duplicate id")
            continue
        targets = _known(
            item.get("target_videos"), known_videos, notes, f"request {identifier}.target_videos"
        )
        observable = str(item.get("observable", "")).strip()
        purpose = str(item.get("purpose", "")).strip()
        related = str(item.get("related_issue", "")).strip()
        if not targets or not observable or not purpose:
            notes.append(f"request {identifier} dropped: incomplete after validation")
            continue
        if related not in issue_ids:
            notes.append(f"request {identifier} dropped: unknown related_issue {related!r}")
            continue
        seen.add(identifier)
        result.append(
            CandidateRequest(
                request_id=identifier,
                target_video_ids=tuple(targets),
                observable=observable,
                purpose=purpose,
                related_issue=related,
            )
        )
    return result


def auto_issue_for(
    pair: tuple[str, str],
    taken_ids: set[str],
) -> UnresolvedIssue:
    """Repair a previously tracked pair made uncovered by a state revision.

    The id scheme is deterministic on purpose: the versioned update layer runs
    this same function, so a replay from ``WorkingState_0`` reproduces the exact
    same auto ids instead of inventing new ones.
    """
    counter = len(taken_ids) + 1
    issue_id = f"U-auto-{counter:02d}"
    while issue_id in taken_ids:
        counter += 1
        issue_id = f"U-auto-{counter:02d}"
    return UnresolvedIssue(
        issue_id=issue_id,
        scope=pair,
        issue=(
            "The relationship between "
            f"{pair[0]} and {pair[1]} was not established."
        ),
        reason=(
            "A state revision removed the last working relation or "
            "unresolved issue for this previously tracked pair."
        ),
        filled_by_module=True,
    )


def derive_relation_statuses(
    relations: Sequence[WorkingRelation],
    *,
    events_by_id: Mapping[str, str],
    issues: Sequence[UnresolvedIssue],
) -> tuple[list[WorkingRelation], list[str]]:
    """Return relations with derived evidence status, plus audit notes.

    This is the only place a relation's status is decided.  Both the initial
    build and every later state revision route through it, so a status can never
    be introduced by a model or by a Review delta.

    The derived value never asserts that the statement is true.  All the module
    can see is which videos in scope contributed a cited event, and whether the
    reasoner has itself named any part of that scope as still undetermined.
    ``evidence_complete`` is withheld unless both hold.
    """
    issue_scopes = {frozenset(issue.scope) for issue in issues}
    derived: list[WorkingRelation] = []
    notes: list[str] = []
    partial = 0
    contested = 0
    for relation in relations:
        scope = frozenset(relation.scope)
        if scope in issue_scopes:
            # The model raised an unresolved issue over exactly this scope, so
            # its own relation on the same scope must not be treated as covered.
            status = "unresolved"
        else:
            # Citation coverage is necessary but not sufficient.  An issue over
            # *part* of this scope contests the relation as much as one over all
            # of it, so the test is overlap and not equality: with equality, a
            # relation spanning {v1..v4} kept evidence_complete while an issue
            # named {v1,v2,v4} as the very videos still undetermined.
            overlaps = any(scope & other for other in issue_scopes)
            evidenced = {events_by_id[item] for item in relation.supporting_events}
            status = (
                "evidence_complete"
                if scope <= evidenced and not overlaps
                else "evidence_partial"
            )
            if status == "evidence_partial":
                partial += 1
                if overlaps:
                    contested += 1
        derived.append(
            WorkingRelation(
                relation_id=relation.relation_id,
                scope=relation.scope,
                statement=relation.statement,
                supporting_events=relation.supporting_events,
                status=status,
            )
        )
    if partial:
        notes.append(
            f"{partial} relations are evidence_partial: at least one video in "
            "scope either contributed no cited event or is still named by an "
            "unresolved issue"
        )
    if contested:
        notes.append(
            f"{contested} of those overlap an unresolved issue, so citing one "
            "event per video in scope was not enough to call them covered"
        )
    return derived, notes


def _covered_pairs(
    relations: Sequence[WorkingRelation],
    issues: Sequence[UnresolvedIssue],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for scope in (
        *(item.scope for item in relations),
        *(item.scope for item in issues),
    ):
        for index, left in enumerate(scope):
            for right in scope[index + 1 :]:
                pairs.add(_ordered(left, right))
    return pairs


def _ordered(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _mapping_list(raw: object, name: str, notes: list[str]) -> list[Mapping[str, Any]]:
    if raw is None:
        notes.append(f"{name} missing from model output")
        return []
    if not isinstance(raw, list):
        notes.append(f"{name} is not a list and was ignored")
        return []
    result: list[Mapping[str, Any]] = []
    for position, item in enumerate(raw):
        if not isinstance(item, Mapping):
            notes.append(f"{name}[{position}] is not an object and was ignored")
            continue
        result.append(item)
    return result


def _known(
    raw: object,
    known: set[str],
    notes: list[str],
    label: str,
) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        notes.append(f"{label} is not a string list and was ignored")
        return []
    result: list[str] = []
    for item in raw:
        if item in known:
            if item not in result:
                result.append(item)
        else:
            notes.append(f"{label} dropped unknown reference {item!r}")
    return result


# ── builders ─────────────────────────────────────────────────────────


class DeterministicMockInitialStateBuilder:
    """CPU-only builder used by the smoke test and the mock pipeline path.

    It asserts no relation and emits one multi-video unresolved issue, which is
    the conservative reading of reports no model has interpreted yet.
    """

    def __init__(self, *, config: InitialStateConfig = InitialStateConfig()) -> None:
        self.config = config

    def build(
        self,
        *,
        question: str,
        reports: Sequence[VideoEvidenceReport],
        options: Sequence[str] = (),
        protocol_notes: Sequence[str] = (),
    ) -> InitialState:
        video_ids = tuple(report.video_id for report in reports)
        pairs = required_pairs(video_ids)
        observed = [
            ObservedInformation(
                information_id=f"I{position:02d}",
                video_ids=(report.video_id,),
                event_refs=(report.events[0].event_id,),
                statement=report.events[0].description.strip(),
            )
            for position, report in enumerate(reports, start=1)
        ]
        issues = [
            UnresolvedIssue(
                issue_id="U01",
                scope=video_ids,
                issue="The question-relevant relationship has not been interpreted yet.",
                reason=(
                    "The deterministic mock builder performs no semantic "
                    "interpretation and asserts no relation."
                ),
            )
        ]
        requests = [
            CandidateRequest(
                request_id="Q01",
                target_video_ids=video_ids,
                observable="Compare the observed states across the cited videos.",
                purpose=f"Resolve {issues[0].issue_id}.",
                related_issue=issues[0].issue_id,
            )
        ]
        return InitialState(
            question=question,
            video_ids=video_ids,
            observed_information=tuple(observed),
            working_relations=(),
            unresolved_issues=tuple(issues),
            candidate_requests=tuple(requests),
            required_pairs=pairs,
            coverage_gaps=(),
            validation_notes=("mock builder: no relation was asserted",),
        )


@dataclass(frozen=True)
class OpenAICompatibleInitialStateBuilder:
    base_url: str
    model: str
    api_key_env: str = "LLM_API_KEY"
    timeout_seconds: float = 120.0
    maximum_tokens: int = 2048
    config: InitialStateConfig = InitialStateConfig()

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("initial state builder requires a base URL and a model")
        if self.maximum_tokens <= 0:
            raise ValueError("initial state maximum_tokens must be positive")

    def build(
        self,
        *,
        question: str,
        reports: Sequence[VideoEvidenceReport],
        options: Sequence[str] = (),
        protocol_notes: Sequence[str] = (),
    ) -> InitialState:
        raw = self._request(question, reports, (), options, protocol_notes)
        data = parse_json_object(raw)
        return normalize_initial_state(
            question=question,
            reports=reports,
            payload=data,
            config=self.config,
        )

    def _request(
        self,
        question: str,
        reports: Sequence[VideoEvidenceReport],
        feedback: Sequence[str],
        options: Sequence[str] = (),
        protocol_notes: Sequence[str] = (),
    ) -> str:
        if self.maximum_tokens <= 0:
            raise ValueError("initial state maximum_tokens must be positive")
        user_payload: dict[str, object] = {
            "question": question,
            "options": list(options),
            "inference_protocol": list(protocol_notes),
            "videos": [
                {
                    "video_id": report.video_id,
                    "events": [
                        {
                            "event_id": event.event_id,
                            "time_span": [event.start_seconds, event.end_seconds],
                            "description": event.description,
                            "inferred_interpretation": event.interpretation,
                            "uncertainty": event.uncertainty,
                        }
                        for event in report.events
                    ],
                    "local_evidence_summary": _local_evidence_summary(report),
                    "observation_gaps": [
                        gap.to_mapping() for gap in report.observation_gaps
                    ],
                }
                for report in reports
            ],
        }
        if feedback:
            user_payload["previous_attempt_rejected"] = list(feedback)
        return OpenAICompatibleTransport(
            base_url=self.base_url,
            model=self.model,
            api_key_env=self.api_key_env,
        ).complete(
            role="initial_state",
            config=APIRequestConfig(
                maximum_tokens=self.maximum_tokens,
                timeout_seconds=self.timeout_seconds,
            ),
            messages=[
                {"role": "system", "content": _INITIAL_STATE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
        )


_INITIAL_STATE_SYSTEM_PROMPT = """You build the initial working state of a multi-video reasoner from
timestamped event reports. You are not answering the question.

Hard prohibitions:
- Do NOT answer the original question.
- Do NOT output a best answer, a final answer, a ranking, or a preferred option.
- Do NOT decide whether reasoning is complete.
- Do NOT output confidence, probability, or a status field.

Return one JSON object with exactly these four lists:
{
  "observed_information": [
    {"id": "I01", "videos": ["v2"], "event_refs": ["v2-b1-e1"], "statement": "one observed fact relevant to the question"}
  ],
  "working_relations": [
    {"id": "R01", "scope": ["v2", "v5"], "statement": "relation between these videos", "supporting_events": ["v2-b1-e1", "v5-b2-e3"]}
  ],
  "unresolved_issues": [
    {"id": "U01", "scope": ["v1", "v4"], "issue": "what remains undetermined", "reason": "why the reports do not settle it", "related_events": ["v1-b2-e2"]}
  ],
  "candidate_requests": [
    {"id": "Q01", "target_videos": ["v2", "v5"], "observable": "one visually answerable comparison", "purpose": "which issue this would resolve", "related_issue": "U01"}
  ]
}

Rules:
- inference_protocol contains dataset-adapter facts, clock semantics, identity
  conventions, counting semantics, and output constraints. Treat these as
  authoritative evidence-handling rules, not as hints about the answer.
- Use every supplied option symmetrically to identify what evidence would
  distinguish the candidates. Never prefer an option during initial state.
- local_evidence_summary is a deterministic aggregation of batch-local hints.
  It says which existing events locally contributed to the question; it does
  not judge whether the video set or the whole sample has enough evidence.
- observation_gaps must never be copied directly into unresolved_issues. First
  compare the supplied events across all videos. Create an issue only for a
  specific fact that remains missing after that comparison. Never infer absence
  from a sparse scan or an observation gap.
- Every id in event_refs, supporting_events, and related_events must be copied
  exactly from the supplied events. Never invent an event id.
- Every video id in videos, scope, and target_videos must come from the supplied list.
- observed_information records what has already been observed. It is not a conclusion.
- Each event description contains direct visual observation. An
  inferred_interpretation is explicitly non-direct and may motivate an unresolved
  issue, but it cannot by itself support observed_information or a working_relation.
- A working_relation must compare at least two videos and must cite events from
  the reports. Do not report its strength; the harness derives that from whether
  each video in scope contributed evidence.
- Build only relations and unresolved issues that are necessary for the
  question. Do not enumerate every video pair by default.
- An unresolved issue may concern one video or multiple videos. Its scope must
  contain exactly the videos whose missing evidence the issue describes.
- unresolved_issues must describe a gap in the evidence, never a doubt about a
  preferred answer. Write "the relative stage relationship between v2 and v3
  remains unresolved", never "if the answer is v2 before v3, check v2".
- candidate_requests form a pool only. Do not execute them, do not rank them, and
  do not pick one. Each request must reference an existing unresolved issue.
- related_events may be empty when no specific event is at fault.
- Keep statements to one sentence. Do not restate the whole report.
- Output compact JSON only, without markdown or explanatory prose."""


def _local_evidence_summary(report: VideoEvidenceReport) -> dict[str, object]:
    """Aggregate local labels without promoting their free-form needs globally.

    Only event ids leave this function, never a count.  "N of this video's
    batches found no local match" is a statement about how the scan went, not
    about the video, and a count is the shape a reasoner lifts most easily into
    a sample-scope verdict.  Ids at least name evidence; a count names only an
    absence, and an absence measured over a CLIP-filtered sparse scan is not a
    fact about the video.  The gaps themselves still travel, in
    ``observation_gaps``, anchored to their frames and timestamps.
    """

    direct = {
        event_id
        for assessment in report.query_assessments
        if assessment.relevance == "direct"
        for event_id in assessment.evidence_event_ids
    }
    indirect = {
        event_id
        for assessment in report.query_assessments
        if assessment.relevance == "indirect"
        for event_id in assessment.evidence_event_ids
    } - direct
    return {
        "direct_event_ids": sorted(direct),
        "indirect_event_ids": sorted(indirect),
        "scope": "batch-local hints only; not a global evidence verdict",
    }
