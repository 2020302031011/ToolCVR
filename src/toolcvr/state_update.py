"""The versioned, non-monotonic working-state update layer.

``apply_state_delta`` is the only way a state changes after it is formed.  A
Review round emits a :class:`StateDelta`; the module applies it, repairs
coverage, re-derives every relation status, and appends the audit trail.

The design rule from the spec is:

    WorkingState is non-monotonic, but revision history is append-only.

so ``observed_information`` and the history only grow, while relations and
issues may be revised, retracted, re-opened and resolved.

Three invariants are enforced here rather than requested of a model:

1. ``RESOLVE`` must name a real relation that covers the issue's pair.
2. ``UPDATE`` may not change a relation's ``scope``.
3. A relation's ``status`` is only ever produced by
   :func:`~toolcvr.initial_state.derive_relation_statuses`.

``apply_state_delta`` is a pure function of ``(state, delta)``: identical inputs
produce an identical state, including module-generated issue ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toolcvr.initial_state import auto_issue_for, derive_relation_statuses
from toolcvr.schemas import (
    AddInformation,
    AddIssue,
    AddRelation,
    InitialState,
    ObservedInformation,
    ReopenIssue,
    ResolveIssue,
    RetractIssue,
    RetractRelation,
    StateDelta,
    StateRevision,
    UnresolvedIssue,
    UpdateRelation,
    WorkingRelation,
    WorkingState,
    ordered_pair,
)


def apply_state_delta(state: WorkingState, delta: StateDelta) -> WorkingState:
    """Apply one Review round atomically and return the next working state."""
    if delta.round_id != state.round_id + 1:
        raise ValueError(
            f"state delta round_id {delta.round_id} does not follow "
            f"working state round {state.round_id}"
        )
    _reject_duplicate_targets(delta)

    known_videos = set(state.video_ids)
    events = state.events_by_id()
    known_events = set(events)

    information = list(state.observed_information)
    relations = list(state.working_relations)
    issues = list(state.unresolved_issues)
    resolved = list(state.resolved_issues)
    pending: list[_Pending] = []

    for operation in delta.operations:
        pending.append(
            _apply_operation(
                operation,
                known_videos=known_videos,
                known_events=known_events,
                events=events,
                information=information,
                relations=relations,
                issues=issues,
                resolved=resolved,
            )
        )

    # Preserve only scopes already present in the question-driven state. This
    # never expands to every possible input-video pair.
    gaps = _uncovered_pairs(state.required_pairs, relations, issues)
    if gaps:
        taken = {item.issue_id for item in issues} | {item.issue_id for item in resolved}
        for pair in gaps:
            issue = auto_issue_for(pair, taken)
            taken.add(issue.issue_id)
            issues.append(issue)
            pending.append(
                _Pending(
                    operation="ADD",
                    target_kind="issue",
                    target_id=issue.issue_id,
                    previous_value=None,
                    evidence_refs=(),
                    note="module repaired a video pair left uncovered by this delta",
                )
            )

    relations, status_notes = derive_relation_statuses(
        relations,
        events_by_id=events,
        issues=issues,
    )
    kept_requests = tuple(
        request
        for request in state.candidate_requests
        if request.related_issue in {item.issue_id for item in issues}
    )
    notes = [
        *state.validation_notes,
        *status_notes,
        *(
            [f"dropped {len(state.candidate_requests) - len(kept_requests)} candidate "
             "requests whose issue is no longer active"]
            if len(kept_requests) != len(state.candidate_requests)
            else []
        ),
    ]

    next_state = InitialState(
        question=state.question,
        video_ids=state.video_ids,
        observed_information=tuple(information),
        working_relations=tuple(relations),
        unresolved_issues=tuple(issues),
        candidate_requests=kept_requests,
        required_pairs=state.required_pairs,
        coverage_gaps=tuple(gaps),
        validation_notes=tuple(notes),
    )

    revisions = tuple(
        StateRevision(
            round_id=delta.round_id,
            operation=item.operation,
            target_kind=item.target_kind,
            target_id=item.target_id,
            previous_value=item.previous_value,
            new_value=_snapshot(
                item.target_kind,
                item.target_id,
                information=information,
                relations=relations,
                issues=issues,
                resolved=resolved,
            ),
            evidence_refs=item.evidence_refs,
            note=item.note or delta.note,
        )
        for item in pending
    )

    return WorkingState(
        state=next_state,
        round_id=delta.round_id,
        event_videos=state.event_videos,
        resolved_issues=tuple(resolved),
        revision_history=(*state.revision_history, *revisions),
        module_filled_pairs=(
            *state.module_filled_pairs,
            *(pair for pair in gaps if pair not in state.module_filled_pairs),
        ),
    )


@dataclass(frozen=True)
class _Pending:
    """One edit awaiting its post-round snapshot and history entry."""

    operation: str
    target_kind: str
    target_id: str
    previous_value: dict[str, Any] | None
    evidence_refs: tuple[str, ...]
    note: str = ""


def _apply_operation(
    operation: object,
    *,
    known_videos: set[str],
    known_events: set[str],
    events: dict[str, str],
    information: list[ObservedInformation],
    relations: list[WorkingRelation],
    issues: list[UnresolvedIssue],
    resolved: list[UnresolvedIssue],
) -> _Pending:
    if isinstance(operation, AddInformation):
        return _add_information(
            operation,
            known_videos=known_videos,
            known_events=known_events,
            events=events,
            information=information,
        )
    if isinstance(operation, AddRelation):
        return _add_relation(
            operation,
            known_videos=known_videos,
            known_events=known_events,
            relations=relations,
        )
    if isinstance(operation, AddIssue):
        return _add_issue(
            operation,
            known_videos=known_videos,
            issues=issues,
            resolved=resolved,
        )
    if isinstance(operation, UpdateRelation):
        return _update_relation(
            operation,
            known_events=known_events,
            relations=relations,
        )
    if isinstance(operation, RetractRelation):
        return _retract_relation(operation, relations=relations)
    if isinstance(operation, RetractIssue):
        return _retract_issue(operation, issues=issues)
    if isinstance(operation, ReopenIssue):
        return _reopen_issue(operation, issues=issues, resolved=resolved)
    if isinstance(operation, ResolveIssue):
        return _resolve_issue(operation, relations=relations, issues=issues, resolved=resolved)
    raise ValueError(f"unsupported state operation: {operation!r}")


def _add_information(
    operation: AddInformation,
    *,
    known_videos: set[str],
    known_events: set[str],
    events: dict[str, str],
    information: list[ObservedInformation],
) -> _Pending:
    identifier = operation.information_id.strip()
    if not identifier:
        raise ValueError("added information needs a nonempty id")
    if identifier in {item.information_id for item in information}:
        raise ValueError(f"information id already exists: {identifier}")
    if not operation.video_ids:
        raise ValueError(f"information {identifier} must name at least one video")
    _require_subset(operation.video_ids, known_videos, f"information {identifier}.videos")
    _require_subset(
        operation.event_refs, known_events, f"information {identifier}.event_refs"
    )
    if not operation.event_refs:
        raise ValueError(f"information {identifier} must cite at least one event")
    if not operation.statement.strip():
        raise ValueError(f"information {identifier} statement must be nonempty")
    owners = {events[ref] for ref in operation.event_refs}
    if not owners <= set(operation.video_ids):
        raise ValueError(
            f"information {identifier} cites events from videos "
            f"{sorted(owners - set(operation.video_ids))} it does not name"
        )
    information.append(
        ObservedInformation(
            information_id=identifier,
            video_ids=operation.video_ids,
            event_refs=operation.event_refs,
            statement=operation.statement.strip(),
        )
    )
    return _Pending(
        operation="ADD",
        target_kind="information",
        target_id=identifier,
        previous_value=None,
        evidence_refs=operation.event_refs,
    )


def _add_relation(
    operation: AddRelation,
    *,
    known_videos: set[str],
    known_events: set[str],
    relations: list[WorkingRelation],
) -> _Pending:
    identifier = operation.relation_id.strip()
    if not identifier:
        raise ValueError("added relation needs a nonempty id")
    if identifier in {item.relation_id for item in relations}:
        raise ValueError(f"relation id already exists: {identifier}")
    if len(operation.scope) < 2:
        raise ValueError(f"relation {identifier} must compare at least two videos")
    _require_subset(operation.scope, known_videos, f"relation {identifier}.scope")
    if not operation.supporting_events:
        raise ValueError(f"relation {identifier} must cite at least one event")
    _require_subset(
        operation.supporting_events, known_events, f"relation {identifier}.supporting_events"
    )
    if not operation.statement.strip():
        raise ValueError(f"relation {identifier} statement must be nonempty")
    relations.append(
        WorkingRelation(
            relation_id=identifier,
            scope=operation.scope,
            statement=operation.statement.strip(),
            supporting_events=operation.supporting_events,
            # Placeholder: every status is recomputed before the state is built.
            status="evidence_partial",
        )
    )
    return _Pending(
        operation="ADD",
        target_kind="relation",
        target_id=identifier,
        previous_value=None,
        evidence_refs=operation.supporting_events,
    )


def _add_issue(
    operation: AddIssue,
    *,
    known_videos: set[str],
    issues: list[UnresolvedIssue],
    resolved: list[UnresolvedIssue],
) -> _Pending:
    identifier = operation.issue_id.strip()
    if not identifier:
        raise ValueError("added issue needs a nonempty id")
    held = {item.issue_id for item in (*issues, *resolved)}
    if identifier in held:
        raise ValueError(f"issue id already exists: {identifier}")
    if not operation.scope:
        raise ValueError(f"issue {identifier} must cover at least one video")
    _require_subset(operation.scope, known_videos, f"issue {identifier}.scope")
    if not operation.issue.strip() or not operation.reason.strip():
        raise ValueError(f"issue {identifier} needs a nonempty issue and reason")
    issues.append(
        UnresolvedIssue(
            issue_id=identifier,
            scope=operation.scope,
            issue=operation.issue.strip(),
            reason=operation.reason.strip(),
            related_events=operation.related_events,
        )
    )
    return _Pending(
        operation="ADD",
        target_kind="issue",
        target_id=identifier,
        previous_value=None,
        evidence_refs=operation.related_events,
    )


def _update_relation(
    operation: UpdateRelation,
    *,
    known_events: set[str],
    relations: list[WorkingRelation],
) -> _Pending:
    identifier = operation.relation_id
    index = _index_of(relations, identifier, "relation")
    current = relations[index]
    if operation.status is not None:
        # Invariant 3: status is derived, so a delta that carries one is a bug
        # in the caller or a model trying to re-assert confidence.
        raise ValueError(
            f"relation {identifier} status is derived by the module and cannot be "
            f"set by a state delta (got {operation.status!r})"
        )
    if operation.scope is not None and set(operation.scope) != set(current.scope):
        # Invariant 2: scope is part of a relation's identity.  Changing it
        # would silently move which video pairs this relation covers.  Compared
        # as sets, because what matters is which videos are compared, not the
        # order they were listed in.
        raise ValueError(
            f"relation {identifier} scope is immutable "
            f"({list(current.scope)} -> {list(operation.scope)}); "
            "retract this relation and add a new one instead"
        )
    statement = current.statement
    if operation.statement is not None:
        if not operation.statement.strip():
            raise ValueError(f"relation {identifier} statement must be nonempty")
        statement = operation.statement.strip()
    evidence = current.supporting_events
    if operation.supporting_events is not None:
        if not operation.supporting_events:
            raise ValueError(f"relation {identifier} must cite at least one event")
        _require_subset(
            operation.supporting_events, known_events, f"relation {identifier}.supporting_events"
        )
        evidence = operation.supporting_events
    relations[index] = WorkingRelation(
        relation_id=current.relation_id,
        scope=current.scope,
        statement=statement,
        supporting_events=evidence,
        status=current.status,
    )
    return _Pending(
        operation="UPDATE",
        target_kind="relation",
        target_id=identifier,
        previous_value=_relation_snapshot(current),
        evidence_refs=evidence,
    )


def _retract_relation(
    operation: RetractRelation,
    *,
    relations: list[WorkingRelation],
) -> _Pending:
    index = _index_of(relations, operation.relation_id, "relation")
    if not operation.reason.strip():
        raise ValueError(f"retracting {operation.relation_id} needs a reason")
    current = relations.pop(index)
    return _Pending(
        operation="RETRACT",
        target_kind="relation",
        target_id=current.relation_id,
        previous_value=_relation_snapshot(current),
        evidence_refs=current.supporting_events,
        note=operation.reason.strip(),
    )


def _retract_issue(
    operation: RetractIssue,
    *,
    issues: list[UnresolvedIssue],
) -> _Pending:
    index = _index_of(issues, operation.issue_id, "issue")
    if not operation.reason.strip():
        raise ValueError(f"retracting {operation.issue_id} needs a reason")
    current = issues.pop(index)
    return _Pending(
        operation="RETRACT",
        target_kind="issue",
        target_id=current.issue_id,
        previous_value=_issue_snapshot(current, "active"),
        evidence_refs=current.related_events,
        note=operation.reason.strip(),
    )


def _reopen_issue(
    operation: ReopenIssue,
    *,
    issues: list[UnresolvedIssue],
    resolved: list[UnresolvedIssue],
) -> _Pending:
    index = _index_of(resolved, operation.issue_id, "resolved issue")
    if not operation.reason.strip():
        raise ValueError(f"reopening {operation.issue_id} needs a reason")
    current = resolved.pop(index)
    issues.append(current)
    return _Pending(
        operation="REOPEN",
        target_kind="issue",
        target_id=current.issue_id,
        previous_value=_issue_snapshot(current, "resolved"),
        evidence_refs=current.related_events,
        note=operation.reason.strip(),
    )


def _resolve_issue(
    operation: ResolveIssue,
    *,
    relations: list[WorkingRelation],
    issues: list[UnresolvedIssue],
    resolved: list[UnresolvedIssue],
) -> _Pending:
    index = _index_of(issues, operation.issue_id, "issue")
    if not operation.reason.strip():
        raise ValueError(f"resolving {operation.issue_id} needs a reason")
    current = issues[index]
    # Invariant 1: closing an issue removes one cover of its video pairs, so the
    # resolution must name the relation that takes over.  "This gap is settled"
    # is not an acceptable edit; "R05 settled it, and R05 really covers that
    # pair" is.
    relation_index = _index_of(relations, operation.resolving_relation, "relation")
    resolver = relations[relation_index]
    uncovered = [
        pair
        for pair in ordered_pairs(current.scope)
        if not _covers(resolver.scope, pair)
    ]
    if uncovered:
        raise ValueError(
            f"cannot resolve {current.issue_id}: relation {resolver.relation_id} "
            f"does not cover {uncovered}"
        )
    issues.pop(index)
    resolved.append(current)
    return _Pending(
        operation="RESOLVE",
        target_kind="issue",
        target_id=current.issue_id,
        previous_value=_issue_snapshot(current, "active"),
        evidence_refs=resolver.supporting_events,
        note=operation.reason.strip(),
    )


# ── helpers ──────────────────────────────────────────────────────────


def _reject_duplicate_targets(delta: StateDelta) -> None:
    """One edit per target per round, so history has no ambiguous ordering."""
    seen: set[tuple[str, str]] = set()
    for operation in delta.operations:
        key = _target_of(operation)
        if key in seen:
            raise ValueError(
                f"state delta edits {key[1]} more than once in round {delta.round_id}"
            )
        seen.add(key)


def _target_of(operation: object) -> tuple[str, str]:
    for kind, attribute in (
        ("information", "information_id"),
        ("relation", "relation_id"),
        ("issue", "issue_id"),
    ):
        value = getattr(operation, attribute, None)
        if value is not None:
            return kind, str(value)
    raise ValueError(f"unsupported state operation: {operation!r}")


def _uncovered_pairs(
    required_pairs: tuple[tuple[str, str], ...],
    relations: list[WorkingRelation],
    issues: list[UnresolvedIssue],
) -> list[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for scope in (
        *(item.scope for item in relations),
        *(item.scope for item in issues),
    ):
        covered.update(ordered_pairs(scope))
    required = {ordered_pair(*pair) for pair in required_pairs}
    return [pair for pair in required_pairs if ordered_pair(*pair) not in covered & required]


def ordered_pairs(scope: tuple[str, ...]) -> set[tuple[str, str]]:
    return {
        ordered_pair(left, right)
        for index, left in enumerate(scope)
        for right in scope[index + 1 :]
    }


def _covers(scope: tuple[str, ...], pair: tuple[str, str]) -> bool:
    return set(pair) <= set(scope)


def _require_subset(values: tuple[str, ...], known: set[str], label: str) -> None:
    unknown = [value for value in values if value not in known]
    if unknown:
        raise ValueError(f"{label} references unknown ids: {unknown}")


def _index_of(items: list[Any], identifier: str, kind: str) -> int:
    for index, item in enumerate(items):
        if _identifier_of(item) == identifier:
            return index
    raise ValueError(f"unknown {kind}: {identifier}")


def _identifier_of(item: Any) -> str:
    for attribute in ("information_id", "relation_id", "issue_id"):
        value = getattr(item, attribute, None)
        if value is not None:
            return str(value)
    raise ValueError(f"object has no state identifier: {item!r}")


def _relation_snapshot(relation: WorkingRelation) -> dict[str, Any]:
    return {
        "id": relation.relation_id,
        "scope": list(relation.scope),
        "statement": relation.statement,
        "supporting_events": list(relation.supporting_events),
        # Derived, never supplied: included here so a reader can see a status
        # change without having to diff two whole states.
        "status": relation.status,
        "status_source": "derived_by_module",
    }


def _issue_snapshot(issue: UnresolvedIssue, status: str) -> dict[str, Any]:
    return {
        "id": issue.issue_id,
        "scope": list(issue.scope),
        "issue": issue.issue,
        "reason": issue.reason,
        "related_events": list(issue.related_events),
        "filled_by_module": issue.filled_by_module,
        "status": status,
    }


def _information_snapshot(information: ObservedInformation) -> dict[str, Any]:
    return {
        "id": information.information_id,
        "videos": list(information.video_ids),
        "event_refs": list(information.event_refs),
        "statement": information.statement,
    }


def _snapshot(
    target_kind: str,
    target_id: str,
    *,
    information: list[ObservedInformation],
    relations: list[WorkingRelation],
    issues: list[UnresolvedIssue],
    resolved: list[UnresolvedIssue],
) -> dict[str, Any] | None:
    """The post-round value of a target, or ``None`` if it no longer exists."""
    if target_kind == "information":
        for item in information:
            if item.information_id == target_id:
                return _information_snapshot(item)
        return None
    if target_kind == "relation":
        for item in relations:
            if item.relation_id == target_id:
                return _relation_snapshot(item)
        return None
    for item in issues:
        if item.issue_id == target_id:
            return _issue_snapshot(item, "active")
    for item in resolved:
        if item.issue_id == target_id:
            return _issue_snapshot(item, "resolved")
    return None
