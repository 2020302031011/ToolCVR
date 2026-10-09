from __future__ import annotations

from typing import Any

from toolcvr.initial_state import normalize_initial_state, render_initial_state
from toolcvr.schemas import (
    AddRelation,
    EventRecord,
    ReopenIssue,
    ResolveIssue,
    RetractIssue,
    RetractRelation,
    StateDelta,
    UpdateRelation,
    VideoEvidenceReport,
    WorkingState,
)
from toolcvr.state_update import apply_state_delta


def _event(video_id: str, index: int) -> EventRecord:
    return EventRecord(
        event_id=f"{video_id}-b1-e{index}",
        start_seconds=float(index),
        end_seconds=float(index) + 1.0,
        description=f"{video_id} observed action {index}",
        evidence_frame_ids=(f"{video_id}-F{index:02d}",),
        confidence=0.5,
    )


def _report(video_id: str, count: int = 2) -> VideoEvidenceReport:
    return VideoEvidenceReport(
        video_id=video_id,
        events=tuple(_event(video_id, index) for index in range(1, count + 1)),
    )


def _working_state(payload: dict[str, Any], *video_ids: str) -> WorkingState:
    reports = tuple(_report(video_id) for video_id in video_ids)
    state = normalize_initial_state(question="Which video differs?", reports=reports, payload=payload)
    events = tuple(
        (event.event_id, report.video_id) for report in reports for event in report.events
    )
    return WorkingState(
        state=state,
        round_id=0,
        event_videos=events,
        module_filled_pairs=state.coverage_gaps,
    )


def _relation(identifier: str, events: list[str], statement: str = "v1 precedes v2") -> dict:
    return {
        "id": identifier,
        "scope": ["v1", "v2"],
        "statement": statement,
        "supporting_events": events,
    }


def _issue(identifier: str, issue: str = "the order of v1 and v2 is undetermined") -> dict:
    return {"id": identifier, "scope": ["v1", "v2"], "issue": issue, "reason": "reports disagree"}


def _two_video_state(relations: list[dict], issues: list[dict]) -> WorkingState:
    return _working_state(
        {"working_relations": relations, "unresolved_issues": issues}, "v1", "v2"
    )


def _expect_value_error(match: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        if match not in str(exc):
            raise AssertionError(f"unexpected error message: {exc}") from exc
    else:
        raise AssertionError(f"expected ValueError containing {match!r}")


# ── coverage repair ──────────────────────────────────────────────────


def test_retracting_the_sole_relation_repairs_coverage() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1", "v2-b1-e1"])], [])
    assert state.unresolved_issues == ()

    next_state = apply_state_delta(
        state, StateDelta(round_id=1, operations=(RetractRelation("R01", "frames show v2 first"),))
    )

    assert next_state.working_relations == ()
    assert [item.issue_id for item in next_state.unresolved_issues] == ["U-auto-01"]
    assert next_state.unresolved_issues[0].filled_by_module is True
    assert next_state.unresolved_issues[0].scope == ("v1", "v2")
    # The pair is still covered, so ``InitialState`` accepted the rebuild.
    assert next_state.state.filled_pairs() == {("v1", "v2")}
    assert next_state.module_filled_pairs == (("v1", "v2"),)
    assert next_state.pair_coverage_recall() == 0.0
    assert next_state.version_of("R01") == 1


def test_retracting_a_relation_whose_pair_is_covered_by_an_issue_repairs_nothing() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [_issue("U01")])
    next_state = apply_state_delta(
        state, StateDelta(round_id=1, operations=(RetractRelation("R01", "out of scope"),))
    )
    assert next_state.unresolved_issues[0].issue_id == "U01"
    assert next_state.module_filled_pairs == ()


def test_retracting_a_sole_issue_repairs_coverage() -> None:
    # No relation at all, so the issue is the only cover of (v1, v2).
    state = _two_video_state([], [_issue("U01")])
    next_state = apply_state_delta(
        state, StateDelta(round_id=1, operations=(RetractIssue("U01", "malformed"),))
    )
    assert next_state.module_filled_pairs == (("v1", "v2"),)
    assert [item.issue_id for item in next_state.unresolved_issues] == ["U-auto-01"]
    assert next_state.state.filled_pairs() == {("v1", "v2")}


# ── RESOLVE ──────────────────────────────────────────────────────────


def test_resolve_without_a_relation_is_rejected() -> None:
    state = _two_video_state([], [_issue("U01")])
    _expect_value_error(
        "unknown relation",
        lambda: apply_state_delta(
            state,
            StateDelta(round_id=1, operations=(ResolveIssue("U01", "R99", "settled"),)),
        ),
    )


def test_resolve_with_a_relation_that_does_not_cover_the_pair_is_rejected() -> None:
    state = _working_state(
        {
            "working_relations": [
                {"id": "R01", "scope": ["v1", "v3"], "statement": "x", "supporting_events": ["v1-b1-e1", "v3-b1-e1"]},
            ],
            "unresolved_issues": [_issue("U01")],
        },
        "v1",
        "v2",
        "v3",
    )
    _expect_value_error(
        "does not cover",
        lambda: apply_state_delta(
            state,
            StateDelta(round_id=1, operations=(ResolveIssue("U01", "R01", "settled"),)),
        ),
    )


def test_resolve_with_a_covering_relation_succeeds() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1", "v2-b1-e1"])], [_issue("U01")])
    assert state.unresolved_issues[0].issue_id == "U01"

    next_state = apply_state_delta(
        state,
        StateDelta(round_id=1, operations=(ResolveIssue("U01", "R01", "R01 settles it"),)),
    )

    assert next_state.unresolved_issues == ()
    assert [item.issue_id for item in next_state.resolved_issues] == ["U01"]
    assert next_state.state.filled_pairs() == {("v1", "v2")}
    assert next_state.module_filled_pairs == ()
    revision = next_state.history_for("U01")[0]
    assert revision.operation == "RESOLVE"
    assert revision.previous_value["status"] == "active"
    assert revision.new_value["status"] == "resolved"
    assert revision.evidence_refs == ("v1-b1-e1", "v2-b1-e1")
    assert revision.round_id == 1


# ── UPDATE ───────────────────────────────────────────────────────────


def test_update_may_not_change_scope() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [])
    _expect_value_error(
        "scope is immutable",
        lambda: apply_state_delta(
            state,
            StateDelta(
                round_id=1,
                operations=(
                    UpdateRelation("R01", scope=("v1", "v2", "v3")),
                ),
            ),
        ),
    )


def test_update_may_restate_the_scope_unchanged() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [])
    next_state = apply_state_delta(
        state,
        StateDelta(
            round_id=1,
            operations=(UpdateRelation("R01", statement="v2 precedes v1", scope=("v2", "v1")),),
        ),
    )
    assert next_state.working_relations[0].statement == "v2 precedes v1"
    assert next_state.working_relations[0].scope == ("v1", "v2")


def test_update_may_not_supply_a_status() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [])
    _expect_value_error(
        "derived by the module",
        lambda: apply_state_delta(
            state,
            StateDelta(
                round_id=1,
                operations=(UpdateRelation("R01", statement="s", status="evidence_complete"),),
            ),
        ),
    )


def test_update_of_evidence_recomputes_status() -> None:
    # Only v1 is cited, so the relation starts as ``evidence_partial``.
    state = _two_video_state([_relation("R01", ["v1-b1-e1"], statement="v1 precedes v2")], [])
    assert state.working_relations[0].status == "evidence_partial"

    next_state = apply_state_delta(
        state,
        StateDelta(
            round_id=1,
            operations=(
                UpdateRelation("R01", supporting_events=("v1-b1-e1", "v2-b1-e2")),
            ),
        ),
    )

    # The delta only ever named evidence; the status change is the module's.
    assert next_state.working_relations[0].status == "evidence_complete"
    assert next_state.working_relations[0].supporting_events == ("v1-b1-e1", "v2-b1-e2")

    revision = next_state.history_for("R01")[0]
    assert revision.operation == "UPDATE"
    assert revision.previous_value["status"] == "evidence_partial"
    assert revision.new_value["status"] == "evidence_complete"
    assert revision.evidence_refs == ("v1-b1-e1", "v2-b1-e2")


def test_status_becomes_evidence_complete_once_every_video_is_cited() -> None:
    state = _two_video_state(
        [_relation("R01", ["v1-b1-e1"])], [{"id": "U02", "scope": ["v1", "v2"], "issue": "x", "reason": "y"}]
    )
    # An issue over the same scope pins the relation to ``unresolved`` first.
    assert state.working_relations[0].status == "unresolved"

    next_state = apply_state_delta(
        state,
        StateDelta(
            round_id=1,
            operations=(
                UpdateRelation("R01", supporting_events=("v1-b1-e1", "v2-b1-e1")),
                ResolveIssue("U02", "R01", "R01 settles it"),
            ),
        ),
    )

    assert next_state.working_relations[0].status == "evidence_complete"
    assert next_state.unresolved_issues == ()


# ── REOPEN ───────────────────────────────────────────────────────────


def test_reopen_restores_an_active_issue() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1", "v2-b1-e1"])], [_issue("U01")])
    resolved = apply_state_delta(
        state, StateDelta(round_id=1, operations=(ResolveIssue("U01", "R01", "settled"),))
    )
    assert resolved.unresolved_issues == ()

    reopened = apply_state_delta(
        resolved,
        StateDelta(round_id=2, operations=(ReopenIssue("U01", "a later frame contradicts R01"),)),
    )

    assert [item.issue_id for item in reopened.unresolved_issues] == ["U01"]
    assert reopened.resolved_issues == ()
    # Re-opening reinstates the issue over the same scope, which pins R01 back
    # to ``unresolved`` -- the state is genuinely non-monotonic.
    assert reopened.working_relations[0].status == "unresolved"
    revision = reopened.history_for("U01")[1]
    assert revision.operation == "REOPEN"
    assert revision.previous_value["status"] == "resolved"
    assert revision.new_value["status"] == "active"


def test_reopen_rejects_an_unknown_or_active_issue() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [_issue("U01")])
    _expect_value_error(
        "unknown resolved issue",
        lambda: apply_state_delta(
            state, StateDelta(round_id=1, operations=(ReopenIssue("U01", "why"),))
        ),
    )


# ── history and replay ───────────────────────────────────────────────


def test_history_is_append_only_across_rounds() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [_issue("U01")])
    assert state.revision_history == ()

    first = apply_state_delta(
        state,
        StateDelta(round_id=1, operations=(UpdateRelation("R01", statement="restated"),)),
    )
    second = apply_state_delta(
        first,
        StateDelta(round_id=2, operations=(RetractRelation("R01", "contradicted"),)),
    )
    third = apply_state_delta(
        second,
        StateDelta(round_id=3, operations=(AddRelation("R02", ("v1", "v2"), "restated", ("v1-b1-e2",)),)),
    )

    history = third.revision_history
    assert len(history) == 3
    # Every earlier entry survived the later rounds unchanged.
    assert history[: len(first.revision_history)] == first.revision_history
    assert history[: len(second.revision_history)] == second.revision_history
    assert [item.round_id for item in history] == [1, 2, 3]
    assert third.version_of("R01") == 2
    assert third.version_of("R02") == 1


def test_revision_round_cannot_skip() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [])
    _expect_value_error(
        "does not follow",
        lambda: apply_state_delta(
            state, StateDelta(round_id=4, operations=(RetractRelation("R01", "x"),))
        ),
    )


def test_a_round_may_not_edit_the_same_target_twice() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [])
    _expect_value_error(
        "more than once",
        lambda: apply_state_delta(
            state,
            StateDelta(
                round_id=1,
                operations=(
                    UpdateRelation("R01", statement="a"),
                    RetractRelation("R01", "b"),
                ),
            ),
        ),
    )


def test_replay_reproduces_the_same_working_state() -> None:
    """Two runs of the same delta sequence agree, module-generated ids included."""

    def base() -> WorkingState:
        # One relation spanning all three videos: every pair starts covered.
        return _working_state(
            {
                "working_relations": [
                    {
                        "id": "R01",
                        "scope": ["v1", "v2", "v3"],
                        "statement": "all three differ in tool use",
                        "supporting_events": ["v1-b1-e1", "v2-b1-e1", "v3-b1-e1"],
                    }
                ]
            },
            "v1",
            "v2",
            "v3",
        )

    deltas = (
        # Removing the only relation uncovers all three pairs at once.
        StateDelta(round_id=1, operations=(RetractRelation("R01", "misread"),)),
        StateDelta(
            round_id=2,
            operations=(AddRelation("R02", ("v1", "v2"), "narrower claim", ("v1-b1-e1", "v2-b1-e1")),),
        ),
        # Dropping one auto issue uncovers its pair again, and the replacement id
        # has to be derived from the ids already taken.
        StateDelta(round_id=3, operations=(RetractIssue("U-auto-02", "withdrawn"),)),
    )

    def run() -> WorkingState:
        state = base()
        for delta in deltas:
            state = apply_state_delta(state, delta)
        return state

    first, second = run(), run()
    assert first.to_dict() == second.to_dict()
    assert [item.issue_id for item in first.unresolved_issues] == [
        "U-auto-01",
        "U-auto-03",
        "U-auto-04",
    ]
    assert first.module_filled_pairs == (("v1", "v2"), ("v1", "v3"), ("v2", "v3"))
    assert first.pair_coverage_recall() == 0.0
    assert first.round_id == 3
    # The audit trail holds the model's edits and the module's repairs, in order.
    assert [(item.round_id, item.operation, item.target_id) for item in first.revision_history] == [
        (1, "RETRACT", "R01"),
        (1, "ADD", "U-auto-01"),
        (1, "ADD", "U-auto-02"),
        (1, "ADD", "U-auto-03"),
        (2, "ADD", "R02"),
        (3, "RETRACT", "U-auto-02"),
        (3, "ADD", "U-auto-04"),
    ]


def test_working_state_still_carries_no_answer_fields() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1"])], [_issue("U01")])
    next_state = apply_state_delta(
        state,
        StateDelta(round_id=1, operations=(UpdateRelation("R01", statement="restated"),)),
    )
    payload = next_state.to_dict()
    for forbidden in ("best_answer", "final_answer", "decision", "confidence"):
        assert forbidden not in payload
    assert payload["contains_answer"] is False
    assert payload["contains_completion_decision"] is False
    assert payload["model_reported_confidence"] is False


def test_render_after_update_declares_coverage_and_no_answer() -> None:
    state = _two_video_state([_relation("R01", ["v1-b1-e1", "v2-b1-e1"])], [_issue("U01")])
    next_state = apply_state_delta(
        state, StateDelta(round_id=1, operations=(ResolveIssue("U01", "R01", "settled"),))
    )
    rendered = render_initial_state(next_state.state)
    assert "COVERAGE pairs=1/1" in rendered
    assert "STATE_CAVEAT" in rendered
    assert "contains no answer and no completion decision" in rendered
