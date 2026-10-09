from __future__ import annotations

from typing import Any

from toolcvr.initial_state import (
    DeterministicMockInitialStateBuilder,
    _local_evidence_summary,
    normalize_initial_state,
    render_initial_state,
    required_pairs,
)
from toolcvr.schemas import (
    EventRecord,
    ObservationGap,
    QueryAssessment,
    VideoEvidenceReport,
)


def _event(video_id: str, index: int, description: str = "an observed action") -> EventRecord:
    return EventRecord(
        event_id=f"{video_id}-b1-e{index}",
        start_seconds=float(index),
        end_seconds=float(index) + 1.0,
        description=description,
        evidence_frame_ids=(f"{video_id}-F{index:02d}",),
        confidence=0.5,
    )


def _report(video_id: str, count: int = 2) -> VideoEvidenceReport:
    return VideoEvidenceReport(
        video_id=video_id,
        events=tuple(_event(video_id, index) for index in range(1, count + 1)),
    )


def _reports(*video_ids: str) -> tuple[VideoEvidenceReport, ...]:
    return tuple(_report(video_id) for video_id in video_ids)


def test_local_assessment_needs_are_not_promoted_into_reasoner_summary() -> None:
    report = VideoEvidenceReport(
        video_id="v1",
        events=(_event("v1", 1),),
        query_assessments=(
            QueryAssessment(
                relevance="insufficient",
                evidence_event_ids=("v1-b1-e1",),
                unresolved_needs=("Video 2 was not provided for comparison.",),
            ),
        ),
    )

    summary = _local_evidence_summary(report)

    assert set(summary) == {"direct_event_ids", "indirect_event_ids", "scope"}
    assert "unresolved_needs" not in summary
    assert "Video 2" not in str(summary)


def test_a_batch_count_of_local_misses_never_reaches_the_reasoner() -> None:
    """A count is the shape a reasoner lifts most easily into a global verdict.

    "One batch found no local match" reads as a fact about the video, when all
    it records is how one CLIP-filtered sample of it went.  The gap still
    travels -- in observation_gaps, anchored to its frames -- but the count
    does not.
    """
    report = VideoEvidenceReport(
        video_id="v1",
        events=(_event("v1", 1),),
        query_assessments=(
            QueryAssessment(
                relevance="insufficient",
                evidence_event_ids=("v1-b1-e1",),
                unresolved_needs=("No local match in this batch.",),
            ),
        ),
        observation_gaps=(
            ObservationGap(
                gap_id="v1-b1-gap1",
                start_seconds=1.0,
                end_seconds=2.0,
                frame_ids=("v1-F01",),
                reason="VLM output remained invalid: event e1 cites frames outside its span",
            ),
        ),
    )

    summary = _local_evidence_summary(report)

    assert not any("count" in key or "batches" in key for key in summary)
    assert summary["direct_event_ids"] == []
    assert summary["indirect_event_ids"] == []


def _expect_value_error(match: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        if match not in str(exc):
            raise AssertionError(f"unexpected error message: {exc}") from exc
    else:
        raise AssertionError(f"expected ValueError containing {match!r}")


def _normalize(payload: dict[str, Any], *video_ids: str):
    return normalize_initial_state(
        question="Which video uses a different tool?",
        reports=_reports(*video_ids),
        payload=payload,
    )


def _relation_payload(issue_scope: list[str] | None) -> dict[str, Any]:
    """A relation over v1..v3, each video citing one event, plus an optional issue."""

    payload: dict[str, Any] = {
        "working_relations": [
            {
                "id": "R01",
                "scope": ["v1", "v2", "v3"],
                "statement": "v1 shows the step while v2 and v3 do not.",
                "supporting_events": ["v1-b1-e1", "v2-b1-e1", "v3-b1-e1"],
            }
        ]
    }
    if issue_scope is not None:
        payload["unresolved_issues"] = [
            {
                "id": "U01",
                "scope": issue_scope,
                "issue": "Whether these videos show the step at all is undetermined.",
                "reason": "The supplied frames do not settle it.",
            }
        ]
    return payload


def test_a_fully_cited_uncontested_relation_is_evidence_complete() -> None:
    state = _normalize(_relation_payload(None), "v1", "v2", "v3")
    assert state.working_relations[0].status == "evidence_complete"


def test_a_relation_missing_a_citation_is_not_evidence_complete() -> None:
    payload = _relation_payload(None)
    payload["working_relations"][0]["supporting_events"] = ["v1-b1-e1", "v2-b1-e1"]
    state = _normalize(payload, "v1", "v2", "v3")
    assert state.working_relations[0].status == "evidence_partial"


def test_an_issue_over_the_exact_scope_leaves_the_relation_unresolved() -> None:
    state = _normalize(_relation_payload(["v1", "v2", "v3"]), "v1", "v2", "v3")
    assert state.working_relations[0].status == "unresolved"


def test_an_issue_over_part_of_the_scope_withholds_evidence_complete() -> None:
    """Overlap, not equality, is what contests a relation.

    The live failure this pins: a relation spanning v1..v4 was stamped
    evidence_complete by the module while an unresolved issue named v1, v2 and
    v4 as the very videos whose step could not be seen.  Every video in scope
    had cited an event, so citation coverage alone said "complete".
    """

    state = _normalize(_relation_payload(["v1", "v2"]), "v1", "v2", "v3")

    assert state.working_relations[0].status == "evidence_partial"
    assert any("overlap an unresolved issue" in note for note in state.validation_notes)


def test_required_pairs_enumerates_every_unordered_pair() -> None:
    assert required_pairs(["v1", "v2"]) == (("v1", "v2"),)
    assert required_pairs(["v1", "v2", "v3"]) == (
        ("v1", "v2"),
        ("v1", "v3"),
        ("v2", "v3"),
    )
    _expect_value_error("at least two videos", lambda: required_pairs(["v1"]))


def test_initial_state_exposes_no_answer_or_completion_field() -> None:
    state = DeterministicMockInitialStateBuilder().build(
        question="Which video differs?",
        reports=_reports("v1", "v2", "v3"),
    )
    payload = state.to_dict()
    for forbidden in ("best_answer", "final_answer", "alternatives", "decision", "confidence"):
        assert forbidden not in payload
    assert payload["contains_answer"] is False
    assert payload["contains_completion_decision"] is False
    assert payload["model_reported_confidence"] is False


def test_mock_builder_is_deterministic_and_covers_every_pair() -> None:
    builder = DeterministicMockInitialStateBuilder()
    reports = _reports("v1", "v2", "v3", "v4")
    first = builder.build(question="Q", reports=reports).to_dict()
    second = builder.build(question="Q", reports=reports).to_dict()
    assert first == second

    coverage = first["coverage"]
    assert coverage["required_pair_count"] == 6
    assert coverage["covered_pair_count"] == 6
    assert coverage["unresolved_pair_count"] == 6
    assert coverage["module_filled_pair_count"] == 0
    assert first["working_relations"] == []
    assert len(first["unresolved_issues"]) == 1
    assert first["unresolved_issues"][0]["scope"] == ["v1", "v2", "v3", "v4"]
    assert len(first["candidate_requests"]) == 1


def test_candidate_request_must_reference_an_existing_issue() -> None:
    state = _normalize(
        {
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "x", "reason": "y"}
            ],
            "candidate_requests": [
                {
                    "id": "Q01",
                    "target_videos": ["v1", "v2"],
                    "observable": "compare the pan contents",
                    "purpose": "resolve U01",
                    "related_issue": "U99",
                }
            ],
        },
        "v1",
        "v2",
    )
    assert state.candidate_requests == ()
    assert any("unknown related_issue" in note for note in state.validation_notes)


def test_unknown_event_reference_is_removed_and_noted() -> None:
    state = _normalize(
        {
            "observed_information": [
                {
                    "id": "I01",
                    "videos": ["v1"],
                    "event_refs": ["v1-b1-e1", "v9-b1-e1"],
                    "statement": "something visible",
                }
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "x", "reason": "y"}
            ],
        },
        "v1",
        "v2",
    )
    assert state.observed_information[0].event_refs == ("v1-b1-e1",)
    assert any("v9-b1-e1" in note for note in state.validation_notes)


def test_observed_information_cannot_cite_an_event_from_an_unlisted_video() -> None:
    state = _normalize(
        {
            "observed_information": [
                {
                    "id": "I01",
                    "videos": ["v1"],
                    "event_refs": ["v2-b1-e1"],
                    "statement": "incorrectly attributed observation",
                }
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "x", "reason": "y"}
            ],
        },
        "v1",
        "v2",
    )
    assert state.observed_information == ()
    assert any("outside videos" in note for note in state.validation_notes)


def test_relation_cannot_cite_an_event_outside_its_scope() -> None:
    state = _normalize(
        {
            "working_relations": [
                {
                    "id": "R01",
                    "scope": ["v1", "v2"],
                    "statement": "invalid cross-video relation",
                    "supporting_events": ["v1-b1-e1", "v3-b1-e1"],
                }
            ]
        },
        "v1",
        "v2",
        "v3",
    )
    assert state.working_relations == ()
    assert any("outside scope" in note for note in state.validation_notes)
    assert state.coverage_gaps == ()
    assert state.required_pairs == ()


def test_issue_removes_related_events_outside_its_scope() -> None:
    state = _normalize(
        {
            "unresolved_issues": [
                {
                    "id": "U01",
                    "scope": ["v1", "v2"],
                    "issue": "x",
                    "reason": "y",
                    "related_events": ["v1-b1-e1", "v3-b1-e1"],
                }
            ]
        },
        "v1",
        "v2",
        "v3",
    )
    issue = next(item for item in state.unresolved_issues if item.issue_id == "U01")
    assert issue.related_events == ("v1-b1-e1",)
    assert any("outside scope" in note for note in state.validation_notes)


def test_relation_status_is_derived_from_evidence_structure() -> None:
    payload = {
        "working_relations": [
            {
                "id": "R01",
                "scope": ["v1", "v2"],
                "statement": "both sides evidenced",
                "supporting_events": ["v1-b1-e1", "v2-b1-e1"],
                # The model claims the strongest possible status on both
                # relations; both claims must be discarded.
                "status": "evidence_complete",
            },
            {
                "id": "R02",
                "scope": ["v1", "v2"],
                "statement": "one side evidenced",
                "supporting_events": ["v1-b1-e1"],
                "status": "evidence_complete",
            },
        ],
        "unresolved_issues": [
            {"id": "U01", "scope": ["v1", "v2"], "issue": "x", "reason": "y"}
        ],
    }
    state = _normalize(payload, "v1", "v2")
    statuses = {item.relation_id: item.status for item in state.working_relations}
    # The issue over the same scope forbids treating either relation as covered,
    # whatever the model declared.
    assert statuses == {"R01": "unresolved", "R02": "unresolved"}


def test_relation_status_distinguishes_complete_from_partial_evidence() -> None:
    payload = {
        "working_relations": [
            {
                "id": "R01",
                "scope": ["v1", "v2"],
                "statement": "both sides evidenced",
                "supporting_events": ["v1-b1-e1", "v2-b1-e2"],
            },
            {
                "id": "R02",
                "scope": ["v1", "v2"],
                "statement": "one side evidenced",
                "supporting_events": ["v1-b1-e1"],
            },
        ]
    }
    state = _normalize(payload, "v1", "v2")
    statuses = {item.relation_id: item.status for item in state.working_relations}
    assert statuses == {"R01": "evidence_complete", "R02": "evidence_partial"}
    assert any("evidence_partial" in note for note in state.validation_notes)


def test_evidence_complete_does_not_mean_the_statement_is_true() -> None:
    """The derived status must stay a coverage fact, not a truth claim."""
    payload = {
        "working_relations": [
            {
                "id": "R01",
                "scope": ["v1", "v2"],
                "statement": "v1 happens before v2",
                "supporting_events": ["v1-b1-e1", "v2-b1-e1"],
            }
        ]
    }
    state = _normalize(payload, "v1", "v2")
    assert state.working_relations[0].status == "evidence_complete"
    rendered = render_initial_state(state)
    assert "does not assert that a statement is true" in rendered


def test_initial_state_does_not_manufacture_unmentioned_video_pairs() -> None:
    payload = {
        "working_relations": [
            {
                "id": "R01",
                "scope": ["v1", "v2"],
                "statement": "covered",
                "supporting_events": ["v1-b1-e1", "v2-b1-e1"],
            }
        ]
    }
    state = _normalize(payload, "v1", "v2", "v3")
    assert state.coverage_gaps == ()
    assert state.unresolved_issues == ()
    assert state.required_pairs == (("v1", "v2"),)
    assert state.filled_pairs() == {("v1", "v2")}


def test_relation_without_surviving_evidence_is_dropped_without_fabricating_an_issue() -> None:
    payload = {
        "working_relations": [
            {
                "id": "R01",
                "scope": ["v1", "v2"],
                "statement": "cites nothing real",
                "supporting_events": ["v1-b1-e99"],
            }
        ]
    }
    state = _normalize(payload, "v1", "v2")
    assert state.working_relations == ()
    assert state.coverage_gaps == ()
    assert state.required_pairs == ()
    assert state.unresolved_issues == ()


def test_single_video_evidence_gap_is_valid() -> None:
    state = _normalize(
        {
            "unresolved_issues": [
                {
                    "id": "U01",
                    "scope": ["v2"],
                    "issue": "The speaker identity in v2 is unknown.",
                    "reason": "Silent visual frames do not establish the role.",
                    "related_events": ["v2-b1-e1"],
                }
            ]
        },
        "v1",
        "v2",
    )
    assert state.unresolved_issues[0].scope == ("v2",)
    assert state.required_pairs == ()


def test_unknown_video_in_scope_is_dropped() -> None:
    payload = {
        "unresolved_issues": [
            {"id": "U01", "scope": ["v1", "v7"], "issue": "x", "reason": "y"},
            {"id": "U02", "scope": ["v1", "v2"], "issue": "x", "reason": "y"},
        ]
    }
    state = _normalize(payload, "v1", "v2")
    assert [item.issue_id for item in state.unresolved_issues] == ["U02"]
    assert any("v7" in note for note in state.validation_notes)


def test_duplicate_information_ids_are_dropped() -> None:
    payload = {
        "observed_information": [
            {"id": "I01", "videos": ["v1"], "event_refs": ["v1-b1-e1"], "statement": "a"},
            {"id": "I01", "videos": ["v2"], "event_refs": ["v2-b1-e1"], "statement": "b"},
        ],
        "unresolved_issues": [
            {"id": "U01", "scope": ["v1", "v2"], "issue": "x", "reason": "y"}
        ],
    }
    state = _normalize(payload, "v1", "v2")
    assert len(state.observed_information) == 1
    assert state.observed_information[0].statement == "a"


def test_duplicate_event_ids_across_reports_fail_loudly() -> None:
    shared = EventRecord(
        event_id="dup-1",
        start_seconds=0.0,
        end_seconds=1.0,
        description="shared",
        evidence_frame_ids=("F01",),
        confidence=0.5,
    )
    reports = (
        VideoEvidenceReport(video_id="v1", events=(shared,)),
        VideoEvidenceReport(video_id="v2", events=(shared,)),
    )
    _expect_value_error(
        "not unique across reports",
        lambda: normalize_initial_state(
            question="Q",
            reports=reports,
            payload={"unresolved_issues": []},
        ),
    )


def test_initial_state_requires_at_least_two_videos() -> None:
    _expect_value_error(
        "at least two video reports",
        lambda: _normalize({"unresolved_issues": []}, "v1"),
    )


def test_render_is_deterministic_and_declares_its_own_limits() -> None:
    state = DeterministicMockInitialStateBuilder().build(
        question="Which video differs?",
        reports=_reports("v1", "v2", "v3"),
    )
    first = render_initial_state(state)
    assert first == render_initial_state(state)
    assert first.startswith("INITIAL_STATE")
    assert "COVERAGE pairs=3/3" in first
    assert "UNRESOLVED ISSUES" in first
    assert "STATE_CAVEAT" in first
    assert "contains no answer and no completion decision" in first
