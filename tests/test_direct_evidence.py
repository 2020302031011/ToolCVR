from __future__ import annotations

import json

from toolcvr.direct_evidence import (
    DeterministicMockDirectEvidenceBackend,
    build_direct_evidence_context,
    parse_direct_evidence_answer,
)


def _report() -> dict[str, object]:
    return {
        "question": "Which video shows the red car?",
        "options": ["A. Video 1", "B. Video 2"],
        "videos": [
            {
                "video_id": "v1",
                "duration_seconds": 10.0,
                "selected_frames": 3,
                "clip_candidate_count": 5,
                "events": [{
                    "event_id": "v1-b0001-e1",
                    "time_span": [1.0, 2.0],
                    "description": "A red car enters the road.",
                    "uncertainty": "",
                }],
                "observation_gaps": [],
            },
            {
                "video_id": "v2",
                "duration_seconds": 12.0,
                "selected_frames": 2,
                "clip_candidate_count": 6,
                "events": [{
                    "event_id": "v2-b0001-e1",
                    "time_span": [3.0, 4.0],
                    "description": "An empty road is visible.",
                    "uncertainty": "Omitted regions remain unobserved.",
                }],
                "observation_gaps": [],
            },
        ],
        "run": {"ground_truth": "A"},
    }


def test_direct_context_preserves_video_time_and_event_identity_without_gt() -> None:
    text, refs, tokens = build_direct_evidence_context(_report())
    payload = json.loads(text)
    assert payload["videos"][0]["video_id"] == "v1"
    assert payload["videos"][0]["events"][0]["time"] == [1.0, 2.0]
    assert refs == {"v1-b0001-e1", "v2-b0001-e1"}
    assert tokens > 0
    assert "ground_truth" not in text


def test_direct_context_fails_instead_of_silently_dropping_events() -> None:
    try:
        build_direct_evidence_context(_report(), token_budget=1)
    except ValueError as exc:
        assert "complete warm evidence exceeds" in str(exc)
    else:
        raise AssertionError("complete warm evidence was silently truncated")


def test_direct_answer_accepts_only_current_evidence_and_option_label() -> None:
    valid = parse_direct_evidence_answer(
        '{"answer":"A","evidence_refs":["v1-b0001-e1"]}',
        known_refs={"v1-b0001-e1"},
        options=("A. Video 1", "B. Video 2"),
    )
    assert valid.answer == "A"
    canonical = parse_direct_evidence_answer(
        '{"answer":"A. Video 1","evidence_refs":"v1-b0001-e1"}',
        known_refs={"v1-b0001-e1"},
        options=("A. Video 1", "B. Video 2"),
    )
    assert canonical.answer == "A"
    assert canonical.evidence_refs == ("v1-b0001-e1",)


def test_direct_answer_audits_missing_and_unknown_citations_without_fabrication() -> None:
    missing = parse_direct_evidence_answer(
        '{"answer":"A","evidence_refs":[]}',
        known_refs={"v1-b0001-e1"},
        options=("A. Video 1", "B. Video 2"),
    )
    assert missing.status == "ANSWER"
    assert missing.evidence_refs == ()
    assert missing.rejected_evidence_refs == ()
    assert missing.citation_status == "missing"

    unknown = parse_direct_evidence_answer(
        '{"answer":"A","evidence_refs":["v1-b0001-e1","invented"]}',
        known_refs={"v1-b0001-e1"},
        options=("A. Video 1", "B. Video 2"),
    )
    assert unknown.evidence_refs == ("v1-b0001-e1",)
    assert unknown.rejected_evidence_refs == ("invented",)
    assert unknown.citation_status == "partial"


def test_direct_empty_answer_is_explicit_abstention() -> None:
    abstention = parse_direct_evidence_answer(
        '{"answer":"","evidence_refs":["v1-b0001-e1"]}',
        known_refs={"v1-b0001-e1"},
        options=("A. Video 1", "B. Video 2"),
    )
    assert abstention.status == "ABSTAIN"
    assert abstention.answer is None
    assert abstention.evidence_refs == ("v1-b0001-e1",)

    empty_interval = parse_direct_evidence_answer(
        '{"answer":[],"evidence_refs":[]}',
        known_refs={"v1-b0001-e1"},
        options=(),
    )
    assert empty_interval.status == "ABSTAIN"


def test_direct_non_choice_answer_preserves_native_json_shape() -> None:
    interval = parse_direct_evidence_answer(
        '{"answer":[139,146],"evidence_refs":["v1-b0001-e1"]}',
        known_refs={"v1-b0001-e1"},
        options=(),
    )
    assert interval.status == "ANSWER"
    assert interval.answer == [139, 146]


def test_direct_malformed_json_still_fails() -> None:
    try:
        parse_direct_evidence_answer(
            '{"answer":"A",',
            known_refs={"v1-b0001-e1"},
            options=("A. Video 1", "B. Video 2"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("malformed JSON was accepted")


def test_direct_mock_backend_is_deterministic_and_cpu_only() -> None:
    text, refs, _ = build_direct_evidence_context(_report())
    backend = DeterministicMockDirectEvidenceBackend()
    first = backend.answer(text, known_refs=refs, options=("A. Video 1", "B. Video 2"))
    second = backend.answer(text, known_refs=refs, options=("A. Video 1", "B. Video 2"))
    assert first == second
    assert first.status == "ANSWER"
    assert first.answer == "A"
