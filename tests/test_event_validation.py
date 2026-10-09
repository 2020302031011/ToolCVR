"""The mechanical checks on a VLM reply, and the one repair round they buy.

An event whose time span does not contain the frames it cites is internally
inconsistent, and the module refuses it.  Refusing is right; killing the whole
batch over it is not, and decoding is greedy, so simply asking again would
reproduce the same reply.  These tests pin both halves: the complaint carries
enough detail to act on, and the reporter sends it back exactly once per
attempt until the budget runs out.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import torch

from toolcvr.reporting import (
    LocalQwen3VLReporter,
    OpenAICompatibleMultimodalReporter,
    _report_prompt,
    _repair_prompt,
    parse_event_response,
)
from toolcvr.schemas import ReportFrame


def _frame(frame_id: str, timestamp: float) -> ReportFrame:
    return ReportFrame(
        video_id="v1",
        segment_id="S1",
        frame_id=frame_id,
        requested_timestamp_seconds=timestamp,
        actual_timestamp_seconds=timestamp,
        source_frame_index=round(timestamp * 30),
        matched_queries=("visible action",),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _event(
    event_id: str,
    start: float,
    end: float,
    frames: Sequence[str],
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "time_span": [start, end],
        "direct_observation": "something visible",
        "interpretation": "",
        "evidence_frame_ids": list(frames),
        "confidence": 0.5,
    }


def _reply(events: Sequence[dict[str, Any]]) -> str:
    event_ids = [str(event["event_id"]) for event in events]
    return json.dumps(
        {
            "events": list(events),
            "query_assessment": {
                "relevance": "indirect" if events else "insufficient",
                "evidence_event_ids": event_ids,
                "unresolved_needs": [] if events else ["No visible observation was emitted."],
                "suggested_tools": [],
            },
        }
    )


# ── the checks themselves ────────────────────────────────────────────


def test_a_span_containing_its_evidence_is_accepted() -> None:
    frames = (_frame("S1-F01", 10.0), _frame("S1-F02", 20.0))
    result = parse_event_response(
        _reply([_event("e1", 9.0, 21.0, ["S1-F01", "S1-F02"])]), frames
    )
    assert [event.event_id for event in result.events] == ["e1"]
    assert result.query_assessment.relevance == "indirect"


def test_visual_fact_and_inferred_identity_are_structurally_separated() -> None:
    frames = (_frame("S1-F01", 10.0),)
    event = _event("e1", 10.0, 10.0, ["S1-F01"])
    event["direct_observation"] = "A pale thick substance is spread from a jar."
    event["interpretation"] = "The substance may be mayonnaise."
    event["uncertainty"] = "The label is unreadable."
    result = parse_event_response(_reply([event]), frames)
    parsed = result.events[0]
    assert parsed.description == "A pale thick substance is spread from a jar."
    assert parsed.interpretation == "The substance may be mayonnaise."
    assert parsed.uncertainty == "The label is unreadable."


def test_vlm_event_boundary_rejects_legacy_undifferentiated_description() -> None:
    frames = (_frame("S1-F01", 10.0),)
    event = _event("e1", 10.0, 10.0, ["S1-F01"])
    event["description"] = event.pop("direct_observation")
    try:
        parse_event_response(_reply([event]), frames)
    except ValueError as exc:
        assert "direct_observation" in str(exc)
    else:
        raise AssertionError("Observer accepted an undifferentiated description field")


def test_inferred_interpretation_requires_explicit_uncertainty() -> None:
    frames = (_frame("S1-F01", 10.0),)
    event = _event("e1", 10.0, 10.0, ["S1-F01"])
    event["interpretation"] = "The person may be the owner."
    try:
        parse_event_response(_reply([event]), frames)
    except ValueError as exc:
        assert "requires an explicit uncertainty" in str(exc)
    else:
        raise AssertionError("unqualified inferred identity entered visual evidence")


def test_query_assessment_is_structurally_required() -> None:
    frames = (_frame("S1-F01", 10.0),)
    raw = json.dumps({"events": [_event("e1", 10.0, 10.0, ["S1-F01"])]})
    try:
        parse_event_response(raw, frames)
    except ValueError as exc:
        assert "query_assessment" in str(exc)
    else:
        raise AssertionError("report without query assessment was accepted")


def test_one_event_cannot_bridge_two_discontinuous_segments() -> None:
    frames = (
        _frame("S1-F01", 1.0),
        ReportFrame(
            video_id="v1",
            segment_id="S2",
            frame_id="S2-F01",
            requested_timestamp_seconds=1.001,
            actual_timestamp_seconds=1.001,
            source_frame_index=1200,
            matched_queries=("visible action",),
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            source_timestamp_seconds=40.0,
        ),
    )
    try:
        parse_event_response(
            _reply([_event("e1", 1.0, 1.001, ["S1-F01", "S2-F01"])]),
            frames,
        )
    except ValueError as exc:
        assert "crosses discontinuous frame segments" in str(exc)
    else:
        raise AssertionError("one event was allowed to bridge a source discontinuity")


def test_remote_observer_prompt_never_exposes_source_coordinates() -> None:
    frame = ReportFrame(
        video_id="v1",
        segment_id="S1",
        frame_id="S1-F01",
        requested_timestamp_seconds=0.5,
        actual_timestamp_seconds=0.5,
        source_frame_index=29630,
        matched_queries=("visible action",),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        source_timestamp_seconds=987.654,
    )
    reporter = OpenAICompatibleMultimodalReporter(
        base_url="https://example.invalid/v1",
        model="test-model",
        maximum_repairs=0,
        overlay_timestamps=False,
    )
    captured: list[object] = []

    class _Transport:
        def complete(self, *, messages, **kwargs):
            captured.extend(messages)
            return _reply([_event("e1", 0.5, 0.5, ["S1-F01"])])

    reporter._transport = _Transport()
    reporter.report(
        video_id="v1",
        frames=(frame,),
        question="What happens?",
        report_hints=(),
    )
    text_blocks = [
        str(block["text"])
        for message in captured
        for block in message["content"]
        if block["type"] == "text"
    ]
    prompt_text = "\n".join(text_blocks)
    assert "0.500s" in prompt_text
    assert "segment=S1" in prompt_text
    assert "987.654" not in prompt_text
    assert "29630" not in prompt_text
    assert "source_frame" not in prompt_text


def test_a_span_starting_after_its_earliest_frame_names_both() -> None:
    frames = (_frame("S1-F01", 10.0), _frame("S1-F02", 20.0))
    try:
        parse_event_response(_reply([_event("e1", 30.0, 40.0, ["S1-F01", "S1-F02"])]), frames)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected the span/evidence check to reject this event")
    # The complaint is also the repair instruction, so it has to name the
    # numbers the model would need to fix its reply.
    assert "30.000" in message and "40.000" in message
    assert "S1-F01" in message and "10.000" in message
    assert "must contain every frame it cites" in message


def test_a_span_ending_before_its_latest_frame_names_both() -> None:
    frames = (_frame("S1-F01", 10.0), _frame("S1-F02", 20.0))
    try:
        parse_event_response(_reply([_event("e1", 5.0, 15.0, ["S1-F01", "S1-F02"])]), frames)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected the span/evidence check to reject this event")
    assert "S1-F02" in message and "20.000" in message
    assert "must contain every frame it cites" in message


def test_the_repair_prompt_repeats_the_frame_manifest() -> None:
    frames = (_frame("S1-F01", 10.0), _frame("S1-F02", 20.0))
    prompt = _repair_prompt("event e1 is wrong", frames)
    assert "event e1 is wrong" in prompt
    assert "S1-F01 @ 10.000s [segment=S1]" in prompt
    assert "S1-F02 @ 20.000s [segment=S1]" in prompt
    assert "JSON object only" in prompt


def test_initial_prompt_uses_the_actual_request_frame_manifest() -> None:
    """The example and the validator must share one source of frame IDs."""
    frames = (
        _frame("FOCUS-0000862997-0000897995-F01", 86.300),
        _frame("FOCUS-0000862997-0000897995-F02", 87.200),
    )
    prompt = _report_prompt(
        video_id="v2",
        frames=frames,
        question="What changes?",
        report_hints=(),
        maximum_events=4,
    )
    assert '"FOCUS-0000862997-0000897995-F01"' in prompt
    assert "FOCUS-0000862997-0000897995-F02 @ 87.200s" in prompt
    assert '"S1-F01"' not in prompt


def test_empty_evidence_repair_requests_visible_state_not_a_span_fix() -> None:
    frames = (_frame("S1-F01", 10.0), _frame("S1-F02", 20.0))
    prompt = _repair_prompt("VLM event report must not silently return empty evidence", frames)
    assert "at least one visible observation" in prompt
    assert "record the remaining limitation in uncertainty" in prompt
    assert "Do not invent a question-relevant event" in prompt
    assert "time_span must contain every frame" not in prompt


def test_empty_reply_can_be_repaired_to_an_uncertain_visible_observation() -> None:
    frames = (_frame("S1-F01", 10.0),)
    empty = _reply([])
    visible = _reply([_event("e1", 10.0, 10.0, ["S1-F01"])])
    reporter = _stub_reporter([empty, visible])

    result = reporter.report(video_id="v1", frames=frames, question="q", report_hints=())

    assert [event.event_id for event in result.events] == ["e1"]
    assert reporter._processor.issued == 2
    assert "at least one visible observation" in reporter._processor.prompts[1]


# ── the repair round ─────────────────────────────────────────────────


class _ScriptedProcessor:
    """Stands in for the Qwen processor, replaying canned replies."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.issued = 0
        self.prompts: list[str] = []

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        self.prompts.append(
            "\n".join(
                block["text"]
                for block in messages[0]["content"]
                if block["type"] == "text"
            )
        )
        return "rendered-prompt"

    def __call__(self, *, text: str, images: Any, return_tensors: str) -> dict[str, Any]:
        del text, images, return_tensors
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

    def batch_decode(self, generated: Any, **_: Any) -> list[str]:
        del generated
        index = min(self.issued, len(self._replies) - 1)
        self.issued += 1
        return [self._replies[index]]


class _NullModel:
    def generate(self, **_: Any) -> Any:
        return torch.zeros((1, 8), dtype=torch.long)


def _stub_reporter(replies: Sequence[str], *, repairs: int = 2) -> LocalQwen3VLReporter:
    """A reporter whose model is a script.

    ``__init__`` is skipped on purpose: it loads a real checkpoint, and none of
    the logic under test here needs one.
    """
    reporter = object.__new__(LocalQwen3VLReporter)
    reporter._torch = torch
    reporter._device = torch.device("cpu")
    reporter._maximum_new_tokens = 64
    reporter._maximum_events = 4
    reporter._overlay_timestamps = False
    reporter._maximum_repairs = repairs
    reporter._processor = _ScriptedProcessor(replies)
    reporter._model = _NullModel()
    return reporter


def test_a_valid_reply_is_returned_without_a_second_call() -> None:
    frames = (_frame("S1-F01", 10.0),)
    reporter = _stub_reporter([_reply([_event("e1", 9.0, 11.0, ["S1-F01"])])])
    result = reporter.report(video_id="v1", frames=frames, question="q", report_hints=())
    assert [event.event_id for event in result.events] == ["e1"]
    assert reporter._processor.issued == 1


def test_a_rejected_reply_is_repaired_instead_of_killing_the_batch() -> None:
    frames = (_frame("S1-F01", 10.0), _frame("S1-F02", 20.0))
    bad = _reply([_event("e1", 30.0, 40.0, ["S1-F01", "S1-F02"])])
    good = _reply([_event("e1", 9.0, 21.0, ["S1-F01", "S1-F02"])])
    reporter = _stub_reporter([bad, good])

    result = reporter.report(video_id="v1", frames=frames, question="q", report_hints=())

    assert len(result.events) == 1
    assert result.events[0].start_seconds == 9.0, "the repaired reply is the one returned"
    processor = reporter._processor
    assert processor.issued == 2, "a rejection must trigger exactly one more call"
    assert "checker rejected" in processor.prompts[1]
    # The second prompt carries the numbers the model needs to fix the span.
    assert "S1-F01 @ 10.000s" in processor.prompts[1]


def test_the_repair_budget_is_finite_and_then_records_an_explicit_gap() -> None:
    frames = (_frame("S1-F01", 10.0),)
    bad = _reply([_event("e1", 30.0, 40.0, ["S1-F01"])])
    reporter = _stub_reporter([bad], repairs=2)

    result = reporter.report(video_id="v1", frames=frames, question="q", report_hints=())
    assert not result.events
    assert len(result.observation_gaps) == 1
    assert "must contain every frame it cites" in result.observation_gaps[0].reason

    # One original attempt plus the two the budget allows, and no more.
    assert reporter._processor.issued == 3


def test_a_zero_budget_records_a_gap_without_retrying() -> None:
    frames = (_frame("S1-F01", 10.0),)
    bad = _reply([_event("e1", 30.0, 40.0, ["S1-F01"])])
    reporter = _stub_reporter([bad], repairs=0)

    result = reporter.report(video_id="v1", frames=frames, question="q", report_hints=())
    assert not result.events
    assert len(result.observation_gaps) == 1

    assert reporter._processor.issued == 1


def test_single_string_unresolved_need_is_normalized_without_a_repair() -> None:
    frames = (_frame("S1-F01", 10.0),)
    payload = json.loads(_reply([_event("e1", 10.0, 10.0, ["S1-F01"])]))
    payload["query_assessment"]["unresolved_needs"] = "inspect nearby frames"
    reporter = _stub_reporter([json.dumps(payload)])

    result = reporter.report(video_id="v1", frames=frames, question="q", report_hints=())

    assert result.query_assessment.unresolved_needs == ("inspect nearby frames",)
    assert reporter._processor.issued == 1
