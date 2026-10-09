from __future__ import annotations

from toolcvr.agents import ObserverAgent, ReasonerAgent
from toolcvr.reporting import DeterministicMockEventReporter
from toolcvr.tool_runtime import ScriptedToolChoice, ToolCall


def test_reasoner_selects_multiple_text_tools_before_requesting_observation() -> None:
    calls: list[str] = []

    def search(arguments):
        calls.append("search_evidence")
        return {"matches": ["v1-e1", "v2-e3"], "query": arguments["query"]}

    def compare(arguments):
        calls.append("compare_reports")
        return {"difference": "dialogue role remains unresolved"}

    result = ReasonerAgent().run_tools(
        reasoning_context="Determine whether an authority figure blocks the protagonist.",
        chooser=ScriptedToolChoice(
            (
                ToolCall("search_evidence", {"query": "authority obstacle"}),
                ToolCall(
                    "compare_reports",
                    {"video_ids": ["v1", "v2"], "question": "who blocks whom"},
                ),
                ToolCall(
                    "request_observation",
                    {
                        "question": "What command or refusal creates the obstacle?",
                        "target_video_ids": ["v1"],
                        "missing_evidence": ["dialogue content", "speaker role"],
                        "anchor_event_ids": ["v1-e1"],
                    },
                ),
            )
        ),
        implementations={"search_evidence": search, "compare_reports": compare},
    )
    assert calls == ["search_evidence", "compare_reports"]
    assert result.terminal_call.name == "request_observation"
    assert len(result.trace) == 2


def test_observer_selects_visual_and_audio_tools_before_returning_evidence() -> None:
    calls: list[str] = []

    def expand(arguments):
        calls.append("temporal_expand")
        return {"time_span": [20.0, 44.0], "video_id": arguments["video_id"]}

    def transcribe(arguments):
        calls.append("asr_search")
        return {"matches": ["You are forbidden to leave."]}

    observer = ObserverAgent(DeterministicMockEventReporter())
    result = observer.run_tools(
        observation_request="Resolve the authority relation around v1-e1.",
        chooser=ScriptedToolChoice(
            (
                ToolCall("temporal_expand", {"video_id": "v1", "anchor_seconds": 30.0}),
                ToolCall(
                    "asr_search",
                    {
                        "video_id": "v1",
                        "time_span": [20.0, 44.0],
                        "query": "command or obstruction",
                    },
                ),
                ToolCall(
                    "return_evidence",
                    {"evidence_event_ids": ["v1-e9"], "unresolved_needs": []},
                ),
            )
        ),
        implementations={"temporal_expand": expand, "asr_search": transcribe},
    )
    assert calls == ["temporal_expand", "asr_search"]
    assert result.terminal_call.name == "return_evidence"


def test_agents_cannot_call_each_others_tools() -> None:
    try:
        ReasonerAgent().run_tools(
            reasoning_context="text-only context",
            chooser=ScriptedToolChoice(
                (ToolCall("dense_focus", {"video_id": "v1", "time_span": [0, 2]}),)
            ),
            implementations={},
        )
    except ValueError as exc:
        assert "unavailable tool" in str(exc)
    else:
        raise AssertionError("ReasonerAgent was allowed to call an Observer tool")


def test_tool_loop_enforces_a_finite_step_budget() -> None:
    try:
        ReasonerAgent().run_tools(
            reasoning_context="keep searching",
            chooser=ScriptedToolChoice(
                (
                    ToolCall("search_evidence", {"query": "one"}),
                    ToolCall("search_evidence", {"query": "two"}),
                )
            ),
            implementations={"search_evidence": lambda arguments: dict(arguments)},
            maximum_steps=2,
        )
    except RuntimeError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("tool loop exceeded its maximum step budget")
