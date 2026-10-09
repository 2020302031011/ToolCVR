import inspect

import numpy as np

import toolcvr.agents as agents
from toolcvr.agents import ObserverAgent, ReasonerAgent
from toolcvr.schemas import (
    EventRecord,
    ObservationBatch,
    QueryAssessment,
    ReportFrame,
    SelectedFrameTrace,
    VideoSketch,
)


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


def _sketch(video_id: str, report_text: str) -> VideoSketch:
    event = EventRecord(
        event_id=f"{video_id}-e1",
        start_seconds=2.0,
        end_seconds=2.0,
        description="An object changes state.",
        evidence_frame_ids=("S1-F01",),
        confidence=0.7,
    )
    trace = SelectedFrameTrace(
        batch_index=1,
        video_id=video_id,
        segment_id="S1",
        frame_id="S1-F01",
        timeline_index=1,
        requested_timestamp_seconds=2.0,
        actual_timestamp_seconds=2.0,
        requested_frame_index=60,
        actual_frame_index=60,
        matched_queries=("visible action",),
    )
    return VideoSketch(
        video_id=video_id,
        duration_seconds=10.0,
        clip_scan_stride_seconds=2.0,
        clip_candidate_count=5,
        cache_hit=True,
        observed_spans=((2.0, 2.0),),
        events=(event,),
        report_batches=1,
        selected_frames=1,
        selection_mode="query_ranked_ratio",
        frame_manifest=(trace,),
        report_text=report_text,
    )


def test_exactly_two_concrete_logical_agent_classes_exist() -> None:
    concrete = {
        name
        for name, value in vars(agents).items()
        if inspect.isclass(value)
        and value.__module__ == agents.__name__
        and name.endswith("Agent")
    }
    assert concrete == {"ObserverAgent", "ReasonerAgent"}


def test_observer_validates_evidence_and_renders_stable_report_text() -> None:
    class Reporter:
        def report(self, **_kwargs: object) -> ObservationBatch:
            event = EventRecord(
                    event_id="e1",
                    start_seconds=2.0,
                    end_seconds=4.0,
                    description="A hand places an object into a bowl.",
                    evidence_frame_ids=("S1-F01", "S1-F02"),
                    confidence=0.8,
                    uncertainty="The object identity is unclear.",
                    interpretation="The object may be a recipe ingredient.",
                )
            return ObservationBatch(
                events=(event,),
                query_assessment=QueryAssessment(
                    relevance="indirect", evidence_event_ids=("e1",)
                ),
            )

    observer = ObserverAgent(Reporter())
    observation = observer.observe(
        video_id="v1",
        frames=(_frame("S1-F01", 2.0), _frame("S1-F02", 4.0)),
        question="What happens?",
        report_hints=(),
    )
    text = observer.render_report_text(
        video_id="v1",
        events=observation.events,
        query_assessments=(observation.query_assessment,),
        observation_gaps=observation.observation_gaps,
    )
    assert text.startswith("VIDEO_REPORT v1\nEVENT INDEX")
    assert "[2.000-4.000s]" in text
    assert "evidence=S1-F01,S1-F02" in text
    assert "The object identity is unclear." in text
    assert "INFERRED INTERPRETATIONS (not direct visual evidence)" in text
    assert "The object may be a recipe ingredient." in text


def test_observer_rejects_unknown_evidence_frame() -> None:
    class Reporter:
        def report(self, **_kwargs: object) -> ObservationBatch:
            event = EventRecord(
                    event_id="e1",
                    start_seconds=2.0,
                    end_seconds=2.0,
                    description="Unsupported observation.",
                    evidence_frame_ids=("missing",),
                    confidence=0.2,
                )
            return ObservationBatch(
                events=(event,),
                query_assessment=QueryAssessment(
                    relevance="indirect", evidence_event_ids=("e1",)
                ),
            )

    try:
        ObserverAgent(Reporter()).observe(
            video_id="v1",
            frames=(_frame("S1-F01", 2.0),),
            question="What happens?",
            report_hints=(),
        )
    except ValueError as exc:
        assert "unknown frames" in str(exc)
    else:
        raise AssertionError("observer accepted an unknown evidence frame")


def test_observer_accepts_sparse_segments_in_temporal_order() -> None:
    class Reporter:
        def report(self, **_kwargs: object) -> ObservationBatch:
            event = EventRecord(
                    event_id="e1",
                    start_seconds=2.0,
                    end_seconds=8.0,
                    description="Two sparse visible states are reported in order.",
                    evidence_frame_ids=("S1-F01", "S2-F01"),
                    confidence=0.5,
                )
            return ObservationBatch(
                events=(event,),
                query_assessment=QueryAssessment(
                    relevance="indirect", evidence_event_ids=("e1",)
                ),
            )

    first = _frame("S1-F01", 2.0)
    second = ReportFrame(
        video_id="v1",
        segment_id="S2",
        frame_id="S2-F01",
        requested_timestamp_seconds=8.0,
        actual_timestamp_seconds=8.0,
        source_frame_index=240,
        matched_queries=("visible action",),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    observation = ObserverAgent(Reporter()).observe(
        video_id="v1",
        frames=(first, second),
        question="What happens?",
        report_hints=(),
    )
    assert observation.events[0].evidence_frame_ids == ("S1-F01", "S2-F01")


def test_observer_rejects_reversed_frame_time() -> None:
    class Reporter:
        def report(self, **_kwargs: object) -> ObservationBatch:
            raise AssertionError("reversed frames reached the visual backend")

    try:
        ObserverAgent(Reporter()).observe(
            video_id="v1",
            frames=(_frame("S1-F02", 4.0), _frame("S1-F01", 2.0)),
            question="What happens?",
            report_hints=(),
        )
    except ValueError as exc:
        assert "strict temporal order" in str(exc)
    else:
        raise AssertionError("observer accepted reversed frame timestamps")


def test_reasoner_consumes_only_nonempty_video_report_text() -> None:
    reasoner = ReasonerAgent()
    context = reasoner.build_evidence_context(
        (
            _sketch("v2", "VIDEO_REPORT v2\nEVENT INDEX\n- second"),
            _sketch("v1", "VIDEO_REPORT v1\nEVENT INDEX\n- first"),
        )
    )
    assert context.index("VIDEO_REPORT v1") < context.index("VIDEO_REPORT v2")
