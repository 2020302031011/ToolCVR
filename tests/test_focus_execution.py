"""The executor: one ``FocusTool`` call per look, one video per call."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from toolcvr.agents import ObserverAgent
from toolcvr.focus import FocusConfig, FocusRequest, FocusTool
from toolcvr.focus_execution import FocusEvidenceBundle, execute_focus_plan
from toolcvr.focus_planner import FocusLook, FocusPlan
from toolcvr.reporting import DeterministicMockEventReporter
from toolcvr.schemas import (
    EventRecord,
    ObservationBatch,
    ObservationGap,
    QueryAssessment,
    ReportFrame,
    VideoInput,
)


SENTENCE = "Is a lid already on the pan before the stirring starts?"


# These three helpers duplicate tests/test_focus.py.  The runner supplied with
# this repository loads test modules by path and does not put ``tests/`` on
# ``sys.path``, so one test module cannot import another's helpers.
def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (96, 64),
    )
    if not writer.isOpened():
        raise RuntimeError("test environment cannot create MJPG video")
    try:
        for index in range(100):
            writer.write(np.full((64, 96, 3), index, dtype=np.uint8))
    finally:
        writer.release()


def _tool() -> FocusTool:
    return FocusTool(
        observer=ObserverAgent(DeterministicMockEventReporter()),
        config=FocusConfig(
            default_radius_seconds=2.0,
            frames_per_call=8,
            decode_long_edge=64,
        ),
    )


def _video_inputs(tmp_path: Path, video_ids: Sequence[str]) -> tuple[VideoInput, ...]:
    inputs = []
    for video_id in video_ids:
        path = tmp_path / f"{video_id}.avi"
        _write_video(path)
        inputs.append(VideoInput(video_id, path))
    return tuple(inputs)


def _plan(looks: Sequence[FocusLook], **kwargs: object) -> FocusPlan:
    return FocusPlan(
        issue_id="U01",
        issue_class="uncited",
        evidence_need=SENTENCE,
        looks=tuple(looks),
        **kwargs,  # type: ignore[arg-type]
    )


def _look(video_id: str, span: tuple[float, float]) -> FocusLook:
    return FocusLook(
        request=FocusRequest(video_id=video_id, evidence_need=SENTENCE, time_span=span),
        anchor_event_ids=(f"{video_id}-e1",),
    )


@dataclass(frozen=True)
class _UnprefixedIdReporter:
    """A reporter whose ids are only unique inside one batch.

    This is what the real VLM reporters do: ``report_generation.py`` prefixes
    the ids with the video before they reach a sample-wide namespace, and the
    focus path does not, so two looks can return the same id for two different
    events.
    """

    def report(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        del video_id, question, report_hints
        if not frames:
            raise ValueError("event reporter requires at least one frame")
        return ObservationBatch(
            events=(
                EventRecord(
                    event_id="e1",
                    start_seconds=frames[0].actual_timestamp_seconds,
                    end_seconds=frames[-1].actual_timestamp_seconds,
                    description="one visible action",
                    evidence_frame_ids=tuple(frame.frame_id for frame in frames),
                    confidence=0.5,
                ),
            ),
            query_assessment=QueryAssessment(
                relevance="insufficient",
                evidence_event_ids=("e1",),
                unresolved_needs=("nothing further",),
                suggested_tools=(),
            ),
        )


@dataclass(frozen=True)
class _GapReporter:
    """A visual failure remains a first-class, source-preserving result."""

    def report(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        del video_id, question, report_hints
        return ObservationBatch(
            events=(),
            observation_gaps=(
                ObservationGap(
                    gap_id="gap1",
                    start_seconds=frames[0].actual_timestamp_seconds,
                    end_seconds=frames[-1].actual_timestamp_seconds,
                    frame_ids=tuple(frame.frame_id for frame in frames),
                    reason="reporter could not validate its visual output",
                ),
            ),
            query_assessment=QueryAssessment(
                relevance="insufficient",
                evidence_event_ids=(),
                unresolved_needs=("this batch produced nothing usable",),
                suggested_tools=("dense_focus",),
            ),
        )


def test_a_two_video_plan_runs_one_focus_call_per_video(tmp_path: Path) -> None:
    """A round is a sequence of single-video calls, in plan order."""
    plan = _plan((_look("v1", (1.0, 2.0)), _look("v2", (3.0, 4.0))))
    bundle = execute_focus_plan(
        plan=plan,
        videos=_video_inputs(tmp_path, ("v1", "v2")),
        tool=_tool(),
        question="How does the object change?",
    )
    assert bundle.issue_id == "U01"
    assert bundle.evidence_need == SENTENCE
    assert bundle.videos == ("v1", "v2")
    for result, expected in zip(bundle.results, ("v1", "v2")):
        assert result.video_id == expected
        assert result.request_kind == "time_span"
        assert result.observation.events
        assert result.report_text.startswith(f"VIDEO_REPORT {expected}\n")
    assert len({event.event_id for event in bundle.events}) == len(bundle.events)


def test_no_look_ever_holds_another_videos_frames(tmp_path: Path) -> None:
    """The frame manifest is the provenance record, so it has to be per video."""
    plan = _plan((_look("v1", (1.0, 2.0)), _look("v2", (3.0, 4.0))))
    bundle = execute_focus_plan(
        plan=plan,
        videos=_video_inputs(tmp_path, ("v1", "v2")),
        tool=_tool(),
        question="How does the object change?",
    )
    for result in bundle.results:
        assert result.frame_manifest
        assert {frame.video_id for frame in result.frame_manifest} == {result.video_id}
        # One segment per call, and every frame of the call carries it.
        assert len({frame.segment_id for frame in result.frame_manifest}) == 1
        assert bundle.to_dict()["looks"][bundle.videos.index(result.video_id)][
            "segment_id"
        ] == result.frame_manifest[0].segment_id


def test_a_plan_naming_a_video_with_no_input_fails_loudly(tmp_path: Path) -> None:
    plan = _plan((_look("v1", (1.0, 2.0)), _look("v2", (3.0, 4.0))))
    try:
        execute_focus_plan(
            plan=plan,
            videos=_video_inputs(tmp_path, ("v1",)),
            tool=_tool(),
            question="How does the object change?",
        )
    except ValueError as exc:
        assert "v2" in str(exc)
        assert "no video input" in str(exc)
    else:
        raise AssertionError("a plan naming a missing video was executed anyway")


def test_duplicate_video_inputs_fail_loudly(tmp_path: Path) -> None:
    inputs = _video_inputs(tmp_path, ("v1",))
    try:
        execute_focus_plan(
            plan=_plan((_look("v1", (1.0, 2.0)),)),
            videos=(inputs[0], inputs[0]),
            tool=_tool(),
            question="How does the object change?",
        )
    except ValueError as exc:
        assert "duplicate video input" in str(exc)
    else:
        raise AssertionError("duplicate video inputs were accepted")


def test_a_bundle_with_no_looks_is_rejected() -> None:
    try:
        FocusEvidenceBundle(issue_id="U01", issue_class="uncited", evidence_need=SENTENCE, results=())
    except ValueError as exc:
        assert "at least one result" in str(exc)
    else:
        raise AssertionError("an empty bundle was accepted")


def test_the_bundle_preserves_the_unanchorable_record(tmp_path: Path) -> None:
    """Executing the round must not turn a partial round into a whole one."""
    plan = _plan(
        (_look("v1", (1.0, 2.0)),),
        bundle_videos=("v2", "v1"),
        unanchorable_videos=("v2",),
    )
    bundle = execute_focus_plan(
        plan=plan,
        videos=_video_inputs(tmp_path, ("v1",)),
        tool=_tool(),
        question="How does the object change?",
    )
    assert bundle.videos == ("v1",)
    assert bundle.unanchorable_videos == ("v2",)
    assert bundle.partial is True
    payload = bundle.to_dict()
    assert payload["partial"] is True
    assert payload["unanchorable_videos"] == ["v2"]


def test_repeated_event_ids_across_looks_are_recorded(tmp_path: Path) -> None:
    """Two looks can return the same event id; the merge must namespace them.

    The ids are only unique within a batch, so a round that does not say so
    would let the state's event-id-keyed merge drop one video's evidence
    silently.  Recorded here rather than renamed: ``report_text`` embeds the
    ids verbatim.
    """
    tool = FocusTool(
        observer=ObserverAgent(_UnprefixedIdReporter()),
        config=FocusConfig(frames_per_call=4, decode_long_edge=64),
    )
    bundle = execute_focus_plan(
        plan=_plan((_look("v1", (1.0, 2.0)), _look("v2", (3.0, 4.0)))),
        videos=_video_inputs(tmp_path, ("v1", "v2")),
        tool=tool,
        question="How does the object change?",
    )
    assert bundle.repeated_event_ids == ("e1",)
    assert bundle.to_dict()["repeated_event_ids"] == ["e1"]
    # The reports still carry the ids they were rendered with: renaming here
    # would desynchronize report_text from the events it describes.
    assert all("e1" in result.report_text for result in bundle.results)


def test_a_look_with_an_observation_gap_is_recorded_without_false_evidence(tmp_path: Path) -> None:
    """A failed visual call is visible to stage 5, not dropped or promoted."""
    tool = FocusTool(
        observer=ObserverAgent(_GapReporter()),
        config=FocusConfig(frames_per_call=4, decode_long_edge=64),
    )
    bundle = execute_focus_plan(
        plan=_plan((_look("v1", (1.0, 2.0)),)),
        videos=_video_inputs(tmp_path, ("v1",)),
        tool=tool,
        question="How does the object change?",
    )
    result = bundle.results[0]
    assert not result.observation.events
    assert len(result.observation.observation_gaps) == 1
    assert "OBSERVATION GAPS (not visual evidence)" in result.report_text
    serialized = bundle.to_dict()["looks"][0]["observation"]
    assert serialized["events"] == []
    assert serialized["observation_gaps"][0]["gap_id"] == "gap1"


def test_focus_execution_is_deterministic(tmp_path: Path) -> None:
    videos = _video_inputs(tmp_path, ("v1", "v2"))
    plan = _plan((_look("v1", (1.0, 2.0)), _look("v2", (3.0, 4.0))))
    first = execute_focus_plan(
        plan=plan, videos=videos, tool=_tool(), question="How does the object change?"
    )
    second = execute_focus_plan(
        plan=plan, videos=videos, tool=_tool(), question="How does the object change?"
    )
    assert first.to_dict() == second.to_dict()
