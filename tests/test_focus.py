from pathlib import Path

import cv2
import numpy as np

from toolcvr.agents import ObserverAgent
from toolcvr.focus import FocusConfig, FocusRequest, FocusTool, split_complete_zoom
from toolcvr.reporting import DeterministicMockEventReporter
from toolcvr.schemas import VideoInput


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


def test_focus_timestamp_expands_to_a_dense_local_clip(tmp_path: Path) -> None:
    path = tmp_path / "video.avi"
    _write_video(path)
    result = _tool().run(
        video=VideoInput("v1", path),
        request=FocusRequest(
            video_id="v1",
            center_timestamp_seconds=5.0,
            evidence_need="verify the visible action around this moment",
        ),
        question="What action occurs?",
    )
    assert result.request_kind == "timestamp"
    assert result.resolved_time_span == (3.0, 7.0)
    assert len(result.frame_manifest) == 8
    assert len({frame.segment_id for frame in result.frame_manifest}) == 1
    assert result.frame_manifest[0].actual_timestamp_seconds == 3.0
    assert result.frame_manifest[-1].actual_timestamp_seconds == 7.0
    assert result.report_text.startswith("VIDEO_REPORT v1")


def test_focus_accepts_an_explicit_time_span(tmp_path: Path) -> None:
    path = tmp_path / "video.avi"
    _write_video(path)
    result = _tool().run(
        video=VideoInput("v1", path),
        request=FocusRequest(
            video_id="v1",
            time_span=(1.0, 2.0),
            evidence_need="inspect before and after object state",
        ),
        question="How does the object change?",
    )
    assert result.request_kind == "time_span"
    assert result.resolved_time_span == (1.0, 2.0)
    assert len(result.frame_manifest) == 8


def test_focus_clamps_a_duration_boundary_to_the_last_decodable_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video.avi"
    _write_video(path)
    result = _tool().run(
        video=VideoInput("v1", path),
        request=FocusRequest(
            video_id="v1",
            center_timestamp_seconds=10.0,
            evidence_need="inspect the final visible state",
        ),
        question="What is visible at the end?",
    )
    assert result.request_kind == "timestamp"
    assert np.allclose(result.resolved_time_span, (7.9, 9.9))
    assert max(
        frame.actual_timestamp_seconds for frame in result.frame_manifest
    ) <= 9.9 + 1e-9


def test_focus_rejects_a_timestamp_more_than_one_frame_past_the_end(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video.avi"
    _write_video(path)
    try:
        _tool().run(
            video=VideoInput("v1", path),
            request=FocusRequest(
                video_id="v1",
                center_timestamp_seconds=10.01,
                evidence_need="inspect beyond the recording",
            ),
            question="What happens later?",
        )
    except ValueError as exc:
        assert "outside video ending" in str(exc)
    else:
        raise AssertionError("focus accepted a locator beyond one-frame tolerance")


def test_focus_uses_local_clock_but_retains_hidden_source_provenance(tmp_path: Path) -> None:
    path = tmp_path / "video.avi"
    _write_video(path)
    result = _tool().run(
        video=VideoInput(
            "v1", path, ((6.0, 8.0),), model_clock="local",
        ),
        request=FocusRequest(
            video_id="v1", time_span=(0.25, 1.25),
            evidence_need="inspect the local transition",
        ),
        question="What changes?",
    )
    assert result.resolved_time_span == (0.25, 1.25)
    assert min(item.actual_timestamp_seconds for item in result.frame_manifest) >= 0.25
    assert max(item.actual_timestamp_seconds for item in result.frame_manifest) <= 1.25
    assert min(float(item.source_timestamp_seconds) for item in result.frame_manifest) >= 6.25
    assert max(float(item.source_timestamp_seconds) for item in result.frame_manifest) <= 7.25
    assert "6.250" not in result.report_text


def test_focus_request_requires_exactly_one_locator() -> None:
    try:
        FocusRequest(video_id="v1", evidence_need="inspect action")
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("focus request without a locator was accepted")


def test_complete_zoom_splits_a_reasoner_selected_region_into_one_fps_calls() -> None:
    requests = split_complete_zoom(
        video_id="v1", time_span=(10.0, 40.0), evidence_need="inspect sequence",
    )
    assert [item.time_span for item in requests] == [(10.0, 25.0), (25.0, 40.0)]
    assert all(item.sampling_fps == 1.0 for item in requests)
