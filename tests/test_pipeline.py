from pathlib import Path

import cv2
import numpy as np

from toolcvr.clip import DeterministicMockClipEncoder
from toolcvr.pipeline import ReportConfig, WarmStartConfig, WarmStartPipeline
from toolcvr.query import ConservativeQueryRewriter
from toolcvr.reporting import DeterministicMockEventReporter, SelectionAuditReporter
from toolcvr.schemas import ReportFrame, VideoInput
from toolcvr.selection import SelectionConfig
from toolcvr.video import TimelinePolicy


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
        for index in range(60):
            image = np.zeros((64, 96, 3), dtype=np.uint8)
            image[:, :, index // 20] = 180
            cv2.putText(image, str(index), (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(image)
    finally:
        writer.release()


def test_mock_pipeline_runs_end_to_end_and_reuses_cache(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_video(video_path)
    config = WarmStartConfig(
        timeline=TimelinePolicy(
            short_stride_seconds=1.0,
            long_stride_seconds=2.0,
            long_video_threshold_seconds=60.0,
            decode_long_edge=64,
            clip_batch_size=4,
        ),
        selection=SelectionConfig(
            frames_per_vlm_call=8,
        ),
        report=ReportConfig(decode_long_edge=64),
        cache_directory=tmp_path / "cache",
    )
    pipeline = WarmStartPipeline(
        config=config,
        rewriter=ConservativeQueryRewriter(),
        clip_encoder=DeterministicMockClipEncoder(),
        reporter=DeterministicMockEventReporter(),
    )
    kwargs = {
        "videos": [VideoInput("v1", video_path)],
        "question": "Which visible action changes the object state?",
        "options": [],
    }
    first = pipeline.run(**kwargs)
    second = pipeline.run(**kwargs)
    assert first.videos[0].cache_hit is False
    assert second.videos[0].cache_hit is True
    assert first.videos[0].events
    assert first.videos[0].report_text.startswith("VIDEO_REPORT v1\nCOVERAGE selected=")
    assert "COVERAGE_CAVEAT" in first.videos[0].report_text
    assert "evidence=" in first.videos[0].report_text
    assert first.videos[0].selected_frames > 0
    assert first.videos[0].report_batches >= 1
    result = first.to_dict()
    assert result["videos"][0]["report_text"] == first.videos[0].report_text
    assert result["videos"][0]["filtered_regions_are_negative_evidence"] is False
    assert result["videos"][0]["observed_spans"]
    manifest = result["videos"][0]["frame_manifest"]
    assert len(manifest) == first.videos[0].selected_frames
    assert all(item["requested_frame_index"] == item["actual_frame_index"] for item in manifest)
    assert all(
        right["actual_timestamp_seconds"] > left["actual_timestamp_seconds"]
        for left, right in zip(manifest[:-1], manifest[1:])
    )


def test_selection_audit_exports_timestamped_contact_sheet(tmp_path: Path) -> None:
    frames = [
        ReportFrame(
            video_id="v1",
            segment_id="S1",
            frame_id=f"S1-F{index:02d}",
            requested_timestamp_seconds=float(index * 2),
            actual_timestamp_seconds=float(index * 2),
            source_frame_index=index * 60,
            matched_queries=("a visible action",),
            rgb=np.full((64, 96, 3), index * 50, dtype=np.uint8),
        )
        for index in range(1, 3)
    ]
    reporter = SelectionAuditReporter(tmp_path / "audit")
    observation = reporter.report(
        video_id="v1",
        frames=frames,
        question="What happens?",
        report_hints=(),
    )
    assert observation.events
    assert (tmp_path / "audit" / "v1" / "batch-001.jpg").is_file()
    assert "no VLM interpretation" in observation.events[0].description


def test_warm_pipeline_exposes_only_local_clip_time(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_video(video_path)
    config = WarmStartConfig(
        timeline=TimelinePolicy(
            short_stride_seconds=0.5,
            long_stride_seconds=1.0,
            long_video_threshold_seconds=60.0,
            decode_long_edge=64,
            clip_batch_size=4,
        ),
        selection=SelectionConfig(frames_per_vlm_call=8),
        report=ReportConfig(decode_long_edge=64),
        cache_directory=tmp_path / "cache",
    )
    result = WarmStartPipeline(
        config=config,
        rewriter=ConservativeQueryRewriter(),
        clip_encoder=DeterministicMockClipEncoder(),
        reporter=DeterministicMockEventReporter(),
    ).run(
        videos=(VideoInput(
            "v1", video_path, ((4.0, 5.5),), model_clock="local",
        ),),
        question="What visible state changes?",
    )
    sketch = result.videos[0]
    assert sketch.model_clock == "local"
    assert 1.4 <= sketch.duration_seconds <= 1.6
    assert max(frame.actual_timestamp_seconds for frame in sketch.frame_manifest) <= 1.5
    assert min(float(frame.source_timestamp_seconds) for frame in sketch.frame_manifest) >= 4.0
    assert "4.000" not in sketch.report_text
    payload = result.to_dict()["videos"][0]
    assert payload["model_clock"] == "local"
    assert "source_timestamp_seconds" in payload["frame_manifest"][0]
