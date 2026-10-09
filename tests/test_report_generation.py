import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from toolcvr.agents import ObserverAgent
from toolcvr.report_generation import (
    SelectedReportGenerator,
    load_frozen_selection,
    remove_completed_batch_checkpoints,
)
from toolcvr.reporting import DeterministicMockEventReporter
from toolcvr.schemas import ObservationBatch, ObservationGap, QueryAssessment, VideoInput


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
        for index in range(30):
            image = np.full((64, 96, 3), index * 4, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()


def _selection(path: Path) -> None:
    frames = [
        {
            "batch_index": 1,
            "video_id": "v1",
            "segment_id": "S1",
            "frame_id": frame_id,
            "timeline_index": timeline_index,
            "requested_timestamp_seconds": timestamp,
            "actual_timestamp_seconds": timestamp,
            "requested_frame_index": frame_index,
            "actual_frame_index": frame_index,
            "matched_queries": ["an object changes"],
        }
        for frame_id, timeline_index, timestamp, frame_index in (
            ("S1-F01", 0, 0.5, 5),
            ("S1-F02", 1, 1.5, 15),
        )
    ]
    path.write_text(
        json.dumps(
            {
                "question": "What changes?",
                "options": ["A. The object"],
                "query_plan": {
                    "phrases": [
                        {
                            "text": "an object changes",
                            "source": "question",
                            "weight": 1.0,
                            "target_video_ids": [],
                        }
                    ],
                    "report_hints": ["Describe the visible change."],
                },
                "videos": [
                    {
                        "video_id": "v1",
                        "duration_seconds": 3.0,
                        "clip_scan_stride_seconds": 1.0,
                        "clip_candidate_count": 3,
                        "cache_hit": False,
                        "observed_spans": [[0.5, 1.5]],
                        "events": [],
                        "report_batches": 1,
                        "selected_frames": 2,
                        "selection_mode": "query_ranked_ratio",
                        "frame_manifest": frames,
                    }
                ],
                "run": {
                    "sample_id": "CC-1",
                    "task": "CC",
                    "selection_only": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_legacy_source_clock_selection_is_rejected_for_a_local_clip(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.avi"
    selection_path = tmp_path / "selection.json"
    _write_video(video_path)
    _selection(selection_path)
    selection = load_frozen_selection(selection_path)
    try:
        selection.validate_video_inputs((
            VideoInput(
                "v1", video_path, ((0.0, 2.0),), model_clock="local",
            ),
        ))
    except ValueError as exc:
        message = str(exc)
        assert "model_clock='source'" in message
        assert "requires 'local'" in message
        assert "source-time leakage" in message
    else:
        raise AssertionError("legacy absolute-time selection was reused for a local clip")


def test_frozen_selection_generates_report_and_resumes_per_batch(tmp_path: Path) -> None:
    video_path = tmp_path / "video.avi"
    selection_path = tmp_path / "selection.json"
    _write_video(video_path)
    _selection(selection_path)
    selection = load_frozen_selection(selection_path)

    class CountingReporter(DeterministicMockEventReporter):
        def __init__(self) -> None:
            self.calls = 0

        def report(self, **kwargs):
            self.calls += 1
            return super().report(**kwargs)

    reporter = CountingReporter()
    generator = SelectedReportGenerator(
        observer=ObserverAgent(reporter),
        reporter_identity="counting-mock-v1",
        decode_long_edge=64,
    )
    arguments = {
        "selection": selection,
        "videos": (VideoInput("v1", video_path),),
        "checkpoint_directory": tmp_path / "checkpoints",
    }
    first = generator.generate(**arguments, resume=False)
    second = generator.generate(**arguments, resume=True)

    assert reporter.calls == 1
    assert first.to_dict() == second.to_dict()
    assert first.videos[0].frame_manifest == selection.videos[0].frame_manifest
    assert first.videos[0].report_text.startswith(
        "VIDEO_REPORT v1\nCOVERAGE selected=2/3"
    )
    assert "omitted sparse candidates are unobserved" in first.videos[0].report_text
    assert "evidence=S1-F01,S1-F02" in first.videos[0].report_text
    assert (tmp_path / "checkpoints" / "v1" / "batch-0001.json").is_file()


def test_explicit_observation_gap_does_not_discard_the_sample(tmp_path: Path) -> None:
    video_path = tmp_path / "video.avi"
    selection_path = tmp_path / "selection.json"
    _write_video(video_path)
    _selection(selection_path)
    selection = load_frozen_selection(selection_path)

    class GapReporter:
        def report(self, *, frames, **kwargs):
            del kwargs
            return ObservationBatch(
                events=(),
                observation_gaps=(
                    ObservationGap(
                        gap_id="gap1",
                        start_seconds=frames[0].actual_timestamp_seconds,
                        end_seconds=frames[-1].actual_timestamp_seconds,
                        frame_ids=tuple(frame.frame_id for frame in frames),
                        reason="model output remained invalid",
                    ),
                ),
                query_assessment=QueryAssessment(
                    relevance="insufficient",
                    unresolved_needs=("local batch has no validated description",),
                ),
            )

    result = SelectedReportGenerator(
        observer=ObserverAgent(GapReporter()),
        reporter_identity="gap-mock-v1",
        decode_long_edge=64,
    ).generate(
        selection=selection,
        videos=(VideoInput("v1", video_path),),
        checkpoint_directory=tmp_path / "checkpoints",
        resume=False,
    )

    video = result.videos[0]
    assert not video.events
    assert len(video.observation_gaps) == 1
    assert "OBSERVATION GAPS (not visual evidence)" in video.report_text
    assert "model output remained invalid" in video.report_text


def test_frozen_selection_rebatches_more_than_sixteen_frames(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    _selection(selection_path)
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    prototype = raw["videos"][0]["frame_manifest"][0]
    manifest = []
    for index in range(17):
        frame = dict(prototype)
        frame["frame_id"] = f"S1-F{index + 1:02d}"
        frame["timeline_index"] = index
        frame["requested_timestamp_seconds"] = float(index)
        frame["actual_timestamp_seconds"] = float(index)
        frame["requested_frame_index"] = index
        frame["actual_frame_index"] = index
        manifest.append(frame)
    raw["videos"][0]["frame_manifest"] = manifest
    raw["videos"][0]["selected_frames"] = 17
    selection_path.write_text(json.dumps(raw), encoding="utf-8")
    selection = load_frozen_selection(selection_path)
    assert selection.videos[0].report_batches == 2
    assert [trace.batch_index for trace in selection.videos[0].frame_manifest] == [
        *([1] * 16),
        2,
    ]


def test_independent_report_batches_run_concurrently_and_merge_in_order(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.avi"
    selection_path = tmp_path / "selection.json"
    _write_video(video_path)
    _selection(selection_path)
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    prototype = raw["videos"][0]["frame_manifest"][0]
    manifest = []
    for index, (timestamp, frame_index) in enumerate(
        ((0.3, 3), (0.8, 8), (1.3, 13), (1.8, 18)),
        start=1,
    ):
        frame = dict(prototype)
        frame.update(
            frame_id=f"S1-F{index:02d}",
            timeline_index=index - 1,
            requested_timestamp_seconds=timestamp,
            actual_timestamp_seconds=timestamp,
            requested_frame_index=frame_index,
            actual_frame_index=frame_index,
        )
        manifest.append(frame)
    raw["videos"][0]["frame_manifest"] = manifest
    raw["videos"][0]["selected_frames"] = len(manifest)
    raw["videos"][0]["clip_candidate_count"] = len(manifest)
    selection_path.write_text(json.dumps(raw), encoding="utf-8")
    selection = load_frozen_selection(selection_path, maximum_frames_per_batch=2)

    class ConcurrentReporter(DeterministicMockEventReporter):
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.maximum_active = 0
            self.calls = 0

        def report(self, **kwargs):
            with self.lock:
                self.active += 1
                self.calls += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                time.sleep(0.05)
                return super().report(**kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    reporter = ConcurrentReporter()
    generator = SelectedReportGenerator(
        observer=ObserverAgent(reporter),
        reporter_identity="concurrent-mock-v1",
        decode_long_edge=64,
        maximum_frames_per_batch=2,
        batch_workers=2,
    )
    arguments = dict(
        selection=selection,
        videos=(VideoInput("v1", video_path),),
        checkpoint_directory=tmp_path / "checkpoints",
    )
    result = generator.generate(**arguments, resume=False)
    resumed = generator.generate(**arguments, resume=True)

    assert reporter.calls == 2
    assert reporter.maximum_active == 2
    assert resumed.to_dict() == result.to_dict()
    assert result.videos[0].report_batches == 2
    assert [item.evidence_frame_ids for item in result.videos[0].events] == [
        ("S1-F01", "S1-F02"),
        ("S1-F03", "S1-F04"),
    ]


def test_frozen_selection_packs_sparse_segments_in_time_order(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    _selection(selection_path)
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    second = dict(raw["videos"][0]["frame_manifest"][-1])
    second.update(
        {
            "batch_index": 1,
            "segment_id": "S2",
            "frame_id": "S2-F01",
            "timeline_index": 2,
            "requested_timestamp_seconds": 2.5,
            "actual_timestamp_seconds": 2.5,
            "requested_frame_index": 25,
            "actual_frame_index": 25,
        }
    )
    raw["videos"][0]["frame_manifest"].append(second)
    raw["videos"][0]["selected_frames"] = 3
    raw["videos"][0]["observed_spans"] = [[0.5, 1.5], [2.5, 2.5]]
    selection_path.write_text(json.dumps(raw), encoding="utf-8")

    selection = load_frozen_selection(selection_path)
    video = selection.videos[0]
    assert video.report_batches == 1
    assert [trace.segment_id for trace in video.frame_manifest] == ["S1", "S1", "S2"]
    assert [trace.actual_timestamp_seconds for trace in video.frame_manifest] == [
        0.5,
        1.5,
        2.5,
    ]


def test_completed_batch_checkpoints_are_removed_safely(tmp_path: Path) -> None:
    sample_root = tmp_path / "CC" / "CC-1"
    checkpoint_root = sample_root / "batches"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "batch.json").write_text("{}", encoding="utf-8")
    remove_completed_batch_checkpoints(
        checkpoint_root,
        sample_directory=sample_root,
    )
    assert not checkpoint_root.exists()

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    try:
        remove_completed_batch_checkpoints(
            unrelated,
            sample_directory=sample_root,
        )
    except ValueError as exc:
        assert "refusing to remove" in str(exc)
    else:
        raise AssertionError("checkpoint cleanup accepted an unrelated directory")
