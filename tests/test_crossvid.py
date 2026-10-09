import json
from pathlib import Path

import cv2
import numpy as np

from toolcvr.crossvid import (
    CrossVidAuditAdapter,
    SUPPORTED_CROSSVID_TASKS,
    crossvid_sample_id,
)
from toolcvr.schemas import VideoInput
from toolcvr.video import OpenCVVideoReader


def _write_video(path: Path, *, frame_count: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "mp4v" if path.suffix == ".mp4" else "MJPG"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 10.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("test environment cannot create MJPG video")
    try:
        for index in range(frame_count):
            image = np.full((48, 64, 3), index, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()


def _build_root(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    root = tmp_path / "crossvid"
    for name in ("a.avi", "b.avi", "c.avi", "d.avi", "e.avi"):
        _write_video(root / "videos" / name)
    for view in (1, 2):
        frame_directory = root / "uav" / "frames" / str(view) / f"22-{view}"
        frame_directory.mkdir(parents=True)
        for index in range(3):
            cv2.imwrite(str(frame_directory / f"{index:04d}.jpg"), np.full((48, 64, 3), 40 * index, np.uint8))
        bbox_directory = root / "uav" / "bbox" / str(view)
        bbox_directory.mkdir(parents=True)
        (bbox_directory / "22.json").write_text(
            json.dumps(
                [
                    {
                        "id": index,
                        "bbox": {
                            str(frame): {"xtl": 5, "ytl": 5, "xbr": 25, "ybr": 25, "outside": False}
                            for frame in range(3)
                        },
                    }
                    for index in range(5)
                ]
            ),
            encoding="utf-8",
        )
    common_choice = {"question": "What visible action occurs?", "options": ["A. one", "B. two"], "answer": "A"}
    rows: dict[str, dict[str, object]] = {
        "CC": {**common_choice, "id": 0, "videos": ["a.avi", "b.avi"]},
        "NC": {**common_choice, "id": 0, "videos": ["a.avi", "b.avi"]},
        "CCQA": {"id": 0, "video A": "a.avi", "video B": "b.avi", "question": "How do they differ?", "answer": "x"},
        "PEA": {**common_choice, "id": 0, "videos": ["a.avi", "b.avi"], "begin": [1, 2], "end": [3, 4]},
        "FSA": {"id": 0, "video A": "a.avi", "video B": "b.avi", "ref_segment": [1, 3], "answer": [2, 4]},
        "PI": {"id": 0, "video": "a.avi", "beginning": [0, 2], "ending": [4, 6], "options": ["A. one"], "answer": "A"},
        "PSS": {"id": 0, "video": "a.avi", "segments": {"1": [[1, 2]], "2": [[3, 4], [5, 6]]}, "answer": "1->2"},
        "MOC": {**common_choice, "id": 0, "vid": 22, "objects": [{"id": index, "label": "car"} for index in range(5)]},
        "MSR": {**common_choice, "id": 0, "vid": 22, "objects": [{"id": index, "label": "car"} for index in range(5)]},
    }
    return root, rows


def test_all_nine_tasks_normalize_to_one_video_input_protocol(tmp_path: Path) -> None:
    root, rows = _build_root(tmp_path)
    adapter = CrossVidAuditAdapter(root, proxy_directory=tmp_path / "proxies")
    for task in SUPPORTED_CROSSVID_TASKS:
        sample = adapter.normalize(task, rows[task], 0)
        assert sample.sample_id == f"{task}-0"
        assert sample.videos
        assert all(isinstance(video, VideoInput) for video in sample.videos)
    assert adapter.normalize("FSA", rows["FSA"], 0).videos[0].time_spans == ((1.0, 3.0),)
    assert adapter.normalize("PSS", rows["PSS"], 0).videos[1].time_spans == ((3.0, 4.0), (5.0, 6.0))
    moc_sample = adapter.normalize("MOC", rows["MOC"], 0)
    moc_video = moc_sample.videos[0]
    assert moc_video.scan_stride_seconds == 2.0
    assert moc_video.synchronization_group == "uav-scene-22"
    assert any("cycles means two-wheeled" in note for note in moc_sample.adaptation_notes)
    assert any("when/after/once" in note for note in moc_sample.adaptation_notes)
    assert moc_sample.answer_kind == "choice"
    assert any("source_track=0" in note for note in moc_sample.adaptation_notes)
    pss_sample = adapter.normalize("PSS", rows["PSS"], 0)
    assert pss_sample.answer_kind == "sequence"
    assert pss_sample.required_answer_ids == ("v1", "v2")
    assert all(video.model_clock == "local" for video in pss_sample.videos)
    assert abs(pss_sample.videos[1].model_duration(8.0) - 2.001) < 1e-9
    assert abs(
        pss_sample.videos[1].source_to_model_timestamp(5.5, 8.0) - 1.501
    ) < 1e-9
    source_timestamp, source_span = pss_sample.videos[1].model_to_source_timestamp(
        1.501, 8.0,
    )
    assert abs(source_timestamp - 5.5) < 1e-9
    assert source_span == 2
    assert pss_sample.videos[1].model_time_domains(2.001) == (
        (0.0, 1.0), (1.001, 2.001),
    )
    assert any("slot ids" in note.lower() for note in pss_sample.adaptation_notes)
    moc_frames = list(
        OpenCVVideoReader().iter_sparse(
            moc_video.path,
            stride_seconds=moc_video.scan_stride_seconds,
            long_edge=64,
        )
    )
    # Three source UAV images at 30 fps span only 0.1 seconds.  A 2-second
    # sparse scan must not reinterpret them as a six-second sequence.
    assert len(moc_frames) == 1
    assert moc_frames[0].timestamp_seconds < 0.1


def test_sample_id_lookup_does_not_validate_unselected_payload() -> None:
    malformed_unselected_row = {
        "id": 91,
        "segments": {"1": [[8, 10], [2, 4]]},
    }
    assert crossvid_sample_id("pss", malformed_unselected_row, 7) == "PSS-91"


def test_sparse_interval_decode_preserves_source_time_and_frame_index(tmp_path: Path) -> None:
    path = tmp_path / "source.avi"
    _write_video(path)
    frames = list(
        OpenCVVideoReader().iter_sparse(
            path,
            stride_seconds=1.0,
            long_edge=64,
            time_spans=((2.0, 4.0), (6.0, 7.5)),
        )
    )
    assert frames
    assert all(2.0 <= frame.timestamp_seconds <= 4.1 or 6.0 <= frame.timestamp_seconds <= 7.6 for frame in frames)
    assert frames[0].frame_index >= 19
    assert all(right.frame_index > left.frame_index for left, right in zip(frames[:-1], frames[1:]))


def test_video_input_rejects_overlapping_source_intervals(tmp_path: Path) -> None:
    path = tmp_path / "source.avi"
    _write_video(path)
    try:
        VideoInput("v1", path, ((1.0, 3.0), (2.0, 4.0)))
    except ValueError as exc:
        assert "non-overlapping" in str(exc)
    else:
        raise AssertionError("overlapping intervals must be rejected")


def test_pea_repairs_only_a_unique_same_recording_rgb_asset(tmp_path: Path) -> None:
    root = tmp_path / "crossvid"
    hmc = root / "videos" / "assembly" / "recording_HMC_camera_mono10bit.mp4"
    rgb = root / "videos" / "assembly" / "recording_C10118_rgb.mp4"
    _write_video(hmc)
    _write_video(rgb, frame_count=220)
    row = {
        "id": 7,
        "question": "Which process is incorrect?",
        "options": ["A. first", "B. second"],
        "answer": "A",
        "videos": ["assembly/recording_HMC_camera_mono10bit.mp4"],
        "begin": [2.0],
        "end": [20.0],
    }
    adapter = CrossVidAuditAdapter(root, proxy_directory=tmp_path / "proxies")
    sample = adapter.normalize("PEA", row, 0)
    assert sample.videos[0].path == rgb.resolve()
    assert sample.adaptation_notes
    assert "interval-incompatible HMC" in sample.adaptation_notes[0]
