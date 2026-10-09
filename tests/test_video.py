from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from toolcvr.video import OpenCVVideoReader


class _UnstablePtsCapture:
    def __init__(self) -> None:
        self.position = 0.0

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: float) -> bool:
        self.position = value
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return 30.0
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return 1380.0
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return 46146.1
        return 0.0

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        return None


def test_exact_frame_redecode_preserves_sequential_scan_pts() -> None:
    cached_pts = 45.912533
    with patch(
        "toolcvr.video.cv2.VideoCapture",
        return_value=_UnstablePtsCapture(),
    ):
        frames = OpenCVVideoReader().extract(
            Path("unstable-vfr.mp4"),
            [cached_pts],
            long_edge=512,
            expected_frame_indices=[1379],
        )
    assert len(frames) == 1
    assert frames[0].frame_index == 1379
    assert frames[0].timestamp_seconds == cached_pts


def test_timestamp_only_decode_still_rejects_pts_mismatch() -> None:
    with patch(
        "toolcvr.video.cv2.VideoCapture",
        return_value=_UnstablePtsCapture(),
    ):
        try:
            OpenCVVideoReader().extract(
                Path("unstable-vfr.mp4"),
                [45.912533],
                long_edge=512,
            )
        except RuntimeError as exc:
            assert "PTS mismatch" in str(exc)
        else:
            raise AssertionError("timestamp-only decode must reject a mismatched PTS")
