from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from toolcvr.clip import ClipEncoder
from toolcvr.schemas import FeatureTimeline, SparseFrame, VideoInput


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0 or self.fps <= 0 or self.frame_count <= 0:
            raise ValueError("video metadata must be positive")


class OpenCVVideoReader:
    def probe(self, path: Path) -> VideoMetadata:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"video has invalid fps/frame count: {path}")
        return VideoMetadata(
            duration_seconds=frame_count / fps,
            fps=fps,
            frame_count=frame_count,
        )

    def iter_sparse(
        self,
        path: Path,
        *,
        stride_seconds: float,
        long_edge: int,
        time_spans: tuple[tuple[float, float], ...] = (),
    ) -> Iterator[SparseFrame]:
        if stride_seconds <= 0 or long_edge <= 0:
            raise ValueError("sparse decode parameters must be positive")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise RuntimeError(f"video has invalid fps: {path}")

        metadata_duration = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        spans = _bounded_spans(time_spans, metadata_duration)
        next_target = spans[0][0]
        span_position = 0
        emitted = 0
        frame_index = -1
        try:
            while True:
                ok = capture.grab()
                if not ok:
                    break
                frame_index += 1
                nominal = frame_index / fps
                while span_position < len(spans) and nominal > spans[span_position][1] + 0.5 / fps:
                    span_position += 1
                    if span_position < len(spans):
                        next_target = spans[span_position][0]
                if span_position >= len(spans):
                    break
                span_start, span_end = spans[span_position]
                if nominal + (0.5 / fps) < span_start or nominal > span_end + (0.5 / fps):
                    continue
                if nominal + (0.5 / fps) < next_target:
                    continue
                ok, bgr = capture.retrieve()
                if not ok or bgr is None:
                    raise RuntimeError(f"failed to decode frame {frame_index} from {path}")
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not np.isfinite(pts) or pts < 0:
                    pts = nominal
                # Container PTS can differ slightly from ``frame_index / fps``.
                # The nominal timestamp is used only to decide when decoding is
                # worthwhile; source provenance must use the actual PTS.  Do not
                # emit a boundary frame whose nominal time is inside the requested
                # span while its actual PTS is outside it (for example 864.997s
                # for a span beginning at 865.000s).  The next decoded frame will
                # still be considered because ``next_target`` advances only after
                # a frame is emitted.
                if pts < span_start - 1e-6 or pts > span_end + 1e-6:
                    continue
                rgb = cv2.cvtColor(_resize_long_edge(bgr, long_edge), cv2.COLOR_BGR2RGB)
                yield SparseFrame(timestamp_seconds=pts, frame_index=frame_index, rgb=rgb)
                emitted += 1
                next_target += stride_seconds
        finally:
            capture.release()
        if emitted == 0:
            raise RuntimeError(f"video produced no sparse frames: {path}")
    def extract(
        self,
        path: Path,
        timestamps: list[float],
        *,
        long_edge: int,
        expected_frame_indices: list[int] | None = None,
    ) -> list[SparseFrame]:
        if not timestamps:
            return []
        if any(value < 0 for value in timestamps):
            raise ValueError("requested timestamps must be nonnegative")
        if any(right <= left for left, right in zip(timestamps[:-1], timestamps[1:])):
            raise ValueError("requested timestamps must be strictly increasing")
        if expected_frame_indices is not None:
            if len(expected_frame_indices) != len(timestamps):
                raise ValueError("expected frame indices must align with timestamps")
            if any(value < 0 for value in expected_frame_indices):
                raise ValueError("expected frame indices must be nonnegative")
            if any(
                right <= left
                for left, right in zip(expected_frame_indices[:-1], expected_frame_indices[1:])
            ):
                raise ValueError("expected frame indices must be strictly increasing")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise RuntimeError(f"video has invalid fps: {path}")
        frames: list[SparseFrame] = []
        try:
            for position, requested in enumerate(timestamps):
                expected_index = (
                    expected_frame_indices[position]
                    if expected_frame_indices is not None
                    else None
                )
                if expected_index is None:
                    capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
                else:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, expected_index)
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"failed to decode requested time {requested:.3f}s from {path}")
                frame_index = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                if expected_index is not None and frame_index != expected_index:
                    raise RuntimeError(
                        f"frame identity mismatch in {path}: requested index {expected_index}, "
                        f"decoded {frame_index} at cached PTS {requested:.6f}s"
                    )
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not np.isfinite(pts) or pts < 0:
                    raise RuntimeError(f"decoder returned invalid PTS for frame {frame_index} in {path}")
                if expected_index is not None:
                    # Some VFR/edit-list MP4 files return unstable POS_MSEC
                    # values after random frame seeking, even though OpenCV
                    # decodes the exact requested source frame.  The requested
                    # timestamp was captured during the original sequential
                    # scan and is therefore the canonical PTS for that exact
                    # frame identity.
                    pts = requested
                else:
                    tolerance = max(0.1, 2.0 / fps)
                    if abs(pts - requested) > tolerance:
                        raise RuntimeError(
                            f"PTS mismatch in {path}: requested={requested:.6f}s "
                            f"decoded={pts:.6f}s tolerance={tolerance:.6f}s"
                        )
                rgb = cv2.cvtColor(_resize_long_edge(bgr, long_edge), cv2.COLOR_BGR2RGB)
                frames.append(SparseFrame(pts, frame_index, rgb))
        finally:
            capture.release()
        return frames


def _bounded_spans(
    time_spans: tuple[tuple[float, float], ...],
    duration_seconds: float,
) -> tuple[tuple[float, float], ...]:
    if not time_spans:
        return ((0.0, duration_seconds),)
    bounded = tuple(
        (start, min(end, duration_seconds))
        for start, end in time_spans
        if start < duration_seconds and min(end, duration_seconds) > start
    )
    if not bounded:
        raise ValueError("all requested time spans lie outside the source video")
    return bounded


def _resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, long_edge / max(height, width))
    if scale >= 1.0:
        return image
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class TimelinePolicy:
    short_stride_seconds: float = 2.0
    long_stride_seconds: float = 4.0
    long_video_threshold_seconds: float = 600.0
    decode_long_edge: int = 336
    clip_batch_size: int = 64

    def __post_init__(self) -> None:
        values = (
            self.short_stride_seconds,
            self.long_stride_seconds,
            self.long_video_threshold_seconds,
        )
        if any(value <= 0 for value in values):
            raise ValueError("timeline stride/threshold values must be positive")
        if self.decode_long_edge <= 0:
            raise ValueError("timeline frame limits must be positive")
        if self.clip_batch_size <= 0:
            raise ValueError("clip_batch_size must be positive")

    def stride_for(self, duration_seconds: float) -> float:
        base = (
            self.short_stride_seconds
            if duration_seconds <= self.long_video_threshold_seconds
            else self.long_stride_seconds
        )
        return base


class ClipFeatureCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def key_path(
        self,
        video_path: Path,
        *,
        encoder_identity: str,
        stride_seconds: float,
        long_edge: int,
        time_spans: tuple[tuple[float, float], ...] = (),
        model_clock: str = "source",
    ) -> Path:
        stat = video_path.stat()
        identity = {
            "path": str(video_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "encoder": encoder_identity,
            "stride_seconds": round(stride_seconds, 6),
            "long_edge": long_edge,
            "time_spans": time_spans,
            "model_clock": model_clock,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.npz"

    @staticmethod
    def load(path: Path) -> FeatureTimeline | None:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return FeatureTimeline(
                    duration_seconds=float(data["duration_seconds"][0]),
                    stride_seconds=float(data["stride_seconds"][0]),
                    timestamps=data["timestamps"].astype(np.float64),
                    frame_indices=data["frame_indices"].astype(np.int64),
                    embeddings=data["embeddings"].astype(np.float32),
                    source_timestamps=(
                        data["source_timestamps"].astype(np.float64)
                        if "source_timestamps" in data
                        else data["timestamps"].astype(np.float64)
                    ),
                    source_span_indices=(
                        data["source_span_indices"].astype(np.int64)
                        if "source_span_indices" in data
                        else np.ones(len(data["timestamps"]), dtype=np.int64)
                    ),
                )
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"invalid CLIP feature cache {path}: {exc}") from exc

    @staticmethod
    def save(path: Path, timeline: FeatureTimeline) -> None:
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    duration_seconds=np.array([timeline.duration_seconds], dtype=np.float64),
                    stride_seconds=np.array([timeline.stride_seconds], dtype=np.float64),
                    timestamps=timeline.timestamps.astype(np.float64),
                    frame_indices=timeline.frame_indices.astype(np.int64),
                    embeddings=timeline.embeddings.astype(np.float16),
                    source_timestamps=timeline.source_timestamps.astype(np.float64),
                    source_span_indices=timeline.source_span_indices.astype(np.int64),
                )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def build_feature_timeline(
    video: VideoInput,
    *,
    reader: OpenCVVideoReader,
    encoder: ClipEncoder,
    policy: TimelinePolicy,
    stride_seconds: float | None = None,
) -> FeatureTimeline:
    video_path = video.path
    metadata = reader.probe(video_path)
    bounded_spans = _bounded_spans(video.time_spans, metadata.duration_seconds)
    covered_duration = sum(end - start for start, end in bounded_spans)
    stride = stride_seconds or policy.stride_for(covered_duration)
    timestamps: list[float] = []
    source_timestamps: list[float] = []
    source_span_indices: list[int] = []
    frame_indices: list[int] = []
    embedding_chunks: list[np.ndarray] = []
    image_batch: list[np.ndarray] = []

    def flush() -> None:
        if not image_batch:
            return
        values = encoder.encode_images(image_batch)
        if len(values) != len(image_batch):
            raise RuntimeError("CLIP image encoder returned the wrong batch size")
        embedding_chunks.append(values)
        image_batch.clear()

    for frame in reader.iter_sparse(
        video_path,
        stride_seconds=stride,
        long_edge=policy.decode_long_edge,
        time_spans=bounded_spans,
    ):
        source_timestamps.append(frame.timestamp_seconds)
        timestamps.append(video.source_to_model_timestamp(
            frame.timestamp_seconds, metadata.duration_seconds,
        ))
        source_span_indices.append(video.source_span_position(
            frame.timestamp_seconds, metadata.duration_seconds,
        ))
        frame_indices.append(frame.frame_index)
        image_batch.append(frame.rgb)
        if len(image_batch) >= policy.clip_batch_size:
            flush()
    flush()
    if not embedding_chunks:
        raise RuntimeError(f"no CLIP features produced for {video_path}")
    return FeatureTimeline(
        duration_seconds=video.model_duration(metadata.duration_seconds),
        stride_seconds=stride,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        embeddings=np.concatenate(embedding_chunks, axis=0).astype(np.float32),
        source_timestamps=np.asarray(source_timestamps, dtype=np.float64),
        source_span_indices=np.asarray(source_span_indices, dtype=np.int64),
    )
