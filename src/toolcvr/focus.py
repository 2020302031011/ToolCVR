from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from toolcvr.agents import ObserverAgent
from toolcvr.schemas import (
    ObservationBatch,
    ReportFrame,
    SelectedFrameTrace,
    VideoInput,
)
from toolcvr.video import OpenCVVideoReader


@dataclass(frozen=True)
class FocusConfig:
    default_radius_seconds: float = 4.0
    frames_per_call: int = 16
    decode_long_edge: int = 512

    def __post_init__(self) -> None:
        if self.default_radius_seconds <= 0:
            raise ValueError("focus default radius must be positive")
        if not 2 <= self.frames_per_call <= 16:
            raise ValueError("focus frames_per_call must lie in [2, 16]")
        if self.decode_long_edge <= 0:
            raise ValueError("focus decode_long_edge must be positive")


@dataclass(frozen=True)
class FocusRequest:
    video_id: str
    evidence_need: str
    center_timestamp_seconds: float | None = None
    time_span: tuple[float, float] | None = None
    sampling_fps: float | None = None

    def __post_init__(self) -> None:
        if not self.video_id.strip() or not self.evidence_need.strip():
            raise ValueError("focus video_id and evidence_need must be nonempty")
        has_center = self.center_timestamp_seconds is not None
        has_span = self.time_span is not None
        if has_center == has_span:
            raise ValueError("focus request requires exactly one of timestamp or time_span")
        if self.center_timestamp_seconds is not None and self.center_timestamp_seconds < 0:
            raise ValueError("focus timestamp must be nonnegative")
        if self.time_span is not None:
            start, end = self.time_span
            if start < 0 or end <= start:
                raise ValueError("focus time_span must be an ordered positive interval")
        if self.sampling_fps is not None and self.sampling_fps <= 0:
            raise ValueError("focus sampling_fps must be positive when provided")


@dataclass(frozen=True)
class FocusResult:
    video_id: str
    request_kind: str
    resolved_time_span: tuple[float, float]
    observation: ObservationBatch
    frame_manifest: tuple[SelectedFrameTrace, ...]
    report_text: str

    def __post_init__(self) -> None:
        if not self.video_id.strip() or not self.report_text.strip():
            raise ValueError("focus result requires video_id and rendered report text")
        if not self.frame_manifest:
            raise ValueError("focus result requires the decoded frame manifest")
        frames = {frame.frame_id: frame for frame in self.frame_manifest}
        if len(frames) != len(self.frame_manifest):
            raise ValueError("focus result frame IDs must be unique")
        if any(frame.video_id != self.video_id for frame in self.frame_manifest):
            raise ValueError("focus result may contain frames from one video only")
        for event in self.observation.events:
            _validate_observation_reference(
                identifier=f"event {event.event_id}",
                frame_ids=event.evidence_frame_ids,
                start_seconds=event.start_seconds,
                end_seconds=event.end_seconds,
                frames=frames,
            )
        for gap in self.observation.observation_gaps:
            _validate_observation_reference(
                identifier=f"gap {gap.gap_id}",
                frame_ids=gap.frame_ids,
                start_seconds=gap.start_seconds,
                end_seconds=gap.end_seconds,
                frames=frames,
            )


def _validate_observation_reference(
    *,
    identifier: str,
    frame_ids: Sequence[str],
    start_seconds: float,
    end_seconds: float,
    frames: dict[str, SelectedFrameTrace],
) -> None:
    unknown = [frame_id for frame_id in frame_ids if frame_id not in frames]
    if unknown:
        raise ValueError(f"focus {identifier} cites frames outside its manifest: {unknown}")
    times = [frames[frame_id].actual_timestamp_seconds for frame_id in frame_ids]
    if start_seconds > min(times) + 0.25 or end_seconds < max(times) - 0.25:
        raise ValueError(f"focus {identifier} time span does not contain its cited frames")


class FocusTool:
    """Resolve a timestamp or interval into one dense, source-preserving clip."""

    def __init__(
        self,
        *,
        observer: ObserverAgent,
        config: FocusConfig = FocusConfig(),
        reader: OpenCVVideoReader | None = None,
    ) -> None:
        self.observer = observer
        self.config = config
        self.reader = reader or OpenCVVideoReader()

    def run(
        self,
        *,
        video: VideoInput,
        request: FocusRequest,
        question: str,
        report_hints: Sequence[str] = (),
    ) -> FocusResult:
        if video.video_id != request.video_id:
            raise ValueError(
                f"focus video mismatch: request={request.video_id} input={video.video_id}"
            )
        if not question.strip():
            raise ValueError("focus question must be nonempty")
        metadata = self.reader.probe(video.path)
        source_duration = metadata.frame_count / metadata.fps
        last_timestamp = min(
            video.model_duration(source_duration),
            max(0.0, video.model_duration(source_duration) - 1.0 / metadata.fps),
        )
        start, end, request_kind = self._resolve_span(
            request,
            last_timestamp=last_timestamp,
            boundary_tolerance_seconds=1.0 / metadata.fps,
        )
        if request.sampling_fps is None:
            model_timestamps = _uniform_model_timestamps(
                start_seconds=start, end_seconds=end, source_fps=metadata.fps,
                maximum_frames=self.config.frames_per_call,
            )
        else:
            model_timestamps = _dense_model_timestamps(
                start_seconds=start, end_seconds=end,
                sampling_fps=request.sampling_fps,
                maximum_frames=self.config.frames_per_call,
            )
        mapped: list[tuple[float, float, int, int]] = []
        seen_frame_indices: set[int] = set()
        for model_timestamp in model_timestamps:
            source_timestamp, source_span = video.model_to_source_timestamp(
                model_timestamp, source_duration,
            )
            frame_index = min(
                metadata.frame_count - 1,
                max(0, round(source_timestamp * metadata.fps)),
            )
            if frame_index in seen_frame_indices:
                continue
            seen_frame_indices.add(frame_index)
            mapped.append((model_timestamp, source_timestamp, frame_index, source_span))
        if not mapped:
            raise RuntimeError("focus clock mapping produced no unique source frames")
        decoded = self.reader.extract(
            video.path,
            [item[1] for item in mapped],
            long_edge=self.config.decode_long_edge,
            expected_frame_indices=[item[2] for item in mapped],
        )
        if len(decoded) != len(mapped):
            raise RuntimeError("focus decode did not return every requested source frame")

        segment_prefix = f"FOCUS-{round(start * 1000):010d}-{round(end * 1000):010d}"
        report_frames = tuple(
            ReportFrame(
                video_id=video.video_id,
                segment_id=f"{segment_prefix}-P{source_span:02d}",
                frame_id=f"{segment_prefix}-P{source_span:02d}-F{position:02d}",
                requested_timestamp_seconds=model_timestamp,
                actual_timestamp_seconds=model_timestamp,
                source_frame_index=frame.frame_index,
                matched_queries=(request.evidence_need,),
                rgb=frame.rgb,
                source_timestamp_seconds=frame.timestamp_seconds,
            )
            for position, ((model_timestamp, _, _, source_span), frame) in enumerate(
                zip(mapped, decoded),
                start=1,
            )
        )
        observation = self.observer.observe(
            video_id=video.video_id,
            frames=report_frames,
            question=question,
            report_hints=(*report_hints, request.evidence_need),
        )
        manifest = tuple(
            SelectedFrameTrace(
                batch_index=1,
                video_id=frame.video_id,
                segment_id=frame.segment_id,
                frame_id=frame.frame_id,
                timeline_index=position,
                requested_timestamp_seconds=frame.requested_timestamp_seconds,
                actual_timestamp_seconds=frame.actual_timestamp_seconds,
                requested_frame_index=frame.source_frame_index,
                actual_frame_index=frame.source_frame_index,
                matched_queries=frame.matched_queries,
                source_timestamp_seconds=frame.source_timestamp_seconds,
            )
            for position, frame in enumerate(report_frames)
        )
        return FocusResult(
            video_id=video.video_id,
            request_kind=request_kind,
            resolved_time_span=(start, end),
            observation=observation,
            frame_manifest=manifest,
            report_text=self.observer.render_report_text(
                video_id=video.video_id,
                events=observation.events,
                query_assessments=(observation.query_assessment,),
                observation_gaps=observation.observation_gaps,
            ),
        )

    def _resolve_span(
        self,
        request: FocusRequest,
        *,
        last_timestamp: float,
        boundary_tolerance_seconds: float = 0.0,
    ) -> tuple[float, float, str]:
        if boundary_tolerance_seconds < 0:
            raise ValueError("focus boundary tolerance must be nonnegative")
        maximum_locator = last_timestamp + boundary_tolerance_seconds + 1e-9
        if request.center_timestamp_seconds is not None:
            center = request.center_timestamp_seconds
            if center > maximum_locator:
                raise ValueError(
                    f"focus timestamp {center:.3f}s lies outside video ending at "
                    f"{last_timestamp:.3f}s"
                )
            # Duration metadata commonly denotes the boundary after the final
            # frame, while the last decodable timestamp is one frame earlier.
            # Clamp only that single-frame rounding gap; larger mistakes still
            # fail loudly above.
            center = min(center, last_timestamp)
            return (
                max(0.0, center - self.config.default_radius_seconds),
                min(last_timestamp, center + self.config.default_radius_seconds),
                "timestamp",
            )
        if request.time_span is None:
            raise AssertionError("validated focus request has no locator")
        start, end = request.time_span
        if start > maximum_locator:
            raise ValueError(
                f"focus interval starts outside video ending at {last_timestamp:.3f}s"
            )
        start = min(start, last_timestamp)
        return start, min(end, last_timestamp), "time_span"


def _uniform_model_timestamps(
    *,
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    maximum_frames: int,
) -> list[float]:
    available = max(1, math.floor((end_seconds - start_seconds) * source_fps) + 1)
    count = min(maximum_frames, available)
    if count == 1:
        return [start_seconds]
    return [float(value) for value in np.linspace(start_seconds, end_seconds, num=count)]


def split_complete_zoom(
    *,
    video_id: str,
    time_span: tuple[float, float],
    evidence_need: str,
    frames_per_call: int = 16,
    sampling_fps: float = 1.0,
) -> tuple[FocusRequest, ...]:
    """Partition a Reasoner-selected complete zoom into dense contiguous calls.

    The caller selects the region; this helper only enforces the visual input
    limit.  Neighbouring calls share one endpoint frame, so every instant of
    the requested region belongs to at least one dense observation.
    """
    start, end = time_span
    if start < 0 or end <= start or frames_per_call < 2 or sampling_fps <= 0:
        raise ValueError("invalid complete zoom specification")
    width = (frames_per_call - 1) / sampling_fps
    values: list[FocusRequest] = []
    cursor = start
    while cursor < end - 1e-9:
        stop = min(end, cursor + width)
        values.append(FocusRequest(
            video_id=video_id, evidence_need=evidence_need, time_span=(cursor, stop),
            sampling_fps=sampling_fps,
        ))
        cursor = stop
    return tuple(values)


def _dense_model_timestamps(
    *,
    start_seconds: float,
    end_seconds: float,
    sampling_fps: float,
    maximum_frames: int,
) -> list[float]:
    # The interval is closed at both ends.  ``split_complete_zoom`` bounds it
    # to (N-1)/fps, hence a 16-frame call at 1fps contains exactly 16 samples.
    count = int(math.floor((end_seconds - start_seconds) * sampling_fps + 1e-8)) + 1
    if count > maximum_frames:
        raise ValueError("dense focus span exceeds the per-call frame budget")
    return [
        min(end_seconds, start_seconds + index / sampling_fps)
        for index in range(count)
    ]
