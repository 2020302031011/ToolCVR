from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np

LOCAL_CLOCK_DISCONTINUITY_SECONDS = 0.001


def _string_list_or_singleton(value: object, label: str) -> list[str]:
    """Normalize the common JSON mistake of emitting one string, not a list."""

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise ValueError(f"{label} must be a string or string list")
    if any(not item.strip() for item in values):
        raise ValueError(f"{label} must contain nonempty strings")
    return values


@dataclass(frozen=True)
class VideoInput:
    video_id: str
    path: Path
    time_spans: tuple[tuple[float, float], ...] = ()
    scan_stride_seconds: float | None = None
    synchronization_group: str | None = None
    model_clock: Literal["source", "local"] = "source"

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must be nonempty")
        if not self.path.is_file():
            raise FileNotFoundError(f"video does not exist: {self.path}")
        if self.scan_stride_seconds is not None and self.scan_stride_seconds <= 0:
            raise ValueError("scan_stride_seconds must be positive when provided")
        if self.synchronization_group is not None and not self.synchronization_group.strip():
            raise ValueError("synchronization_group must be nonempty when provided")
        if self.model_clock not in ("source", "local"):
            raise ValueError("model_clock must be source or local")
        if self.model_clock == "local" and not self.time_spans:
            raise ValueError("a local model clock requires explicit source time spans")
        previous_end = -1.0
        for start, end in self.time_spans:
            if start < 0 or end <= start:
                raise ValueError("video time spans must be ordered positive intervals")
            if start < previous_end:
                raise ValueError("video time spans must be sorted and non-overlapping")
            previous_end = end

    def model_duration(self, source_duration: float) -> float:
        """Return the duration visible to models, never a hidden source offset."""
        spans = self._bounded_source_spans(source_duration)
        if self.model_clock == "source":
            return source_duration
        return (
            sum(end - start for start, end in spans)
            + LOCAL_CLOCK_DISCONTINUITY_SECONDS * max(0, len(spans) - 1)
        )

    def model_time_domains(
        self, model_duration: float,
    ) -> tuple[tuple[float, float], ...]:
        """Return independently observable domains in model coordinates.

        The method intentionally needs only the already-published model
        duration.  Source offsets never leave this object, while callers can
        still prevent a coverage or Focus region from bridging two clips that
        were disjoint in the source recording.
        """
        if model_duration <= 0:
            raise ValueError("model duration must be positive")
        if self.model_clock == "source":
            return self.time_spans or ((0.0, model_duration),)
        domains: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in self.time_spans:
            if cursor >= model_duration - 1e-9:
                break
            domain_end = min(model_duration, cursor + (end - start))
            if domain_end > cursor:
                domains.append((cursor, domain_end))
            cursor = domain_end + LOCAL_CLOCK_DISCONTINUITY_SECONDS
        if not domains:
            raise ValueError("local model clock produced no visible time domain")
        return tuple(domains)

    def source_to_model_timestamp(
        self, source_timestamp: float, source_duration: float,
    ) -> float:
        if self.model_clock == "source":
            return source_timestamp
        elapsed = 0.0
        spans = self._bounded_source_spans(source_duration)
        for position, (start, end) in enumerate(spans):
            if start - 1e-6 <= source_timestamp <= end + 1e-6:
                return elapsed + min(max(source_timestamp, start), end) - start
            elapsed += end - start
            if position + 1 < len(spans):
                elapsed += LOCAL_CLOCK_DISCONTINUITY_SECONDS
        raise ValueError(
            f"source timestamp {source_timestamp:.6f}s is outside logical video {self.video_id}"
        )

    def model_to_source_timestamp(
        self, model_timestamp: float, source_duration: float,
    ) -> tuple[float, int]:
        """Map one model-visible time back to source time and span identity."""
        if model_timestamp < 0:
            raise ValueError("model timestamp must be nonnegative")
        if self.model_clock == "source":
            if model_timestamp > source_duration + 1e-6:
                raise ValueError("model timestamp lies outside source video")
            return min(model_timestamp, source_duration), 1
        spans = self._bounded_source_spans(source_duration)
        elapsed = 0.0
        for position, (start, end) in enumerate(spans, 1):
            width = end - start
            boundary = elapsed + width
            if model_timestamp <= boundary + 1e-9:
                return start + min(max(model_timestamp - elapsed, 0.0), width), position
            if position < len(spans):
                next_start = boundary + LOCAL_CLOCK_DISCONTINUITY_SECONDS
                if model_timestamp < next_start:
                    return spans[position][0], position + 1
                elapsed = next_start
        raise ValueError(
            f"model timestamp {model_timestamp:.6f}s lies outside logical video {self.video_id}"
        )

    def source_span_position(
        self, source_timestamp: float, source_duration: float,
    ) -> int:
        for position, (start, end) in enumerate(
            self._bounded_source_spans(source_duration), 1,
        ):
            if start - 1e-6 <= source_timestamp <= end + 1e-6:
                return position
        raise ValueError(
            f"source timestamp {source_timestamp:.6f}s is outside logical video {self.video_id}"
        )

    def _bounded_source_spans(
        self, source_duration: float,
    ) -> tuple[tuple[float, float], ...]:
        if source_duration <= 0:
            raise ValueError("source duration must be positive")
        if not self.time_spans:
            return ((0.0, source_duration),)
        spans = tuple(
            (start, min(end, source_duration))
            for start, end in self.time_spans
            if start < source_duration and min(end, source_duration) > start
        )
        if not spans:
            raise ValueError("all logical video spans lie outside the source video")
        return spans


@dataclass(frozen=True)
class RetrievalPhrase:
    text: str
    source: str
    weight: float = 1.0
    target_video_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("retrieval phrase must be nonempty")
        if not self.source.strip():
            raise ValueError("retrieval phrase source must be nonempty")
        if self.weight <= 0:
            raise ValueError("retrieval phrase weight must be positive")


@dataclass(frozen=True)
class QueryPlan:
    phrases: tuple[RetrievalPhrase, ...]
    report_hints: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.phrases) > 24:
            raise ValueError("query plan may contain at most 24 retrieval phrases")
        if any(not item.strip() for item in self.report_hints):
            raise ValueError("report hints must be nonempty")


@dataclass(frozen=True)
class SparseFrame:
    timestamp_seconds: float
    frame_index: int
    rgb: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.timestamp_seconds < 0:
            raise ValueError("frame timestamp must be nonnegative")
        if self.frame_index < 0:
            raise ValueError("frame index must be nonnegative")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("frame image must be RGB HxWx3")


@dataclass(frozen=True)
class FeatureTimeline:
    duration_seconds: float
    stride_seconds: float
    timestamps: np.ndarray = field(repr=False, compare=False)
    frame_indices: np.ndarray = field(repr=False, compare=False)
    embeddings: np.ndarray = field(repr=False, compare=False)
    source_timestamps: np.ndarray | None = field(default=None, repr=False, compare=False)
    source_span_indices: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("timeline duration and stride must be positive")
        if self.timestamps.ndim != 1 or self.frame_indices.ndim != 1:
            raise ValueError("timeline timestamps and frame indices must be vectors")
        if self.embeddings.ndim != 2:
            raise ValueError("timeline embeddings must be a matrix")
        size = len(self.timestamps)
        if size == 0 or len(self.frame_indices) != size or len(self.embeddings) != size:
            raise ValueError("timeline arrays must be nonempty and aligned")
        if self.source_timestamps is None:
            object.__setattr__(self, "source_timestamps", self.timestamps.copy())
        assert self.source_timestamps is not None
        if self.source_timestamps.ndim != 1 or len(self.source_timestamps) != size:
            raise ValueError("timeline source timestamps must be an aligned vector")
        if self.source_span_indices is None:
            object.__setattr__(
                self, "source_span_indices", np.ones(size, dtype=np.int64),
            )
        assert self.source_span_indices is not None
        if self.source_span_indices.ndim != 1 or len(self.source_span_indices) != size:
            raise ValueError("timeline source span indices must be an aligned vector")
        if np.any(self.source_span_indices <= 0):
            raise ValueError("timeline source span indices must be positive")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timeline timestamps must be strictly increasing")
        if np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("timeline frame indices must be strictly increasing")
        tolerance = max(0.25, self.stride_seconds)
        if self.timestamps[0] < 0 or self.timestamps[-1] > self.duration_seconds + tolerance:
            raise ValueError("timeline timestamps lie outside the video duration")


@dataclass(frozen=True)
class CandidateSpan:
    segment_id: str
    timeline_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    score: float
    matched_queries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must be nonempty")
        if not self.timeline_indices:
            raise ValueError("candidate span must retain at least one frame")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("candidate span time range is invalid")


@dataclass(frozen=True)
class BatchFrameRequest:
    segment_id: str
    timeline_index: int
    requested_timestamp_seconds: float
    source_timestamp_seconds: float
    requested_frame_index: int
    matched_queries: tuple[str, ...]


@dataclass(frozen=True)
class ReportFrame:
    video_id: str
    segment_id: str
    frame_id: str
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float
    source_frame_index: int
    matched_queries: tuple[str, ...]
    rgb: np.ndarray = field(repr=False, compare=False)
    source_timestamp_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.source_timestamp_seconds is None:
            object.__setattr__(self, "source_timestamp_seconds", self.actual_timestamp_seconds)
        if self.source_timestamp_seconds is not None and self.source_timestamp_seconds < 0:
            raise ValueError("source frame timestamp must be nonnegative")


@dataclass(frozen=True)
class SelectedFrameTrace:
    batch_index: int
    video_id: str
    segment_id: str
    frame_id: str
    timeline_index: int
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float
    requested_frame_index: int
    actual_frame_index: int
    matched_queries: tuple[str, ...]
    source_timestamp_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.batch_index <= 0:
            raise ValueError("batch_index must be positive")
        if self.requested_frame_index != self.actual_frame_index:
            raise ValueError("selected frame identity changed during VLM re-decode")
        if abs(self.actual_timestamp_seconds - self.requested_timestamp_seconds) > 0.1:
            raise ValueError("selected frame PTS changed during VLM re-decode")
        if self.source_timestamp_seconds is None:
            object.__setattr__(
                self, "source_timestamp_seconds", self.actual_timestamp_seconds,
            )
        if self.source_timestamp_seconds is not None and self.source_timestamp_seconds < 0:
            raise ValueError("selected source timestamp must be nonnegative")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SelectedFrameTrace":
        matched = value.get("matched_queries", [])
        if not isinstance(matched, list) or not all(
            isinstance(item, str) for item in matched
        ):
            raise ValueError("selected frame matched_queries must be a string list")
        return cls(
            batch_index=int(value["batch_index"]),
            video_id=str(value["video_id"]),
            segment_id=str(value["segment_id"]),
            frame_id=str(value["frame_id"]),
            timeline_index=int(value["timeline_index"]),
            requested_timestamp_seconds=float(value["requested_timestamp_seconds"]),
            actual_timestamp_seconds=float(value["actual_timestamp_seconds"]),
            requested_frame_index=int(value["requested_frame_index"]),
            actual_frame_index=int(value["actual_frame_index"]),
            matched_queries=tuple(matched),
            source_timestamp_seconds=float(
                value.get("source_timestamp_seconds", value["actual_timestamp_seconds"])
            ),
        )


@dataclass(frozen=True)
class EventRecord:
    """One source-cited visual event.

    ``description`` is restricted to directly visible attributes, actions and
    state changes. ``interpretation`` carries any query-conditioned identity,
    role, intent or causal hypothesis and is never direct visual evidence.
    """

    event_id: str
    start_seconds: float
    end_seconds: float
    description: str
    evidence_frame_ids: tuple[str, ...]
    confidence: float
    uncertainty: str = ""
    interpretation: str = ""

    def __post_init__(self) -> None:
        if not self.event_id or not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("event id and description must be nonempty")
        if not isinstance(self.uncertainty, str) or not isinstance(self.interpretation, str):
            raise ValueError("event uncertainty and interpretation must be strings")
        if self.interpretation.strip() and not self.uncertainty.strip():
            raise ValueError("an inferred event interpretation requires an explicit uncertainty")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("event time range is invalid")
        if not self.evidence_frame_ids:
            raise ValueError("event must cite at least one evidence frame")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must lie in [0, 1]")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EventRecord":
        span = value.get("time_span")
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("event time_span must contain [start, end]")
        evidence = value.get("evidence_frame_ids")
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise ValueError("event evidence_frame_ids must be a string list")
        return cls(
            event_id=str(value.get("event_id", "")).strip(),
            start_seconds=float(span[0]),
            end_seconds=float(span[1]),
            description=str(value.get("description", "")).strip(),
            evidence_frame_ids=tuple(evidence),
            confidence=float(value.get("confidence", 0.0)),
            uncertainty=str(value.get("uncertainty", "")).strip(),
            interpretation=str(value.get("interpretation", "")).strip(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "time_span": [self.start_seconds, self.end_seconds],
            "description": self.description,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class ObservationGap:
    """An explicit, provenance-preserving failure to obtain a visual event.

    A gap is not visual evidence and must never be cited as an event.  It lets
    one malformed VLM batch remain visible without discarding every valid
    batch from the same sample.
    """

    gap_id: str
    start_seconds: float
    end_seconds: float
    frame_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.gap_id.strip() or not self.reason.strip():
            raise ValueError("observation gap id and reason must be nonempty")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("observation gap time range is invalid")
        if not self.frame_ids or any(not item.strip() for item in self.frame_ids):
            raise ValueError("observation gap must retain nonempty frame ids")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("observation gap frame ids must be unique")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ObservationGap":
        span = value.get("time_span")
        frames = value.get("frame_ids")
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("observation gap time_span must contain [start, end]")
        if not isinstance(frames, list) or not all(isinstance(item, str) for item in frames):
            raise ValueError("observation gap frame_ids must be a string list")
        return cls(
            gap_id=str(value.get("gap_id", "")).strip(),
            start_seconds=float(span[0]),
            end_seconds=float(span[1]),
            frame_ids=tuple(frames),
            reason=str(value.get("reason", "")).strip(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "time_span": [self.start_seconds, self.end_seconds],
            "frame_ids": list(self.frame_ids),
            "reason": self.reason,
        }


QueryRelevance = Literal["direct", "indirect", "insufficient"]

VISUAL_TOOL_NAMES = frozenset(
    {
        "semantic_seek",
        "temporal_expand",
        "dense_focus",
        "boundary_refine",
        "before_after",
        "entity_track",
        "visual_count",
        "pair_verify",
        "ocr_read",
        "asr_search",
    }
)


@dataclass(frozen=True)
class QueryAssessment:
    """Question relevance kept separate from grounded visual observations."""

    relevance: QueryRelevance
    evidence_event_ids: tuple[str, ...] = ()
    unresolved_needs: tuple[str, ...] = ()
    suggested_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.relevance not in {"direct", "indirect", "insufficient"}:
            raise ValueError(f"invalid query relevance: {self.relevance!r}")
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            raise ValueError("query assessment evidence IDs must be unique")
        if any(not item.strip() for item in self.unresolved_needs):
            raise ValueError("query assessment unresolved needs must be nonempty strings")
        unknown = [item for item in self.suggested_tools if item not in VISUAL_TOOL_NAMES]
        if unknown:
            raise ValueError(f"query assessment suggests unknown visual tools: {unknown}")
        if self.relevance == "insufficient" and not self.unresolved_needs:
            raise ValueError("insufficient query evidence must name an unresolved need")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "QueryAssessment":
        evidence = _string_list_or_singleton(
            value.get("evidence_event_ids", []),
            "query assessment evidence_event_ids",
        )
        needs = _string_list_or_singleton(
            value.get("unresolved_needs", []),
            "query assessment unresolved_needs",
        )
        tools = _string_list_or_singleton(
            value.get("suggested_tools", []),
            "query assessment suggested_tools",
        )
        return cls(
            relevance=str(value.get("relevance", "")).strip(),  # type: ignore[arg-type]
            evidence_event_ids=tuple(evidence),
            unresolved_needs=tuple(needs),
            suggested_tools=tuple(tools),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance,
            "evidence_event_ids": list(self.evidence_event_ids),
            "unresolved_needs": list(self.unresolved_needs),
            "suggested_tools": list(self.suggested_tools),
        }


@dataclass(frozen=True)
class ObservationBatch:
    events: tuple[EventRecord, ...]
    query_assessment: QueryAssessment
    observation_gaps: tuple[ObservationGap, ...] = ()

    def __post_init__(self) -> None:
        if not self.events and not self.observation_gaps:
            raise ValueError("observation batch requires events or an explicit observation gap")
        event_ids = {event.event_id for event in self.events}
        if len(event_ids) != len(self.events):
            raise ValueError("observation batch event IDs must be unique")
        unknown = [
            item
            for item in self.query_assessment.evidence_event_ids
            if item not in event_ids
        ]
        if unknown:
            raise ValueError(f"query assessment cites unknown events: {unknown}")
        gap_ids = [gap.gap_id for gap in self.observation_gaps]
        if len(set(gap_ids)) != len(gap_ids):
            raise ValueError("observation batch gap IDs must be unique")

@dataclass(frozen=True)
class VideoSketch:
    video_id: str
    duration_seconds: float
    clip_scan_stride_seconds: float
    clip_candidate_count: int
    cache_hit: bool
    observed_spans: tuple[tuple[float, float], ...]
    events: tuple[EventRecord, ...]
    report_batches: int
    selected_frames: int
    selection_mode: str
    frame_manifest: tuple[SelectedFrameTrace, ...]
    report_text: str
    query_assessments: tuple[QueryAssessment, ...] = ()
    observation_gaps: tuple[ObservationGap, ...] = ()
    model_clock: Literal["source", "local"] = "source"

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video sketch video_id must be nonempty")
        if (not self.events and not self.observation_gaps) or not self.report_text.strip():
            raise ValueError("video sketch requires observations and report text")
        if self.selected_frames != len(self.frame_manifest):
            raise ValueError("video sketch selected frame count does not match manifest")
        if self.model_clock not in ("source", "local"):
            raise ValueError("video sketch model_clock must be source or local")
        event_ids = {event.event_id for event in self.events}
        for assessment in self.query_assessments:
            unknown = [item for item in assessment.evidence_event_ids if item not in event_ids]
            if unknown:
                raise ValueError(f"video sketch assessment cites unknown events: {unknown}")


@dataclass(frozen=True)
class WarmStartResult:
    question: str
    options: tuple[str, ...]
    query_plan: QueryPlan
    videos: tuple[VideoSketch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "options": list(self.options),
            "query_plan": {
                "phrases": [
                    {
                        "text": item.text,
                        "source": item.source,
                        "weight": item.weight,
                        "target_video_ids": list(item.target_video_ids),
                    }
                    for item in self.query_plan.phrases
                ],
                "report_hints": list(self.query_plan.report_hints),
            },
            "videos": [
                {
                    "video_id": item.video_id,
                    "model_clock": item.model_clock,
                    "duration_seconds": item.duration_seconds,
                    "clip_scan_stride_seconds": item.clip_scan_stride_seconds,
                    "clip_candidate_count": item.clip_candidate_count,
                    "cache_hit": item.cache_hit,
                    "observed_spans": [list(span) for span in item.observed_spans],
                    "events": [
                        {
                            "event_id": event.event_id,
                            "time_span": [event.start_seconds, event.end_seconds],
                            "description": event.description,
                            "evidence_frame_ids": list(event.evidence_frame_ids),
                            "confidence": event.confidence,
                            "uncertainty": event.uncertainty,
                            "interpretation": event.interpretation,
                        }
                        for event in item.events
                    ],
                    "observation_gaps": [
                        gap.to_mapping() for gap in item.observation_gaps
                    ],
                    "report_batches": item.report_batches,
                    "selected_frames": item.selected_frames,
                    "selection_mode": item.selection_mode,
                    "report_text": item.report_text,
                    "query_assessments": [
                        assessment.to_mapping() for assessment in item.query_assessments
                    ],
                    "frame_manifest": [
                        {
                            "batch_index": frame.batch_index,
                            "video_id": frame.video_id,
                            "segment_id": frame.segment_id,
                            "frame_id": frame.frame_id,
                            "timeline_index": frame.timeline_index,
                            "requested_timestamp_seconds": frame.requested_timestamp_seconds,
                            "actual_timestamp_seconds": frame.actual_timestamp_seconds,
                            "requested_frame_index": frame.requested_frame_index,
                            "actual_frame_index": frame.actual_frame_index,
                            "source_timestamp_seconds": frame.source_timestamp_seconds,
                            "matched_queries": list(frame.matched_queries),
                        }
                        for frame in item.frame_manifest
                    ],
                    "filtered_regions_are_negative_evidence": False,
                }
                for item in self.videos
            ],
        }


# ── initial state formation ──────────────────────────────────────────
#
# The initial state is the reasoner-side working memory built from the
# observer reports.  It deliberately contains no answer field, no completeness
# flag, and no model-reported confidence: the initial formation stage must not
# be able to commit to an answer before any targeted evidence is gathered.


@dataclass(frozen=True)
class VideoEvidenceReport:
    """The only observer output the initial state builder consumes."""

    video_id: str
    events: tuple[EventRecord, ...]
    query_assessments: tuple[QueryAssessment, ...] = ()
    observation_gaps: tuple[ObservationGap, ...] = ()

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("evidence report video_id must be nonempty")
        if not self.events and not self.observation_gaps:
            raise ValueError("evidence report requires events or explicit observation gaps")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("evidence report event IDs must be unique within one video")
        event_ids = self.event_ids
        for assessment in self.query_assessments:
            unknown = [item for item in assessment.evidence_event_ids if item not in event_ids]
            if unknown:
                raise ValueError(f"evidence assessment cites unknown events: {unknown}")

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(event.event_id for event in self.events)

    @classmethod
    def from_video_sketch(cls, sketch: VideoSketch) -> "VideoEvidenceReport":
        return cls(
            video_id=sketch.video_id,
            events=tuple(sketch.events),
            query_assessments=tuple(sketch.query_assessments),
            observation_gaps=tuple(sketch.observation_gaps),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "VideoEvidenceReport":
        raw_events = value.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("evidence report must contain an events list")
        raw_assessments = value.get("query_assessments", [])
        if not isinstance(raw_assessments, list):
            raise ValueError("evidence report query_assessments must be a list")
        raw_gaps = value.get("observation_gaps", [])
        if not isinstance(raw_gaps, list):
            raise ValueError("evidence report observation_gaps must be a list")
        return cls(
            video_id=str(value.get("video_id", "")).strip(),
            events=tuple(EventRecord.from_mapping(item) for item in raw_events),
            query_assessments=tuple(
                QueryAssessment.from_mapping(item) for item in raw_assessments
            ),
            observation_gaps=tuple(
                ObservationGap.from_mapping(item) for item in raw_gaps
            ),
        )


EvidenceStatus = Literal["evidence_complete", "evidence_partial", "unresolved"]


@dataclass(frozen=True)
class ObservedInformation:
    """One observed fact the reasoner judged relevant to the question."""

    information_id: str
    video_ids: tuple[str, ...]
    event_refs: tuple[str, ...]
    statement: str

    def __post_init__(self) -> None:
        if not self.information_id.strip():
            raise ValueError("observed information needs a nonempty id")
        if not self.video_ids:
            raise ValueError("observed information must name at least one video")
        if not self.event_refs:
            raise ValueError("observed information must cite at least one event")
        if not self.statement.strip():
            raise ValueError("observed information statement must be nonempty")


@dataclass(frozen=True)
class WorkingRelation:
    """A proposed cross-video relation and how well the reports cover it.

    ``status`` is never reported by the model and never claims that the
    statement is true.  It is derived by the module from the structure of
    ``supporting_events`` and only records evidence coverage: whether every
    video in ``scope`` actually contributed a cited event.  A relation with
    complete evidence coverage can still be a wrong reading of the videos.
    """

    relation_id: str
    scope: tuple[str, ...]
    statement: str
    supporting_events: tuple[str, ...]
    status: EvidenceStatus

    def __post_init__(self) -> None:
        if not self.relation_id.strip():
            raise ValueError("working relation needs a nonempty id")
        if len(self.scope) < 2:
            raise ValueError("working relation must compare at least two videos")
        if not self.statement.strip():
            raise ValueError("working relation statement must be nonempty")
        if self.status not in ("evidence_complete", "evidence_partial", "unresolved"):
            raise ValueError(f"unknown relation evidence status: {self.status}")


@dataclass(frozen=True)
class UnresolvedIssue:
    """A gap that stays unresolved after the first pass over the reports.

    The issue must be phrased as a property of the evidence, never as a doubt
    about a preferred answer.
    """

    issue_id: str
    scope: tuple[str, ...]
    issue: str
    reason: str
    related_events: tuple[str, ...] = ()
    filled_by_module: bool = False

    def __post_init__(self) -> None:
        if not self.issue_id.strip():
            raise ValueError("unresolved issue needs a nonempty id")
        if not self.scope:
            raise ValueError("unresolved issue must cover at least one video")
        if not self.issue.strip():
            raise ValueError("unresolved issue statement must be nonempty")
        if not self.reason.strip():
            raise ValueError("unresolved issue reason must be nonempty")


@dataclass(frozen=True)
class CandidateRequest:
    """A proposed but unexecuted evidence request, always tied to an issue."""

    request_id: str
    target_video_ids: tuple[str, ...]
    observable: str
    purpose: str
    related_issue: str

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("candidate request needs a nonempty id")
        if not self.target_video_ids:
            raise ValueError("candidate request must target at least one video")
        if not self.observable.strip() or not self.purpose.strip():
            raise ValueError("candidate request needs an observable and a purpose")
        if not self.related_issue.strip():
            raise ValueError("candidate request must reference an unresolved issue")


@dataclass(frozen=True)
class InitialState:
    """Reasoner-side working memory formed from the observer reports."""

    question: str
    video_ids: tuple[str, ...]
    observed_information: tuple[ObservedInformation, ...]
    working_relations: tuple[WorkingRelation, ...]
    unresolved_issues: tuple[UnresolvedIssue, ...]
    candidate_requests: tuple[CandidateRequest, ...]
    required_pairs: tuple[tuple[str, str], ...]
    coverage_gaps: tuple[tuple[str, str], ...]
    validation_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("initial state question must be nonempty")
        if len(self.video_ids) < 2:
            raise ValueError("initial state requires at least two videos")
        if len(set(self.video_ids)) != len(self.video_ids):
            raise ValueError("initial state video IDs must be unique")
        known = set(self.video_ids)
        for information in self.observed_information:
            if not set(information.video_ids) <= known:
                raise ValueError("observed information references an unknown video")
        for relation in self.working_relations:
            if not set(relation.scope) <= known:
                raise ValueError("working relation references an unknown video")
        for issue in self.unresolved_issues:
            if not set(issue.scope) <= known:
                raise ValueError("unresolved issue references an unknown video")
        issue_ids = {issue.issue_id for issue in self.unresolved_issues}
        for request in self.candidate_requests:
            if request.related_issue not in issue_ids:
                raise ValueError("candidate request references an unknown issue")
        required = {ordered_pair(*pair) for pair in self.required_pairs}
        missing = required - self.filled_pairs()
        if missing:
            raise AssertionError(f"required video pairs left uncovered: {sorted(missing)}")
        stray = {ordered_pair(*pair) for pair in self.coverage_gaps} - required
        if stray:
            raise AssertionError(f"coverage gaps report non-required pairs: {sorted(stray)}")

    def covered_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for scope in (
            *(item.scope for item in self.working_relations),
            *(item.scope for item in self.unresolved_issues),
        ):
            for left_index, left in enumerate(scope):
                for right in scope[left_index + 1 :]:
                    pairs.add(ordered_pair(left, right))
        return pairs

    def filled_pairs(self) -> set[tuple[str, str]]:
        return {ordered_pair(*pair) for pair in self.required_pairs} & self.covered_pairs()

    def unresolved_pair_count(self) -> int:
        pairs: set[tuple[str, str]] = set()
        for issue in self.unresolved_issues:
            for left_index, left in enumerate(issue.scope):
                for right in issue.scope[left_index + 1 :]:
                    pairs.add(ordered_pair(left, right))
        return len(pairs & {ordered_pair(*pair) for pair in self.required_pairs})

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "video_ids": list(self.video_ids),
            "observed_information": [
                {
                    "id": item.information_id,
                    "videos": list(item.video_ids),
                    "event_refs": list(item.event_refs),
                    "statement": item.statement,
                }
                for item in self.observed_information
            ],
            "working_relations": [
                {
                    "id": item.relation_id,
                    "scope": list(item.scope),
                    "statement": item.statement,
                    "supporting_events": list(item.supporting_events),
                    "status": item.status,
                    "status_source": "derived_by_module",
                }
                for item in self.working_relations
            ],
            "unresolved_issues": [
                {
                    "id": item.issue_id,
                    "scope": list(item.scope),
                    "issue": item.issue,
                    "reason": item.reason,
                    "related_events": list(item.related_events),
                    "filled_by_module": item.filled_by_module,
                }
                for item in self.unresolved_issues
            ],
            "candidate_requests": [
                {
                    "id": item.request_id,
                    "target_videos": list(item.target_video_ids),
                    "observable": item.observable,
                    "purpose": item.purpose,
                    "related_issue": item.related_issue,
                }
                for item in self.candidate_requests
            ],
            "coverage": {
                "required_pair_count": len(self.required_pairs),
                "covered_pair_count": len(self.filled_pairs()),
                "module_filled_pair_count": len(self.coverage_gaps),
                "unresolved_pair_count": self.unresolved_pair_count(),
                "required_pairs": [list(pair) for pair in self.required_pairs],
                "module_filled_pairs": [list(pair) for pair in self.coverage_gaps],
            },
            "validation_notes": list(self.validation_notes),
            "contains_answer": False,
            "contains_completion_decision": False,
            "model_reported_confidence": False,
        }


def ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


# ── versioned working state ──────────────────────────────────────────
#
# Everything below belongs to the update layer that sits on top of
# ``InitialState``.  State edits are expressed as auditable deltas; the state
# itself is non-monotonic while the revision history is append-only.
#
# Three invariants are deliberately encoded in the *shapes* of these
# dataclasses rather than checked after the fact, so that a caller who tries to
# violate them cannot even build the object:
#
#   1. ``ResolveIssue.resolving_relation`` is required, so a resolution must
#      name the relation that replaced the issue.
#   2. ``AddRelation`` / ``UpdateRelation`` have no ``status`` parameter, so a
#      status can never be supplied by a model or a Review delta.
#   3. ``UpdateRelation.scope`` is accepted only when it equals the current
#      scope -- see ``apply_state_delta``.


@dataclass(frozen=True)
class AddInformation:
    """ADD: append one observed fact."""

    information_id: str
    video_ids: tuple[str, ...]
    event_refs: tuple[str, ...]
    statement: str


@dataclass(frozen=True)
class AddRelation:
    """ADD: introduce a relation.  Its status is derived, never supplied."""

    relation_id: str
    scope: tuple[str, ...]
    statement: str
    supporting_events: tuple[str, ...]


@dataclass(frozen=True)
class AddIssue:
    """ADD: introduce an unresolved issue."""

    issue_id: str
    scope: tuple[str, ...]
    issue: str
    reason: str
    related_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpdateRelation:
    """UPDATE: revise a relation's statement and/or its cited evidence.

    ``scope`` and ``status`` exist only so that a delta carrying them is
    rejected loudly.  Passing a *changed* scope or any status at all raises.
    """

    relation_id: str
    statement: str | None = None
    supporting_events: tuple[str, ...] | None = None
    scope: tuple[str, ...] | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        # A scope or status is never a legitimate edit, but it must reach
        # ``apply_state_delta`` to be rejected with a useful message, so it also
        # counts as "this update does something".
        if all(
            value is None
            for value in (self.statement, self.supporting_events, self.scope, self.status)
        ):
            raise ValueError("relation update must change the statement or its evidence")


@dataclass(frozen=True)
class RetractRelation:
    """RETRACT: the cited evidence shows this relation is wrong."""

    relation_id: str
    reason: str


@dataclass(frozen=True)
class RetractIssue:
    """RETRACT: the issue itself was malformed and is withdrawn."""

    issue_id: str
    reason: str


@dataclass(frozen=True)
class ReopenIssue:
    """REOPEN: new evidence undermines an earlier resolution."""

    issue_id: str
    reason: str


@dataclass(frozen=True)
class ResolveIssue:
    """RESOLVE: an issue is closed by a named relation.

    ``resolving_relation`` is required because closing an issue removes one
    cover of its video pairs; the named relation is what takes over.
    """

    issue_id: str
    resolving_relation: str
    reason: str


# ``Union`` rather than ``X | Y``: this alias is evaluated at import time and
# the package still supports Python 3.9.
StateOperation = Union[
    AddInformation,
    AddRelation,
    AddIssue,
    UpdateRelation,
    RetractRelation,
    RetractIssue,
    ReopenIssue,
    ResolveIssue,
]


@dataclass(frozen=True)
class StateDelta:
    """One Review round's worth of edits, applied atomically."""

    round_id: int
    operations: tuple[StateOperation, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if self.round_id <= 0:
            raise ValueError("state delta round_id must be positive")
        if not self.operations:
            raise ValueError("state delta must contain at least one operation")


@dataclass(frozen=True)
class StateRevision:
    """An append-only record of one state edit."""

    round_id: int
    operation: str
    target_kind: str
    target_id: str
    previous_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    evidence_refs: tuple[str, ...]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "operation": self.operation,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "evidence_refs": list(self.evidence_refs),
            "note": self.note,
        }


@dataclass(frozen=True)
class WorkingState:
    """``InitialState`` plus its version, its audit trail, and its held-out ids.

    ``event_videos`` (event id -> owning video id) is carried along so that the
    update layer can validate citations and re-derive statuses without the
    original observer reports.  That is what makes a run replayable from
    ``WorkingState_0`` plus the delta sequence alone.
    """

    state: InitialState
    round_id: int
    event_videos: tuple[tuple[str, str], ...]
    resolved_issues: tuple[UnresolvedIssue, ...] = ()
    revision_history: tuple[StateRevision, ...] = ()
    module_filled_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.round_id < 0:
            raise ValueError("working state round_id must be nonnegative")
        required = {ordered_pair(*pair) for pair in self.state.required_pairs}
        stray = {ordered_pair(*pair) for pair in self.module_filled_pairs} - required
        if stray:
            raise ValueError(f"module filled pairs lie outside the required pairs: {sorted(stray)}")
        known = set(self.events_by_id())
        for information in self.state.observed_information:
            _require_known(known, information.event_refs, "observed information")
        for relation in self.state.working_relations:
            _require_known(known, relation.supporting_events, "working relation")
        for issue in (*self.state.unresolved_issues, *self.resolved_issues):
            _require_known(known, issue.related_events, "unresolved issue")
        active = {issue.issue_id for issue in self.state.unresolved_issues}
        held = {issue.issue_id for issue in self.resolved_issues}
        if active & held:
            raise ValueError("an issue cannot be both active and resolved")
        for revision in self.revision_history:
            if revision.round_id > self.round_id:
                raise ValueError("revision history contains a future round")

    # The state fields are surfaced directly so callers do not have to reach
    # through ``.state`` for the common reads.

    @property
    def question(self) -> str:
        return self.state.question

    @property
    def video_ids(self) -> tuple[str, ...]:
        return self.state.video_ids

    @property
    def observed_information(self) -> tuple[ObservedInformation, ...]:
        return self.state.observed_information

    @property
    def working_relations(self) -> tuple[WorkingRelation, ...]:
        return self.state.working_relations

    @property
    def unresolved_issues(self) -> tuple[UnresolvedIssue, ...]:
        return self.state.unresolved_issues

    @property
    def candidate_requests(self) -> tuple[CandidateRequest, ...]:
        return self.state.candidate_requests

    @property
    def required_pairs(self) -> tuple[tuple[str, str], ...]:
        return self.state.required_pairs

    @property
    def coverage_gaps(self) -> tuple[tuple[str, str], ...]:
        return self.state.coverage_gaps

    @property
    def validation_notes(self) -> tuple[str, ...]:
        return self.state.validation_notes

    def events_by_id(self) -> dict[str, str]:
        return dict(self.event_videos)

    def history_for(self, target_id: str) -> tuple[StateRevision, ...]:
        return tuple(item for item in self.revision_history if item.target_id == target_id)

    def version_of(self, target_id: str) -> int:
        return len(self.history_for(target_id))

    def pair_coverage_recall(self) -> float:
        """Share of required pairs the builder covered without module repair.

        Cumulative across rounds, so it stays comparable to the round-zero
        figure even after later revisions repair or re-open pairs.
        """
        total = len(self.state.required_pairs)
        if total == 0:
            return 1.0
        filled = {ordered_pair(*pair) for pair in self.module_filled_pairs}
        return 1.0 - len(filled & {ordered_pair(*p) for p in self.state.required_pairs}) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "module_filled_pair_count": len(self.module_filled_pairs),
            "pair_coverage_recall": self.pair_coverage_recall(),
            "resolved_issues": [
                {
                    "id": item.issue_id,
                    "scope": list(item.scope),
                    "issue": item.issue,
                    "reason": item.reason,
                    "related_events": list(item.related_events),
                    "filled_by_module": item.filled_by_module,
                }
                for item in self.resolved_issues
            ],
            "revision_history": [item.to_dict() for item in self.revision_history],
            "working_state": self.state.to_dict(),
            "contains_answer": False,
            "contains_completion_decision": False,
            "model_reported_confidence": False,
        }


def _require_known(known: set[str], refs: tuple[str, ...], label: str) -> None:
    unknown = [item for item in refs if item not in known]
    if unknown:
        raise ValueError(f"{label} cites events absent from the working state: {unknown}")
