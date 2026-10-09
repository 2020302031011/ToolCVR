from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from toolcvr.agents import ObserverAgent
from toolcvr.clip import ClipEncoder
from toolcvr.query import QueryRewriter
from toolcvr.reporting import EventReporter
from toolcvr.schemas import (
    EventRecord,
    FeatureTimeline,
    ObservationGap,
    QueryPlan,
    QueryAssessment,
    ReportFrame,
    SelectedFrameTrace,
    VideoInput,
    VideoSketch,
    WarmStartResult,
)
from toolcvr.selection import (
    SelectionConfig,
    SelectionResult,
    TemporalClipSelector,
    synchronize_selection_union,
)
from toolcvr.video import (
    ClipFeatureCache,
    OpenCVVideoReader,
    TimelinePolicy,
    build_feature_timeline,
)


@dataclass(frozen=True)
class ReportConfig:
    maximum_events_per_batch: int = 8
    maximum_new_tokens: int = 512
    overlay_timestamps: bool = True
    decode_long_edge: int = 512

    def __post_init__(self) -> None:
        if self.maximum_events_per_batch <= 0 or self.maximum_new_tokens <= 0:
            raise ValueError("report generation limits must be positive")
        if self.decode_long_edge <= 0:
            raise ValueError("report decode_long_edge must be positive")


@dataclass(frozen=True)
class WarmStartConfig:
    timeline: TimelinePolicy = field(default_factory=TimelinePolicy)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    cache_directory: Path = Path(".cache/clip")

    @classmethod
    def from_yaml(cls, path: Path) -> "WarmStartConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("warm-start config root must be a mapping")
        timeline_raw = raw.get("timeline", {})
        selection_raw = raw.get("selection", {})
        cache_raw = raw.get("cache", {})
        report_raw = raw.get("report", {})
        if not all(isinstance(item, dict) for item in (timeline_raw, selection_raw, cache_raw, report_raw)):
            raise ValueError("timeline/selection/cache/report config blocks must be mappings")
        allowed_timeline = set(TimelinePolicy.__dataclass_fields__)
        allowed_selection = set(SelectionConfig.__dataclass_fields__)
        unknown_timeline = set(timeline_raw) - allowed_timeline
        unknown_selection = set(selection_raw) - allowed_selection
        if unknown_timeline or unknown_selection:
            raise ValueError(
                f"unknown config keys: timeline={sorted(unknown_timeline)}, "
                f"selection={sorted(unknown_selection)}"
            )
        return cls(
            timeline=TimelinePolicy(**timeline_raw),
            selection=SelectionConfig(**selection_raw),
            report=ReportConfig(**report_raw),
            cache_directory=Path(cache_raw.get("directory", ".cache/clip")),
        )


class WarmStartPipeline:
    def __init__(
        self,
        *,
        config: WarmStartConfig,
        rewriter: QueryRewriter,
        clip_encoder: ClipEncoder,
        reporter: EventReporter,
        reader: OpenCVVideoReader | None = None,
    ) -> None:
        self.config = config
        self.rewriter = rewriter
        self.clip_encoder = clip_encoder
        self.reporter = reporter
        self.observer = ObserverAgent(reporter)
        self.reader = reader or OpenCVVideoReader()
        self.cache = ClipFeatureCache(config.cache_directory)
        self.selector = TemporalClipSelector(config.selection)

    def run(
        self,
        *,
        videos: Sequence[VideoInput],
        question: str,
        options: Sequence[str] = (),
    ) -> WarmStartResult:
        if not videos:
            raise ValueError("warm start requires at least one video")
        if not question.strip():
            raise ValueError("question must be nonempty")
        video_ids = [item.video_id for item in videos]
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("video IDs must be unique")

        plan = self.rewriter.rewrite(question, options)
        all_text_embeddings = self.clip_encoder.encode_texts([item.text for item in plan.phrases])
        if len(all_text_embeddings) != len(plan.phrases):
            raise RuntimeError("CLIP text encoder returned the wrong number of embeddings")

        timelines: list[FeatureTimeline] = []
        cache_hits: list[bool] = []
        selections: list[SelectionResult] = []
        for video in videos:
            timeline, cache_hit = self._timeline(video)
            phrase_positions = [
                index
                for index, phrase in enumerate(plan.phrases)
                if not phrase.target_video_ids or video.video_id in phrase.target_video_ids
            ]
            phrases = [plan.phrases[index] for index in phrase_positions]
            if phrase_positions:
                query_embeddings = all_text_embeddings[phrase_positions]
            else:
                query_embeddings = np.empty((0, timeline.embeddings.shape[1]), dtype=np.float32)
            selection = self.selector.select(
                timeline,
                query_embeddings=query_embeddings,
                phrases=phrases,
            )
            timelines.append(timeline)
            cache_hits.append(cache_hit)
            selections.append(selection)

        synchronization_groups: dict[str, list[int]] = defaultdict(list)
        for position, video in enumerate(videos):
            if video.synchronization_group is not None:
                synchronization_groups[video.synchronization_group].append(position)
        for group, positions in synchronization_groups.items():
            if len(positions) < 2:
                raise ValueError(
                    f"synchronization group {group!r} must contain at least two views"
                )
            synchronized = synchronize_selection_union(
                [timelines[position] for position in positions],
                [selections[position] for position in positions],
                config=self.config.selection,
            )
            for position, selection in zip(positions, synchronized):
                selections[position] = selection

        sketches = [
            self._observe_selection(
                video=video,
                timeline=timeline,
                cache_hit=cache_hit,
                selection=selection,
                question=question,
                report_hints=plan.report_hints,
            )
            for video, timeline, cache_hit, selection in zip(
                videos, timelines, cache_hits, selections
            )
        ]

        return WarmStartResult(
            question=question,
            options=tuple(options),
            query_plan=plan,
            videos=tuple(sketches),
        )

    def _observe_selection(
        self,
        *,
        video: VideoInput,
        timeline: FeatureTimeline,
        cache_hit: bool,
        selection: SelectionResult,
        question: str,
        report_hints: Sequence[str],
    ) -> VideoSketch:
        segment_counters: dict[str, int] = defaultdict(int)
        events: list[EventRecord] = []
        gaps: list[ObservationGap] = []
        assessments: list[QueryAssessment] = []
        frame_manifest: list[SelectedFrameTrace] = []
        selected_frames = 0
        for batch_position, batch in enumerate(selection.batches, start=1):
            requested = [item.source_timestamp_seconds for item in batch]
            expected_indices = [item.requested_frame_index for item in batch]
            decoded = self.reader.extract(
                video.path,
                requested,
                long_edge=self.config.report.decode_long_edge,
                expected_frame_indices=expected_indices,
            )
            if len(decoded) != len(batch):
                raise RuntimeError("decoded VLM frames do not align with selection batch")
            report_frames: list[ReportFrame] = []
            for request, frame in zip(batch, decoded):
                segment_counters[request.segment_id] += 1
                frame_id = f"{request.segment_id}-F{segment_counters[request.segment_id]:02d}"
                report_frames.append(
                    ReportFrame(
                        video_id=video.video_id,
                        segment_id=request.segment_id,
                        frame_id=frame_id,
                        requested_timestamp_seconds=request.requested_timestamp_seconds,
                        actual_timestamp_seconds=request.requested_timestamp_seconds,
                        source_frame_index=frame.frame_index,
                        matched_queries=request.matched_queries,
                        rgb=frame.rgb,
                        source_timestamp_seconds=frame.timestamp_seconds,
                    )
                )
                frame_manifest.append(
                    SelectedFrameTrace(
                        batch_index=batch_position,
                        video_id=video.video_id,
                        segment_id=request.segment_id,
                        frame_id=frame_id,
                        timeline_index=request.timeline_index,
                        requested_timestamp_seconds=request.requested_timestamp_seconds,
                        actual_timestamp_seconds=request.requested_timestamp_seconds,
                        requested_frame_index=request.requested_frame_index,
                        actual_frame_index=frame.frame_index,
                        matched_queries=request.matched_queries,
                        source_timestamp_seconds=frame.timestamp_seconds,
                    )
                )
            observed = self.observer.observe(
                video_id=video.video_id,
                frames=report_frames,
                question=question,
                report_hints=report_hints,
            )
            prefixed_ids: dict[str, str] = {}
            for event in observed.events:
                prefixed_ids[event.event_id] = (
                    f"{video.video_id}-b{batch_position}-{event.event_id}"
                )
                events.append(
                    EventRecord(
                        event_id=prefixed_ids[event.event_id],
                        start_seconds=event.start_seconds,
                        end_seconds=event.end_seconds,
                        description=event.description,
                        evidence_frame_ids=event.evidence_frame_ids,
                        confidence=event.confidence,
                        uncertainty=event.uncertainty,
                        interpretation=event.interpretation,
                    )
                )
            assessments.append(
                QueryAssessment(
                    relevance=observed.query_assessment.relevance,
                    evidence_event_ids=tuple(
                        prefixed_ids[item]
                        for item in observed.query_assessment.evidence_event_ids
                    ),
                    unresolved_needs=observed.query_assessment.unresolved_needs,
                    suggested_tools=observed.query_assessment.suggested_tools,
                )
            )
            gaps.extend(
                ObservationGap(
                    gap_id=f"{video.video_id}-b{batch_position}-{gap.gap_id}",
                    start_seconds=gap.start_seconds,
                    end_seconds=gap.end_seconds,
                    frame_ids=gap.frame_ids,
                    reason=gap.reason,
                )
                for gap in observed.observation_gaps
            )
            selected_frames += len(report_frames)

        events.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.event_id))
        report_text = self.observer.render_report_text(
            video_id=video.video_id,
            events=events,
            selected_frames=selected_frames,
            candidate_frames=len(timeline.timestamps),
            scan_stride_seconds=timeline.stride_seconds,
            selection_mode=selection.mode,
            query_assessments=assessments,
            observation_gaps=gaps,
        )
        actual_spans: list[tuple[float, float]] = []
        for span in selection.spans:
            span_frames = [item for item in frame_manifest if item.segment_id == span.segment_id]
            if not span_frames:
                raise RuntimeError(f"selected segment {span.segment_id} has no decoded frames")
            actual_spans.append(
                (
                    min(item.actual_timestamp_seconds for item in span_frames),
                    max(item.actual_timestamp_seconds for item in span_frames),
                )
            )
        return VideoSketch(
            video_id=video.video_id,
            duration_seconds=timeline.duration_seconds,
            clip_scan_stride_seconds=timeline.stride_seconds,
            clip_candidate_count=len(timeline.timestamps),
            cache_hit=cache_hit,
            observed_spans=tuple(actual_spans),
            events=tuple(events),
            report_batches=len(selection.batches),
            selected_frames=selected_frames,
            selection_mode=selection.mode,
            frame_manifest=tuple(frame_manifest),
            report_text=report_text,
            query_assessments=tuple(assessments),
            observation_gaps=tuple(gaps),
            model_clock=video.model_clock,
        )

    def _timeline(self, video: VideoInput) -> tuple[FeatureTimeline, bool]:
        metadata = self.reader.probe(video.path)
        covered_duration = (
            sum(
                min(end, metadata.duration_seconds) - start
                for start, end in video.time_spans
                if start < metadata.duration_seconds
            )
            if video.time_spans
            else metadata.duration_seconds
        )
        if covered_duration <= 0:
            raise ValueError(f"requested time spans lie outside video: {video.path}")
        stride = video.scan_stride_seconds or self.config.timeline.stride_for(covered_duration)
        cache_path = self.cache.key_path(
            video.path,
            encoder_identity=self.clip_encoder.identity,
            stride_seconds=stride,
            long_edge=self.config.timeline.decode_long_edge,
            time_spans=video.time_spans,
            model_clock=video.model_clock,
        )
        cached = self.cache.load(cache_path)
        if cached is not None:
            return cached, True
        timeline = build_feature_timeline(
            video,
            reader=self.reader,
            encoder=self.clip_encoder,
            policy=self.config.timeline,
            stride_seconds=video.scan_stride_seconds,
        )
        self.cache.save(cache_path, timeline)
        return timeline, False


def config_from_mapping(raw: dict[str, Any]) -> WarmStartConfig:
    """Small programmatic helper used by tests and notebooks."""
    timeline = raw.get("timeline", {})
    selection = raw.get("selection", {})
    if not isinstance(timeline, dict) or not isinstance(selection, dict):
        raise ValueError("timeline and selection config blocks must be mappings")
    return WarmStartConfig(
        timeline=TimelinePolicy(**timeline),
        selection=SelectionConfig(**selection),
        report=ReportConfig(**raw.get("report", {})),
        cache_directory=Path(raw.get("cache_directory", ".cache/clip")),
    )
