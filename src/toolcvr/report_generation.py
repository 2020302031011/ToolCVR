from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from toolcvr.agents import ObserverAgent
from toolcvr.schemas import (
    EventRecord,
    ObservationBatch,
    ObservationGap,
    QueryPlan,
    QueryAssessment,
    ReportFrame,
    RetrievalPhrase,
    SelectedFrameTrace,
    VideoInput,
    VideoSketch,
    WarmStartResult,
)
from toolcvr.video import OpenCVVideoReader


_BATCH_CHECKPOINT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class FrozenVideoSelection:
    video_id: str
    model_clock: str
    duration_seconds: float
    clip_scan_stride_seconds: float
    clip_candidate_count: int
    cache_hit: bool
    observed_spans: tuple[tuple[float, float], ...]
    report_batches: int
    selected_frames: int
    selection_mode: str
    frame_manifest: tuple[SelectedFrameTrace, ...]


@dataclass(frozen=True)
class FrozenSelection:
    source_path: Path
    source_sha256: str
    sample_id: str
    task: str
    question: str
    options: tuple[str, ...]
    query_plan: QueryPlan
    videos: tuple[FrozenVideoSelection, ...]

    def validate_video_inputs(self, videos: Sequence[VideoInput]) -> None:
        """Reject selections whose coordinate contract no longer matches."""
        video_map = {video.video_id: video for video in videos}
        if len(video_map) != len(videos):
            raise ValueError("source video IDs must be unique")
        expected_ids = {video.video_id for video in self.videos}
        if set(video_map) != expected_ids:
            raise ValueError(
                "source videos do not match frozen selection: "
                f"expected={sorted(expected_ids)} actual={sorted(video_map)}"
            )
        for frozen in self.videos:
            required_clock = video_map[frozen.video_id].model_clock
            if frozen.model_clock != required_clock:
                raise ValueError(
                    f"{self.sample_id}/{frozen.video_id} selection uses "
                    f"model_clock={frozen.model_clock!r}, but the current adapter "
                    f"requires {required_clock!r}; rebuild selection to prevent "
                    "source-time leakage"
                )


def load_frozen_selection(path: Path, *, maximum_frames_per_batch: int = 16) -> FrozenSelection:
    """Load and validate a selection-only result without trusting its audit events."""

    source_path = path.expanduser().resolve()
    raw_bytes = source_path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"selection result is not valid JSON: {source_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"selection result root must be an object: {source_path}")
    run = _mapping(raw.get("run"), "run", source_path)
    if run.get("selection_only") is not True:
        raise ValueError(f"result is not marked selection_only: {source_path}")

    question = _nonempty_string(raw.get("question"), "question", source_path)
    options = _string_tuple(raw.get("options"), "options", source_path)
    query_plan = _query_plan(raw.get("query_plan"), source_path)
    raw_videos = raw.get("videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise ValueError(f"selection result videos must be a nonempty list: {source_path}")

    videos = tuple(
        _frozen_video(
            _mapping(value, f"videos[{position}]", source_path),
            source_path=source_path,
            maximum_frames_per_batch=maximum_frames_per_batch,
        )
        for position, value in enumerate(raw_videos)
    )
    video_ids = [video.video_id for video in videos]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError(f"selection result contains duplicate video IDs: {source_path}")
    return FrozenSelection(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        sample_id=_nonempty_string(run.get("sample_id"), "run.sample_id", source_path),
        task=_nonempty_string(run.get("task"), "run.task", source_path).upper(),
        question=question,
        options=options,
        query_plan=query_plan,
        videos=videos,
    )


class SelectedReportGenerator:
    """Generate resumable Observer reports from an immutable frame selection."""

    def __init__(
        self,
        *,
        observer: ObserverAgent,
        reporter_identity: str,
        compatible_reporter_identities: Sequence[str] = (),
        decode_long_edge: int = 512,
        maximum_frames_per_batch: int = 16,
        batch_workers: int = 1,
        reader: OpenCVVideoReader | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not reporter_identity.strip():
            raise ValueError("reporter_identity must be nonempty")
        if decode_long_edge <= 0:
            raise ValueError("decode_long_edge must be positive")
        if not 1 <= maximum_frames_per_batch <= 16:
            raise ValueError("maximum_frames_per_batch must lie in [1, 16]")
        if batch_workers <= 0:
            raise ValueError("batch_workers must be positive")
        self.observer = observer
        self.reporter_identity = reporter_identity
        self.compatible_reporter_identities = tuple(compatible_reporter_identities)
        self.decode_long_edge = decode_long_edge
        self.maximum_frames_per_batch = maximum_frames_per_batch
        self.batch_workers = batch_workers
        self.reader = reader or OpenCVVideoReader()
        self.progress = progress

    def generate(
        self,
        *,
        selection: FrozenSelection,
        videos: Sequence[VideoInput],
        checkpoint_directory: Path,
        resume: bool,
    ) -> WarmStartResult:
        selection.validate_video_inputs(videos)
        video_map = {video.video_id: video for video in videos}
        checkpoint_root = checkpoint_directory.expanduser().resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        # Report batches for one video are observationally independent.  API
        # backends can therefore keep several requests in flight without
        # changing evidence identity or temporal ordering.  Aggregation below
        # still consumes results in batch-index order, so concurrency cannot
        # make the committed report nondeterministic.
        with ThreadPoolExecutor(max_workers=self.batch_workers) as batch_executor:
            with ThreadPoolExecutor(max_workers=len(selection.videos)) as video_executor:
                video_futures = tuple(
                    video_executor.submit(
                        self._generate_video,
                        selection=selection,
                        frozen=frozen,
                        video=video_map[frozen.video_id],
                        checkpoint_root=checkpoint_root,
                        resume=resume,
                        executor=batch_executor,
                    )
                    for frozen in selection.videos
                )
                # Calls may finish in any order; resolving futures in source
                # video order keeps the result deterministic.
                sketches = tuple(future.result() for future in video_futures)
        return WarmStartResult(
            question=selection.question,
            options=selection.options,
            query_plan=selection.query_plan,
            videos=sketches,
        )

    def _generate_video(
        self,
        *,
        selection: FrozenSelection,
        frozen: FrozenVideoSelection,
        video: VideoInput,
        checkpoint_root: Path,
        resume: bool,
        executor: ThreadPoolExecutor,
    ) -> VideoSketch:
        grouped: dict[int, list[SelectedFrameTrace]] = defaultdict(list)
        for trace in frozen.frame_manifest:
            grouped[trace.batch_index].append(trace)
        batch_indices = sorted(grouped)
        if batch_indices != list(range(1, frozen.report_batches + 1)):
            raise ValueError(
                f"{selection.sample_id}/{frozen.video_id} batch indices are not contiguous"
            )

        completed: dict[int, ObservationBatch] = {}
        pending: dict[int, Future[ObservationBatch]] = {}
        for batch_index in batch_indices:
            traces = grouped[batch_index]
            checkpoint_path = (
                checkpoint_root / frozen.video_id / f"batch-{batch_index:04d}.json"
            )
            signature = self._batch_signature(
                selection=selection,
                video_id=frozen.video_id,
                batch_index=batch_index,
                traces=traces,
            )
            if resume and checkpoint_path.is_file():
                compatible_signatures = tuple(
                    self._batch_signature(
                        selection=selection,
                        video_id=frozen.video_id,
                        batch_index=batch_index,
                        traces=traces,
                        reporter_identity=identity,
                    )
                    for identity in self.compatible_reporter_identities
                )
                batch_result, legacy = _load_batch_checkpoint(
                    checkpoint_path,
                    expected_signature=signature,
                    compatible_signatures=compatible_signatures,
                )
                if legacy:
                    batch_result = _localize_legacy_assessment(batch_result)
                    _save_batch_checkpoint(
                        checkpoint_path,
                        signature=signature,
                        selection_sha256=selection.source_sha256,
                        reporter_identity=self.reporter_identity,
                        video_id=frozen.video_id,
                        batch_index=batch_index,
                        traces=traces,
                        result=batch_result,
                    )
                _validate_checkpoint_result(batch_result, traces=traces)
                self._progress(
                    f"[{selection.sample_id}][{frozen.video_id}] "
                    f"batch={batch_index}/{frozen.report_batches} "
                    f"resume{' migrated-v4' if legacy else ''} "
                    f"events={len(batch_result.events)} "
                    f"gaps={len(batch_result.observation_gaps)}"
                )
                completed[batch_index] = batch_result
            else:
                self._progress(
                    f"[{selection.sample_id}][{frozen.video_id}] "
                    f"batch={batch_index}/{frozen.report_batches} frames={len(traces)} start"
                )
                pending[batch_index] = executor.submit(
                    self._observe_and_checkpoint_batch,
                    selection=selection,
                    frozen=frozen,
                    video=video,
                    batch_index=batch_index,
                    traces=tuple(traces),
                    checkpoint_path=checkpoint_path,
                    signature=signature,
                )

        # Resolve in chronological order even though requests finish out of
        # order.  This preserves stable event, gap, and assessment ordering.
        events: list[EventRecord] = []
        gaps: list[ObservationGap] = []
        assessments: list[QueryAssessment] = []
        for batch_index in batch_indices:
            batch_result = completed.get(batch_index)
            if batch_result is None:
                batch_result = pending[batch_index].result()
            events.extend(batch_result.events)
            gaps.extend(batch_result.observation_gaps)
            assessments.append(batch_result.query_assessment)

        event_ids = [event.event_id for event in events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError(f"duplicate aggregated event IDs for {frozen.video_id}")
        report_text = self.observer.render_report_text(
            video_id=frozen.video_id,
            events=events,
            selected_frames=frozen.selected_frames,
            candidate_frames=frozen.clip_candidate_count,
            scan_stride_seconds=frozen.clip_scan_stride_seconds,
            selection_mode=frozen.selection_mode,
            query_assessments=assessments,
            observation_gaps=gaps,
        )
        return VideoSketch(
            video_id=frozen.video_id,
            duration_seconds=frozen.duration_seconds,
            clip_scan_stride_seconds=frozen.clip_scan_stride_seconds,
            clip_candidate_count=frozen.clip_candidate_count,
            cache_hit=frozen.cache_hit,
            observed_spans=frozen.observed_spans,
            events=tuple(events),
            report_batches=frozen.report_batches,
            selected_frames=frozen.selected_frames,
            selection_mode=frozen.selection_mode,
            frame_manifest=frozen.frame_manifest,
            report_text=report_text,
            query_assessments=tuple(assessments),
            observation_gaps=tuple(gaps),
            model_clock=frozen.model_clock,
        )

    def _observe_and_checkpoint_batch(
        self,
        *,
        selection: FrozenSelection,
        frozen: FrozenVideoSelection,
        video: VideoInput,
        batch_index: int,
        traces: Sequence[SelectedFrameTrace],
        checkpoint_path: Path,
        signature: str,
    ) -> ObservationBatch:
        report_frames = self._decode_batch(video=video, traces=traces)
        observed = self.observer.observe(
            video_id=frozen.video_id,
            frames=report_frames,
            question=selection.question,
            report_hints=selection.query_plan.report_hints,
        )
        prefixed_ids = {
            event.event_id: f"{frozen.video_id}-b{batch_index:04d}-{event.event_id}"
            for event in observed.events
        }
        batch_result = ObservationBatch(
            events=tuple(
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
                for event in observed.events
            ),
            observation_gaps=tuple(
                ObservationGap(
                    gap_id=f"{frozen.video_id}-b{batch_index:04d}-{gap.gap_id}",
                    start_seconds=gap.start_seconds,
                    end_seconds=gap.end_seconds,
                    frame_ids=gap.frame_ids,
                    reason=gap.reason,
                )
                for gap in observed.observation_gaps
            ),
            query_assessment=QueryAssessment(
                relevance=observed.query_assessment.relevance,
                evidence_event_ids=tuple(
                    prefixed_ids[item]
                    for item in observed.query_assessment.evidence_event_ids
                ),
                unresolved_needs=observed.query_assessment.unresolved_needs,
                suggested_tools=observed.query_assessment.suggested_tools,
            ),
        )
        _validate_checkpoint_result(batch_result, traces=traces)
        _save_batch_checkpoint(
            checkpoint_path,
            signature=signature,
            selection_sha256=selection.source_sha256,
            reporter_identity=self.reporter_identity,
            video_id=frozen.video_id,
            batch_index=batch_index,
            traces=traces,
            result=batch_result,
        )
        self._progress(
            f"[{selection.sample_id}][{frozen.video_id}] "
            f"batch={batch_index}/{frozen.report_batches} "
            f"done events={len(batch_result.events)} "
            f"gaps={len(batch_result.observation_gaps)}"
        )
        return batch_result

    def _decode_batch(
        self,
        *,
        video: VideoInput,
        traces: Sequence[SelectedFrameTrace],
    ) -> tuple[ReportFrame, ...]:
        if not traces or len(traces) > self.maximum_frames_per_batch:
            raise ValueError(
                f"report batch must contain 1..{self.maximum_frames_per_batch} frames"
            )
        requested = [float(trace.source_timestamp_seconds) for trace in traces]
        expected_indices = [trace.requested_frame_index for trace in traces]
        if any(right <= left for left, right in zip(requested[:-1], requested[1:])):
            raise ValueError(
                f"report batch timestamps are not strictly increasing for {video.video_id}"
            )
        if any(
            right <= left
            for left, right in zip(expected_indices[:-1], expected_indices[1:])
        ):
            raise ValueError(
                f"report batch frame indices are not strictly increasing for {video.video_id}"
            )
        decoded = self.reader.extract(
            video.path,
            requested,
            long_edge=self.decode_long_edge,
            expected_frame_indices=expected_indices,
        )
        if len(decoded) != len(traces):
            raise RuntimeError("decoded report frames do not align with frozen selection")
        return tuple(
            ReportFrame(
                video_id=trace.video_id,
                segment_id=trace.segment_id,
                frame_id=trace.frame_id,
                requested_timestamp_seconds=trace.requested_timestamp_seconds,
                actual_timestamp_seconds=trace.actual_timestamp_seconds,
                source_frame_index=frame.frame_index,
                matched_queries=trace.matched_queries,
                rgb=frame.rgb,
                source_timestamp_seconds=frame.timestamp_seconds,
            )
            for trace, frame in zip(traces, decoded)
        )

    def _batch_signature(
        self,
        *,
        selection: FrozenSelection,
        video_id: str,
        batch_index: int,
        traces: Sequence[SelectedFrameTrace],
        reporter_identity: str | None = None,
    ) -> str:
        identity = {
            "schema_version": _BATCH_CHECKPOINT_SCHEMA_VERSION,
            "selection_sha256": selection.source_sha256,
            "reporter_identity": reporter_identity or self.reporter_identity,
            "decode_long_edge": self.decode_long_edge,
            "video_id": video_id,
            "batch_index": batch_index,
            "frames": [_trace_identity(trace) for trace in traces],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


def _frozen_video(
    raw: dict[str, Any],
    *,
    source_path: Path,
    maximum_frames_per_batch: int,
) -> FrozenVideoSelection:
    video_id = _nonempty_string(raw.get("video_id"), "video_id", source_path)
    model_clock = str(raw.get("model_clock", "source"))
    if model_clock not in ("source", "local"):
        raise ValueError(f"{video_id} has invalid model_clock: {source_path}")
    raw_manifest = raw.get("frame_manifest")
    if not isinstance(raw_manifest, list) or not raw_manifest:
        raise ValueError(f"{video_id} frame_manifest must be a nonempty list: {source_path}")
    manifest = tuple(
        SelectedFrameTrace.from_mapping(_mapping(value, "frame_manifest item", source_path))
        for value in raw_manifest
    )
    if any(trace.video_id != video_id for trace in manifest):
        raise ValueError(f"{video_id} manifest contains a foreign video ID: {source_path}")
    frame_ids = [trace.frame_id for trace in manifest]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"{video_id} manifest frame IDs are not unique: {source_path}")
    selected_frames = int(raw.get("selected_frames", -1))
    if selected_frames != len(manifest):
        raise ValueError(f"{video_id} selected_frames does not match manifest: {source_path}")
    manifest = _rebatch_chronologically(
        manifest,
        maximum_frames_per_batch=maximum_frames_per_batch,
        source_path=source_path,
    )
    report_batches = len({trace.batch_index for trace in manifest})
    observed_spans_raw = raw.get("observed_spans")
    if not isinstance(observed_spans_raw, list):
        raise ValueError(f"{video_id} observed_spans must be a list: {source_path}")
    observed_spans: list[tuple[float, float]] = []
    for span in observed_spans_raw:
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"{video_id} has an invalid observed span: {source_path}")
        start, end = float(span[0]), float(span[1])
        if start < 0 or end < start:
            raise ValueError(f"{video_id} has an invalid observed span: {source_path}")
        observed_spans.append((start, end))
    return FrozenVideoSelection(
        video_id=video_id,
        model_clock=model_clock,
        duration_seconds=float(raw["duration_seconds"]),
        clip_scan_stride_seconds=float(raw["clip_scan_stride_seconds"]),
        clip_candidate_count=int(raw["clip_candidate_count"]),
        cache_hit=bool(raw["cache_hit"]),
        observed_spans=tuple(observed_spans),
        report_batches=report_batches,
        selected_frames=selected_frames,
        selection_mode=_nonempty_string(raw.get("selection_mode"), "selection_mode", source_path),
        frame_manifest=manifest,
    )


def _rebatch_chronologically(
    manifest: Sequence[SelectedFrameTrace],
    *,
    maximum_frames_per_batch: int,
    source_path: Path,
) -> tuple[SelectedFrameTrace, ...]:
    """Pack one video's selected frames by time without changing frame identity."""

    ordered = sorted(
        manifest,
        key=lambda trace: (
            trace.actual_timestamp_seconds,
            trace.actual_frame_index,
            trace.frame_id,
        ),
    )
    if any(
        right.actual_timestamp_seconds <= left.actual_timestamp_seconds
        for left, right in zip(ordered[:-1], ordered[1:])
    ):
        raise ValueError(f"selected frame timestamps are not unique: {source_path}")
    return tuple(
        replace(
            trace,
            batch_index=(position // maximum_frames_per_batch) + 1,
        )
        for position, trace in enumerate(ordered)
    )


def _query_plan(value: object, source_path: Path) -> QueryPlan:
    raw = _mapping(value, "query_plan", source_path)
    raw_phrases = raw.get("phrases")
    raw_hints = raw.get("report_hints")
    if not isinstance(raw_phrases, list) or not isinstance(raw_hints, list):
        raise ValueError(f"query_plan requires phrases and report_hints lists: {source_path}")
    phrases: list[RetrievalPhrase] = []
    for position, value in enumerate(raw_phrases):
        phrase = _mapping(value, f"query_plan.phrases[{position}]", source_path)
        targets = _string_tuple(
            phrase.get("target_video_ids", []),
            f"query_plan.phrases[{position}].target_video_ids",
            source_path,
        )
        phrases.append(
            RetrievalPhrase(
                text=_nonempty_string(phrase.get("text"), "phrase.text", source_path),
                source=_nonempty_string(phrase.get("source"), "phrase.source", source_path),
                weight=float(phrase.get("weight", 1.0)),
                target_video_ids=targets,
            )
        )
    hints = _string_tuple(raw_hints, "query_plan.report_hints", source_path)
    if any(not hint.strip() for hint in hints):
        raise ValueError(f"query plan contains an empty report hint: {source_path}")
    return QueryPlan(phrases=tuple(phrases), report_hints=hints)


def _trace_identity(trace: SelectedFrameTrace) -> dict[str, object]:
    return {
        "batch_index": trace.batch_index,
        "video_id": trace.video_id,
        "segment_id": trace.segment_id,
        "frame_id": trace.frame_id,
        "timeline_index": trace.timeline_index,
        "requested_timestamp_seconds": trace.requested_timestamp_seconds,
        "actual_timestamp_seconds": trace.actual_timestamp_seconds,
        "requested_frame_index": trace.requested_frame_index,
        "actual_frame_index": trace.actual_frame_index,
        "source_timestamp_seconds": trace.source_timestamp_seconds,
        "matched_queries": list(trace.matched_queries),
    }


def _save_batch_checkpoint(
    path: Path,
    *,
    signature: str,
    selection_sha256: str,
    reporter_identity: str,
    video_id: str,
    batch_index: int,
    traces: Sequence[SelectedFrameTrace],
    result: ObservationBatch,
) -> None:
    payload = {
        "schema_version": _BATCH_CHECKPOINT_SCHEMA_VERSION,
        "signature": signature,
        "selection_sha256": selection_sha256,
        "reporter_identity": reporter_identity,
        "video_id": video_id,
        "batch_index": batch_index,
        "frame_manifest": [_trace_identity(trace) for trace in traces],
        "events": [event.to_mapping() for event in result.events],
        "observation_gaps": [gap.to_mapping() for gap in result.observation_gaps],
        "query_assessment": result.query_assessment.to_mapping(),
    }
    _atomic_json(path, payload)


def _load_batch_checkpoint(
    path: Path,
    *,
    expected_signature: str,
    compatible_signatures: Sequence[str] = (),
) -> tuple[ObservationBatch, bool]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"batch checkpoint is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"batch checkpoint root must be an object: {path}")
    if raw.get("schema_version") != _BATCH_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported batch checkpoint schema: {path}")
    actual_signature = raw.get("signature")
    legacy = actual_signature in set(compatible_signatures)
    if actual_signature != expected_signature and not legacy:
        raise ValueError(
            f"batch checkpoint does not match current selection/reporter: {path}"
        )
    raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"batch checkpoint events must be a list: {path}")
    events = tuple(
        EventRecord.from_mapping(_mapping(value, "checkpoint event", path))
        for value in raw_events
    )
    raw_assessment = raw.get("query_assessment")
    if not isinstance(raw_assessment, dict):
        raise ValueError(f"batch checkpoint query_assessment must be an object: {path}")
    raw_gaps = raw.get("observation_gaps", [])
    if not isinstance(raw_gaps, list):
        raise ValueError(f"batch checkpoint observation_gaps must be a list: {path}")
    return (
        ObservationBatch(
            events=events,
            observation_gaps=tuple(
                ObservationGap.from_mapping(_mapping(value, "checkpoint gap", path))
                for value in raw_gaps
            ),
            query_assessment=QueryAssessment.from_mapping(raw_assessment),
        ),
        legacy,
    )


def _localize_legacy_assessment(result: ObservationBatch) -> ObservationBatch:
    """Keep grounded events while discarding old batch-to-global conclusions."""

    assessment = result.query_assessment
    needs = (
        ("This batch has no direct local match to the question.",)
        if assessment.relevance == "insufficient"
        else ()
    )
    return ObservationBatch(
        events=result.events,
        observation_gaps=result.observation_gaps,
        query_assessment=QueryAssessment(
            relevance=assessment.relevance,
            evidence_event_ids=assessment.evidence_event_ids,
            unresolved_needs=needs,
            suggested_tools=assessment.suggested_tools,
        ),
    )


def _validate_checkpoint_result(
    result: ObservationBatch,
    *,
    traces: Sequence[SelectedFrameTrace],
) -> None:
    known = {trace.frame_id: trace for trace in traces}
    event_ids: set[str] = set()
    for event in result.events:
        if event.event_id in event_ids:
            raise ValueError(f"batch checkpoint contains duplicate event ID: {event.event_id}")
        event_ids.add(event.event_id)
        unknown = [frame_id for frame_id in event.evidence_frame_ids if frame_id not in known]
        if unknown:
            raise ValueError(
                f"batch checkpoint event {event.event_id} cites unknown frames: {unknown}"
            )
        evidence_times = [
            known[frame_id].actual_timestamp_seconds
            for frame_id in event.evidence_frame_ids
        ]
        if event.start_seconds > min(evidence_times) + 0.25:
            raise ValueError(
                f"batch checkpoint event {event.event_id} starts after its evidence"
            )
        if event.end_seconds < max(evidence_times) - 0.25:
            raise ValueError(
                f"batch checkpoint event {event.event_id} ends before its evidence"
            )
    for gap in result.observation_gaps:
        unknown = [frame_id for frame_id in gap.frame_ids if frame_id not in known]
        if unknown:
            raise ValueError(
                f"batch checkpoint gap {gap.gap_id} cites unknown frames: {unknown}"
            )
        evidence_times = [known[frame_id].actual_timestamp_seconds for frame_id in gap.frame_ids]
        if gap.start_seconds > min(evidence_times) + 0.25:
            raise ValueError(f"batch checkpoint gap {gap.gap_id} starts after its frames")
        if gap.end_seconds < max(evidence_times) - 0.25:
            raise ValueError(f"batch checkpoint gap {gap.gap_id} ends before its frames")


def atomic_write_result(path: Path, payload: dict[str, object]) -> None:
    _atomic_json(path, payload)


def remove_completed_batch_checkpoints(
    checkpoint_directory: Path,
    *,
    sample_directory: Path,
) -> None:
    """Remove only the exact per-sample batches directory after final commit."""

    sample_root = sample_directory.expanduser().resolve()
    checkpoint_root = Path(
        os.path.abspath(str(checkpoint_directory.expanduser()))
    )
    expected = sample_root / "batches"
    if checkpoint_root != expected or checkpoint_root.is_symlink():
        raise ValueError(
            "refusing to remove checkpoints outside the exact sample batches directory: "
            f"expected={expected} actual={checkpoint_root}"
        )
    if checkpoint_root.is_dir():
        shutil.rmtree(checkpoint_root)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping(value: object, label: str, source_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object: {source_path}")
    return value


def _nonempty_string(value: object, label: str, source_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string: {source_path}")
    return value.strip()


def _string_tuple(value: object, label: str, source_path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string list: {source_path}")
    return tuple(value)
