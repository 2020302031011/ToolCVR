from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from toolcvr.schemas import (
    BatchFrameRequest,
    CandidateSpan,
    FeatureTimeline,
    RetrievalPhrase,
)


@dataclass(frozen=True)
class SelectionConfig:
    peak_z: float = 1.5
    minimum_peak_prominence: float = 0.4
    peak_radius_samples: int = 5
    context_samples: int = 2
    frames_per_vlm_call: int = 16
    retention_ratio: float | None = None
    semantic_change_distance: float = 0.08
    maximum_semantic_gap_samples: int = 8

    def __post_init__(self) -> None:
        if self.peak_radius_samples <= 0 or not 1 <= self.frames_per_vlm_call <= 16:
            raise ValueError("selection count limits must be positive")
        if self.context_samples < 0:
            raise ValueError("selection context count must be nonnegative")
        if not 0.0 < self.semantic_change_distance <= 2.0:
            raise ValueError("semantic_change_distance must lie in (0, 2]")
        if self.maximum_semantic_gap_samples <= 0:
            raise ValueError("maximum_semantic_gap_samples must be positive")
        if self.retention_ratio is not None and not 0.0 < self.retention_ratio <= 1.0:
            raise ValueError("retention_ratio must lie in (0, 1]")


@dataclass(frozen=True)
class SelectionResult:
    spans: tuple[CandidateSpan, ...]
    batches: tuple[tuple[BatchFrameRequest, ...], ...]
    mode: str
    informative_query_count: int


class TemporalClipSelector:
    def __init__(self, config: SelectionConfig) -> None:
        self.config = config

    def select(
        self,
        timeline: FeatureTimeline,
        *,
        query_embeddings: np.ndarray,
        phrases: Sequence[RetrievalPhrase],
    ) -> SelectionResult:
        embeddings = _normalize_rows(timeline.embeddings)
        if query_embeddings.ndim != 2:
            raise ValueError("query_embeddings must be a matrix")
        if len(query_embeddings) != len(phrases):
            raise ValueError("query embeddings and phrases must align")
        if query_embeddings.shape[0] and query_embeddings.shape[1] != embeddings.shape[1]:
            raise ValueError("query and frame embeddings use different dimensions")

        similarities = np.empty((len(embeddings), 0), dtype=np.float32)
        if phrases:
            similarities = embeddings @ _normalize_rows(query_embeddings).T

        if phrases and self.config.retention_ratio is not None:
            raw_spans, informative_query_count = self._make_ranked_ratio_spans(
                similarities,
                timeline=timeline,
                phrases=phrases,
                retention_ratio=self.config.retention_ratio,
            )
            mode = "query_ranked_ratio"
        else:
            raw_spans, mode, informative_query_count = self._select_peaks_or_fallback(
                embeddings=embeddings,
                similarities=similarities,
                timeline=timeline,
                phrases=phrases,
            )

        # Timeline positions can be adjacent while belonging to disjoint source
        # intervals (for example two PSS clips cut from one long recording).
        # Segment identity is therefore derived from provenance, never merely
        # from adjacency in the model-visible local clock.
        raw_spans = self._split_source_discontinuities(raw_spans, timeline)
        chosen = self._order_spans(raw_spans)
        batches = self._make_batches(chosen, timeline)
        if not chosen or not batches:
            raise RuntimeError("selector produced no visual evidence")
        return SelectionResult(
            spans=tuple(chosen),
            batches=tuple(tuple(batch) for batch in batches),
            mode=mode,
            informative_query_count=informative_query_count,
        )

    @staticmethod
    def _split_source_discontinuities(
        spans: Sequence[CandidateSpan],
        timeline: FeatureTimeline,
    ) -> list[CandidateSpan]:
        """Split selected runs at every physical source-span boundary.

        A logical clip may concatenate multiple source intervals for compact
        model-facing time.  Those intervals remain separate observations:
        no CandidateSpan, event, or motion claim may bridge their hidden gap.
        """
        source_spans = timeline.source_span_indices
        assert source_spans is not None
        split: list[CandidateSpan] = []
        for span in spans:
            runs: list[list[int]] = []
            for timeline_index in span.timeline_indices:
                if not runs:
                    runs.append([timeline_index])
                    continue
                previous = runs[-1][-1]
                same_source_span = int(source_spans[previous]) == int(
                    source_spans[timeline_index]
                )
                if timeline_index == previous + 1 and same_source_span:
                    runs[-1].append(timeline_index)
                else:
                    runs.append([timeline_index])
            for run in runs:
                split.append(
                    CandidateSpan(
                        segment_id=span.segment_id,
                        timeline_indices=tuple(run),
                        start_seconds=float(timeline.timestamps[run[0]]),
                        end_seconds=float(timeline.timestamps[run[-1]]),
                        score=span.score,
                        matched_queries=span.matched_queries,
                    )
                )
        return split

    def _select_peaks_or_fallback(
        self,
        *,
        embeddings: np.ndarray,
        similarities: np.ndarray,
        timeline: FeatureTimeline,
        phrases: Sequence[RetrievalPhrase],
    ) -> tuple[list[CandidateSpan], str, int]:
        curves = np.empty((len(embeddings), 0), dtype=np.float32)
        if phrases:
            curves = np.column_stack(
                [
                    _smooth_three(_robust_zscore(similarities[:, index])) * phrase.weight
                    for index, phrase in enumerate(phrases)
                ]
            ).astype(np.float32)

        proposals: list[tuple[int, int, float, int]] = []
        for phrase_index in range(curves.shape[1]):
            curve = curves[:, phrase_index]
            for peak_index in _find_local_peaks(
                curve,
                minimum_height=self.config.peak_z,
                minimum_prominence=self.config.minimum_peak_prominence,
                radius=self.config.peak_radius_samples,
            ):
                proposals.append(
                    (
                        max(0, peak_index - self.config.context_samples),
                        min(len(embeddings) - 1, peak_index + self.config.context_samples),
                        float(curve[peak_index]),
                        phrase_index,
                    )
                )

        if proposals:
            raw_spans = self._make_peak_spans(
                proposals,
                timeline=timeline,
                phrases=phrases,
            )
            mode = "query_peak_union"
            informative_query_count = len({proposal[3] for proposal in proposals})
        else:
            # An absent or uninformative text query must not turn a sparse
            # scan into an expensive request for every frame.  CLIP has
            # already produced a semantic timeline, so preserve visible
            # changes plus a bounded temporal coverage gap.  The number of
            # retained frames is content driven; there is deliberately no
            # per-video frame ceiling.
            raw_spans = self._make_semantic_coverage_spans(
                embeddings=embeddings,
                timeline=timeline,
            )
            mode = "semantic_novelty_fallback"
            informative_query_count = 0
        return raw_spans, mode, informative_query_count

    def _make_semantic_coverage_spans(
        self,
        *,
        embeddings: np.ndarray,
        timeline: FeatureTimeline,
    ) -> list[CandidateSpan]:
        size = len(embeddings)
        if size == 1:
            selected = [0]
        else:
            selected = [0]
            last_selected = 0
            for index in range(1, size - 1):
                distance_from_last = 1.0 - float(
                    np.dot(embeddings[last_selected], embeddings[index])
                )
                distance_from_previous = 1.0 - float(
                    np.dot(embeddings[index - 1], embeddings[index])
                )
                semantic_change = max(distance_from_last, distance_from_previous)
                coverage_due = (
                    index - last_selected >= self.config.maximum_semantic_gap_samples
                )
                if semantic_change >= self.config.semantic_change_distance or coverage_due:
                    selected.append(index)
                    last_selected = index
            if selected[-1] != size - 1:
                selected.append(size - 1)

        runs: list[list[int]] = []
        for index in selected:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        return [
            CandidateSpan(
                segment_id=f"raw-{position}",
                timeline_indices=tuple(run),
                start_seconds=float(timeline.timestamps[run[0]]),
                end_seconds=float(timeline.timestamps[run[-1]]),
                score=max(
                    0.0,
                    max(
                        (
                            1.0 - float(np.dot(embeddings[index - 1], embeddings[index]))
                            for index in run
                            if index > 0
                        ),
                        default=0.0,
                    ),
                ),
                matched_queries=(),
            )
            for position, run in enumerate(runs, start=1)
        ]

    @staticmethod
    def _make_ranked_ratio_spans(
        similarities: np.ndarray,
        *,
        timeline: FeatureTimeline,
        phrases: Sequence[RetrievalPhrase],
        retention_ratio: float,
    ) -> tuple[list[CandidateSpan], int]:
        """Retain a fixed high-recall fraction ranked by raw CLIP cosine.

        Every video uses the same ratio and the same score: the strongest raw
        cosine across all applicable original/rewrite phrases. No task label,
        z-score, peak threshold, or per-video frame ceiling participates.
        """
        if similarities.shape != (len(timeline.timestamps), len(phrases)):
            raise ValueError("ranked similarities do not align with timeline and phrases")
        weights = np.asarray([phrase.weight for phrase in phrases], dtype=np.float32)
        weighted = similarities * weights[None, :]
        winning_phrases = np.argmax(weighted, axis=1)
        frame_scores = np.max(weighted, axis=1)
        retain_count = max(1, math.ceil(len(frame_scores) * retention_ratio))
        # Stable tie-breaking by timeline position keeps repeated runs deterministic.
        ranked = np.argsort(-frame_scores, kind="stable")[:retain_count]
        selected = sorted(int(index) for index in ranked)

        runs: list[list[int]] = []
        for index in selected:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)

        spans: list[CandidateSpan] = []
        contributing_phrases: set[int] = set()
        for position, run in enumerate(runs, start=1):
            phrase_indices = {int(winning_phrases[index]) for index in run}
            contributing_phrases.update(phrase_indices)
            spans.append(
                CandidateSpan(
                    segment_id=f"raw-{position}",
                    timeline_indices=tuple(run),
                    start_seconds=float(timeline.timestamps[run[0]]),
                    end_seconds=float(timeline.timestamps[run[-1]]),
                    score=max(float(frame_scores[index]) for index in run),
                    matched_queries=tuple(
                        phrases[index].text for index in sorted(phrase_indices)
                    ),
                )
            )
        return spans, len(contributing_phrases)

    @staticmethod
    def _make_peak_spans(
        proposals: Sequence[tuple[int, int, float, int]],
        *,
        timeline: FeatureTimeline,
        phrases: Sequence[RetrievalPhrase],
    ) -> list[CandidateSpan]:
        # Merge overlapping context windows only. Merely adjacent activations
        # must never bridge two otherwise distinct moments.
        merged: list[tuple[int, int, float, set[int]]] = []
        for start, end, score, phrase_index in sorted(proposals):
            if merged and start <= merged[-1][1]:
                previous_start, previous_end, previous_score, phrase_indices = merged[-1]
                phrase_indices.add(phrase_index)
                merged[-1] = (
                    previous_start,
                    max(previous_end, end),
                    max(previous_score, score),
                    phrase_indices,
                )
            else:
                merged.append((start, end, score, {phrase_index}))

        spans: list[CandidateSpan] = []
        for position, (start, end, score, phrase_indices) in enumerate(merged, start=1):
            spans.append(
                CandidateSpan(
                    segment_id=f"raw-{position}",
                    timeline_indices=tuple(range(start, end + 1)),
                    start_seconds=float(timeline.timestamps[start]),
                    end_seconds=float(timeline.timestamps[end]),
                    score=score,
                    matched_queries=tuple(
                        phrases[index].text for index in sorted(phrase_indices)
                    ),
                )
            )
        return spans

    @staticmethod
    def _order_spans(spans: list[CandidateSpan]) -> list[CandidateSpan]:
        ordered: list[CandidateSpan] = []
        for position, span in enumerate(sorted(spans, key=lambda item: item.start_seconds), start=1):
            ordered.append(
                CandidateSpan(
                    segment_id=f"S{position}",
                    timeline_indices=span.timeline_indices,
                    start_seconds=span.start_seconds,
                    end_seconds=span.end_seconds,
                    score=span.score,
                    matched_queries=span.matched_queries,
                )
            )
        return ordered

    def _make_batches(
        self,
        spans: list[CandidateSpan],
        timeline: FeatureTimeline,
    ) -> list[list[BatchFrameRequest]]:
        limit = self.config.frames_per_vlm_call
        ordered = [
            BatchFrameRequest(
                segment_id=span.segment_id,
                timeline_index=index,
                requested_timestamp_seconds=float(timeline.timestamps[index]),
                source_timestamp_seconds=float(timeline.source_timestamps[index]),
                requested_frame_index=int(timeline.frame_indices[index]),
                matched_queries=span.matched_queries,
            )
            for span in spans
            for index in span.timeline_indices
        ]
        if any(
            right.requested_timestamp_seconds <= left.requested_timestamp_seconds
            for left, right in zip(ordered[:-1], ordered[1:])
        ):
            raise AssertionError("selected frames must remain in strict temporal order")
        batches = [
            ordered[start : start + limit]
            for start in range(0, len(ordered), limit)
        ]
        if any(len(batch) > limit for batch in batches):
            raise AssertionError("VLM batch exceeds configured frame limit")
        return batches


def synchronize_selection_union(
    timelines: Sequence[FeatureTimeline],
    selections: Sequence[SelectionResult],
    *,
    config: SelectionConfig,
) -> tuple[SelectionResult, ...]:
    """Apply the union of aligned source-frame selections to every view.

    Callers must only group inputs whose source frame indices share one clock.
    Provenance remains view-local; this function only synchronizes which frame
    positions survive selection.
    """
    if len(timelines) != len(selections) or len(timelines) < 2:
        raise ValueError("synchronized selection requires at least two aligned timelines")

    union_frame_indices: set[int] = set()
    queries_by_frame: dict[int, set[str]] = {}
    maximum_score = 0.0
    for selection in selections:
        maximum_score = max(maximum_score, *(span.score for span in selection.spans))
        for batch in selection.batches:
            for request in batch:
                frame_index = request.requested_frame_index
                union_frame_indices.add(frame_index)
                queries_by_frame.setdefault(frame_index, set()).update(request.matched_queries)
    if not union_frame_indices:
        raise RuntimeError("synchronized selection union is empty")

    mode = (
        "synchronized_union_with_fallback"
        if any(selection.mode == "unfiltered_timeline_fallback" for selection in selections)
        else "synchronized_union"
    )
    batch_builder = TemporalClipSelector(config)
    synchronized: list[SelectionResult] = []
    for timeline in timelines:
        positions = [
            position
            for position, frame_index in enumerate(timeline.frame_indices)
            if int(frame_index) in union_frame_indices
        ]
        if not positions:
            raise RuntimeError("synchronized views have no overlapping source frame indices")
        runs: list[list[int]] = []
        source_spans = timeline.source_span_indices
        assert source_spans is not None
        for position in positions:
            previous = runs[-1][-1] if runs else None
            same_source_span = (
                previous is not None
                and int(source_spans[previous]) == int(source_spans[position])
            )
            if previous is None or position != previous + 1 or not same_source_span:
                runs.append([position])
            else:
                runs[-1].append(position)
        spans = [
            CandidateSpan(
                segment_id=f"S{span_position}",
                timeline_indices=tuple(run),
                start_seconds=float(timeline.timestamps[run[0]]),
                end_seconds=float(timeline.timestamps[run[-1]]),
                score=maximum_score,
                matched_queries=tuple(
                    sorted(
                        {
                            query
                            for timeline_position in run
                            for query in queries_by_frame.get(
                                int(timeline.frame_indices[timeline_position]), set()
                            )
                        }
                    )
                ),
            )
            for span_position, run in enumerate(runs, start=1)
        ]
        batches = batch_builder._make_batches(spans, timeline)
        synchronized.append(
            SelectionResult(
                spans=tuple(spans),
                batches=tuple(tuple(batch) for batch in batches),
                mode=mode,
                informative_query_count=len(
                    {query for span in spans for query in span.matched_queries}
                ),
            )
        )
    return tuple(synchronized)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norms, 1e-12)


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    median = float(np.median(value))
    mad = float(np.median(np.abs(value - median)))
    standard = float(np.std(value))
    scale = 1.4826 * mad if mad > 1e-6 else standard
    if scale <= 1e-6:
        return np.zeros_like(value)
    return ((value - median) / scale).astype(np.float32)


def _smooth_three(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, np.asarray([0.25, 0.5, 0.25]), mode="valid").astype(np.float32)


def _find_local_peaks(
    values: np.ndarray,
    *,
    minimum_height: float,
    minimum_prominence: float,
    radius: int,
) -> list[int]:
    candidates: list[int] = []
    for index, value in enumerate(values):
        left_neighbor = values[index - 1] if index else -np.inf
        right_neighbor = values[index + 1] if index + 1 < len(values) else -np.inf
        if value < left_neighbor or value < right_neighbor:
            continue
        if value == left_neighbor == right_neighbor:
            continue

        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        left_floor = float(np.min(values[start : index + 1]))
        right_floor = float(np.min(values[index:end]))
        if index == start:
            left_floor = right_floor
        if index + 1 == end:
            right_floor = left_floor
        prominence = float(value) - max(left_floor, right_floor)
        if value >= minimum_height and prominence >= minimum_prominence:
            candidates.append(index)

    # Nearby maxima describe the same coarse event. Suppress duplicates, but
    # impose no limit on how many distinct peaks a video may retain.
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: float(values[item]), reverse=True):
        if all(abs(index - previous) >= radius for previous in selected):
            selected.append(index)
    return sorted(selected)
