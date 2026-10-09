import numpy as np

from toolcvr.schemas import FeatureTimeline, RetrievalPhrase
from toolcvr.selection import (
    SelectionConfig,
    TemporalClipSelector,
    synchronize_selection_union,
)


def _timeline(embeddings: np.ndarray) -> FeatureTimeline:
    size = len(embeddings)
    return FeatureTimeline(
        duration_seconds=max(2.0, size * 2.0),
        stride_seconds=2.0,
        timestamps=np.arange(size, dtype=np.float64) * 2.0,
        frame_indices=np.arange(size, dtype=np.int64) * 60,
        embeddings=embeddings.astype(np.float32),
    )


def test_query_peak_keeps_a_short_timestamped_context() -> None:
    embeddings = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (12, 1))
    embeddings[4:7] = np.array([1.0, 0.0], dtype=np.float32)
    selector = TemporalClipSelector(
        SelectionConfig(
            peak_z=1.0,
            minimum_peak_prominence=0.5,
        )
    )
    result = selector.select(
        _timeline(embeddings),
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        phrases=[RetrievalPhrase("a visible target action", "question")],
    )
    assert result.mode == "query_peak_union"
    assert result.spans
    assert any(span.start_seconds <= 8.0 and span.end_seconds >= 12.0 for span in result.spans)
    assert all(len(batch) <= 16 for batch in result.batches)


def test_flat_similarity_uses_semantic_coverage_instead_of_every_frame() -> None:
    embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (20, 1))
    result = TemporalClipSelector(SelectionConfig()).select(
        _timeline(embeddings),
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        phrases=[RetrievalPhrase("generic scene", "question")],
    )
    assert result.mode == "semantic_novelty_fallback"
    selected = [item.timeline_index for batch in result.batches for item in batch]
    assert selected == [0, 8, 16, 19]
    assert all(len(batch) <= 16 for batch in result.batches)


def test_ranked_ratio_retains_exact_highest_scoring_fraction() -> None:
    scores = np.linspace(-0.8, 0.9, 10, dtype=np.float32)
    embeddings = np.column_stack([scores, np.sqrt(1.0 - scores**2)])
    result = TemporalClipSelector(SelectionConfig(retention_ratio=0.5)).select(
        _timeline(embeddings),
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        phrases=[RetrievalPhrase("a visible target", "question")],
    )
    selected = [
        request.timeline_index
        for batch in result.batches
        for request in batch
    ]
    assert result.mode == "query_ranked_ratio"
    assert selected == [5, 6, 7, 8, 9]
    assert result.informative_query_count == 1


def test_ranked_ratio_does_not_filter_an_empty_query_plan() -> None:
    embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (10, 1))
    result = TemporalClipSelector(SelectionConfig(retention_ratio=0.5)).select(
        _timeline(embeddings),
        query_embeddings=np.empty((0, 2), dtype=np.float32),
        phrases=[],
    )
    assert result.mode == "semantic_novelty_fallback"
    selected = [item.timeline_index for batch in result.batches for item in batch]
    assert selected == [0, 8, 9]


def test_semantic_fallback_keeps_changes_without_a_total_frame_cap() -> None:
    embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (24, 1))
    embeddings[5:10] = np.array([0.0, 1.0], dtype=np.float32)
    embeddings[16:20] = np.array([-1.0, 0.0], dtype=np.float32)
    result = TemporalClipSelector(
        SelectionConfig(
            semantic_change_distance=0.1,
            maximum_semantic_gap_samples=6,
        )
    ).select(
        _timeline(embeddings),
        query_embeddings=np.empty((0, 2), dtype=np.float32),
        phrases=[],
    )
    selected = [item.timeline_index for batch in result.batches for item in batch]
    assert 0 in selected and 23 in selected
    assert 5 in selected and 10 in selected and 16 in selected and 20 in selected
    assert max(right - left for left, right in zip(selected[:-1], selected[1:])) <= 6


def test_hard_batch_limit_is_never_exceeded() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(80, 8)).astype(np.float32)
    query = embeddings[20:21]
    config = SelectionConfig(
        peak_z=-10.0,
        minimum_peak_prominence=-10.0,
        frames_per_vlm_call=16,
    )
    result = TemporalClipSelector(config).select(
        _timeline(embeddings),
        query_embeddings=query,
        phrases=[RetrievalPhrase("event", "question")],
    )
    assert all(len(batch) <= 16 for batch in result.batches)


def test_discontinuous_frames_are_packed_only_in_temporal_order() -> None:
    scores = np.full(30, 0.05, dtype=np.float32)
    scores[5:8] = 0.95
    scores[20:23] = 0.90
    embeddings = np.column_stack([scores, np.sqrt(1.0 - scores**2)])
    result = TemporalClipSelector(
        SelectionConfig(
            peak_z=1.0,
            minimum_peak_prominence=0.5,
            context_samples=1,
        )
    ).select(
        _timeline(embeddings),
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        phrases=[RetrievalPhrase("a visible action", "question")],
    )
    assert len(result.spans) == 2
    assert len(result.batches) == 1
    assert len({frame.segment_id for frame in result.batches[0]}) == 2
    timestamps = [frame.requested_timestamp_seconds for frame in result.batches[0]]
    assert timestamps == sorted(timestamps)


def test_neighboring_responses_do_not_chain_into_a_long_interval() -> None:
    scores = np.full(40, 0.1, dtype=np.float32)
    scores[5:31] = 0.4
    scores[12] = 0.8
    scores[24] = 0.9
    embeddings = np.column_stack([scores, np.sqrt(1.0 - scores**2)])
    selector = TemporalClipSelector(
        SelectionConfig(
            peak_z=0.8,
            minimum_peak_prominence=0.5,
            peak_radius_samples=5,
            context_samples=2,
        )
    )
    result = selector.select(
        _timeline(embeddings),
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        phrases=[RetrievalPhrase("a visible target action", "question")],
    )
    assert result.mode == "query_peak_union"
    assert len(result.spans) == 2
    assert all(len(span.timeline_indices) == 5 for span in result.spans)
    assert sum(len(batch) for batch in result.batches) == 10


def test_each_rewrite_phrase_contributes_candidates_before_union() -> None:
    size = 30
    first = np.zeros((size, 3), dtype=np.float32)
    first[:, 2] = 1.0
    first[6:9] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    first[21:24] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    selector = TemporalClipSelector(
        SelectionConfig(
            peak_z=1.0,
            minimum_peak_prominence=0.5,
            peak_radius_samples=5,
            context_samples=1,
        )
    )
    result = selector.select(
        _timeline(first),
        query_embeddings=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        phrases=[
            RetrievalPhrase("first visible action", "rewrite"),
            RetrievalPhrase("second visible action", "rewrite"),
        ],
    )
    assert result.mode == "query_peak_union"
    assert len(result.spans) == 2
    assert result.spans[0].matched_queries == ("first visible action",)
    assert result.spans[1].matched_queries == ("second visible action",)


def test_timeline_rejects_duplicate_or_reversed_timestamps() -> None:
    try:
        FeatureTimeline(
            duration_seconds=10.0,
            stride_seconds=2.0,
            timestamps=np.array([0.0, 2.0, 2.0], dtype=np.float64),
            frame_indices=np.array([0, 60, 120], dtype=np.int64),
            embeddings=np.ones((3, 2), dtype=np.float32),
        )
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("duplicate timestamps must be rejected")


def test_synchronized_views_receive_the_union_of_selected_frame_indices() -> None:
    first = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (20, 1))
    second = first.copy()
    first[4:7] = np.array([1.0, 0.0], dtype=np.float32)
    second[13:16] = np.array([1.0, 0.0], dtype=np.float32)
    timelines = (_timeline(first), _timeline(second))
    config = SelectionConfig(
        peak_z=1.0,
        minimum_peak_prominence=0.5,
        context_samples=1,
    )
    selector = TemporalClipSelector(config)
    selections = tuple(
        selector.select(
            timeline,
            query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            phrases=[RetrievalPhrase("a visible synchronized event", "question")],
        )
        for timeline in timelines
    )
    original_sets = [
        {request.requested_frame_index for batch in selection.batches for request in batch}
        for selection in selections
    ]
    assert original_sets[0] != original_sets[1]

    synchronized = synchronize_selection_union(timelines, selections, config=config)
    synchronized_sets = [
        {request.requested_frame_index for batch in selection.batches for request in batch}
        for selection in synchronized
    ]
    assert synchronized_sets[0] == synchronized_sets[1]
    assert synchronized_sets[0] == original_sets[0] | original_sets[1]
    assert all(selection.mode == "synchronized_union" for selection in synchronized)


def test_selection_never_bridges_a_hidden_source_span_boundary() -> None:
    timeline = FeatureTimeline(
        duration_seconds=4.001,
        stride_seconds=1.0,
        timestamps=np.array([0.0, 1.0, 2.0, 2.001, 3.001, 4.001]),
        source_timestamps=np.array([10.0, 11.0, 12.0, 40.0, 41.0, 42.0]),
        source_span_indices=np.array([1, 1, 1, 2, 2, 2]),
        frame_indices=np.array([300, 330, 360, 1200, 1230, 1260]),
        embeddings=np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1)),
    )
    result = TemporalClipSelector(
        SelectionConfig(retention_ratio=1.0),
    ).select(
        timeline,
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        phrases=[RetrievalPhrase("visible event", "question")],
    )
    assert len(result.spans) == 2
    assert result.spans[0].timeline_indices == (0, 1, 2)
    assert result.spans[1].timeline_indices == (3, 4, 5)
    assert result.spans[0].segment_id != result.spans[1].segment_id
    assert [
        request.source_timestamp_seconds
        for batch in result.batches
        for request in batch
    ] == [10.0, 11.0, 12.0, 40.0, 41.0, 42.0]
