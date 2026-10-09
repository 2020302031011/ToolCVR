from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from toolcvr.focus import FocusRequest
from toolcvr.focus_planner import (
    DeterministicMockEvidenceNeed,
    FocusLook,
    FocusPlan,
    FocusPlannerConfig,
    _MEDIUM_TOLERANCE_SECONDS,
    _place_window,
    cited_event_ids,
    classify_issue,
    focus_window,
    observed_event_ids,
    plan_focus,
)
from toolcvr.initial_state import normalize_initial_state
from toolcvr.schemas import (
    EventRecord,
    ObservationGap,
    UnresolvedIssue,
    VideoEvidenceReport,
    WorkingState,
)


@dataclass(frozen=True)
class _ScriptedNeed:
    """A builder stub that returns fixed text, so the guard can be tested."""

    text: str
    calls: int = 0

    def evidence_need(self, **_: Any) -> str:
        return self.text


@dataclass(frozen=True)
class _ExplodingNeed:
    """A builder that must not be consulted: a usable request already exists."""

    def evidence_need(self, **_: Any) -> str:
        raise AssertionError(
            "the builder was consulted even though a candidate request supplied the sentence"
        )


def _event(video_id: str, index: int, start: float, end: float) -> EventRecord:
    return EventRecord(
        event_id=f"{video_id}-e{index}",
        start_seconds=start,
        end_seconds=end,
        description=f"{video_id} action {index}",
        evidence_frame_ids=(f"{video_id}-F{index:02d}",),
        confidence=0.5,
    )


def _report(video_id: str, spans: Sequence[tuple[float, float]]) -> VideoEvidenceReport:
    return VideoEvidenceReport(
        video_id=video_id,
        events=tuple(
            _event(video_id, index, start, end)
            for index, (start, end) in enumerate(spans, start=1)
        ),
    )


def _unobserved_report(video_id: str) -> VideoEvidenceReport:
    """A report with no events at all: the batch ran and produced nothing usable.

    The schema forbids an event-less report, so the honest form of "nothing was
    seen here" is an explicit observation gap.  This is the shape a genuinely
    ``starved`` video arrives in -- as opposed to a video the *state* simply
    never cited, which still has a full event list behind it.
    """

    return VideoEvidenceReport(
        video_id=video_id,
        events=(),
        observation_gaps=(
            ObservationGap(
                gap_id=f"{video_id}-gap1",
                start_seconds=0.0,
                end_seconds=1.0,
                frame_ids=(f"{video_id}-F01",),
                reason="VLM output remained invalid for every attempt in this batch",
            ),
        ),
    )


def _catalogue(reports: Sequence[VideoEvidenceReport]) -> dict[str, EventRecord]:
    return {event.event_id: event for report in reports for event in report.events}


# Long enough that the end of the medium never decides a test that is not about
# it: the synthetic reports below carry no duration of their own.
_MEDIUM = 10_000.0


def _medium_for(state: WorkingState, *, seconds: float = _MEDIUM) -> dict[str, float]:
    """A duration for every video of the sample, observed or not.

    ``InitialState.video_ids`` is the whole list, which is what the planner
    checks its bundle against -- a video can be in scope, never observed, and
    still need a duration on file for the bundle to be plannable at all.
    """
    return {video_id: seconds for video_id in state.state.video_ids}


def _state(
    payload: dict[str, Any],
    reports: Sequence[VideoEvidenceReport],
) -> WorkingState:
    state = normalize_initial_state(
        question="Which video differs?",
        reports=reports,
        payload=payload,
    )
    return WorkingState(
        state=state,
        round_id=0,
        event_videos=tuple(
            (event.event_id, report.video_id)
            for report in reports
            for event in report.events
        ),
        module_filled_pairs=state.coverage_gaps,
    )


def _request(
    request_id: str, targets: Sequence[str], observable: str, issue_id: str
) -> dict[str, Any]:
    return {
        "id": request_id,
        "target_videos": list(targets),
        "observable": observable,
        "purpose": f"to settle {issue_id}",
        "related_issue": issue_id,
    }


def _refusal_for(result: Any, issue_id: str) -> Any:
    matches = [item for item in result.refusals if item.issue_id == issue_id]
    assert matches, (
        f"no refusal recorded for {issue_id}; "
        f"recorded: {[item.issue_id for item in result.refusals]}"
    )
    return matches[0]


def _look(plan: Any, video_id: str) -> Any:
    matches = [item for item in plan.looks if item.video_id == video_id]
    assert matches, (
        f"no look for {video_id}; planned: {[item.video_id for item in plan.looks]}"
    )
    return matches[0]


def _three_video_fixture() -> tuple[WorkingState, dict[str, EventRecord]]:
    """v1 and v2 are cited; v3 is cited nowhere.  One relation covers (v1, v3)."""
    reports = (
        _report("v1", [(10.0, 12.0), (30.0, 34.0)]),
        _report("v2", [(5.0, 6.0)]),
        _report("v3", [(20.0, 21.0)]),
    )
    payload = {
        "observed_information": [
            {
                "id": "I01",
                "videos": ["v1"],
                "event_refs": ["v1-e1", "v1-e2"],
                "statement": "v1 does two things",
            },
            {
                "id": "I02",
                "videos": ["v2"],
                "event_refs": ["v2-e1"],
                "statement": "v2 does one thing",
            },
        ],
        "working_relations": [
            {
                "id": "R01",
                "scope": ["v1", "v3"],
                "statement": "v1 and v3 agree",
                "supporting_events": ["v1-e1"],
            }
        ],
        "unresolved_issues": [
            {"id": "U01", "scope": ["v1", "v2"], "issue": "order unclear", "reason": "conflict"}
        ],
    }
    return _state(payload, reports), _catalogue(reports)


# ── classification ───────────────────────────────────────────────────


def test_classify_separates_uncited_from_uninterpreted() -> None:
    state, _ = _three_video_fixture()
    issues = {issue.issue_id: issue for issue in state.unresolved_issues}
    # Every video U01 names is cited, and v1 sits in a relation.
    assert classify_issue(state, issues["U01"]) == "uninterpreted"

    # v3 is cited nowhere -- but its report *did* observe it, so a window
    # exists and the issue is merely uncited, not starved.
    observed_but_uncited = UnresolvedIssue(
        issue_id="U-auto-01", scope=("v2", "v3"), issue="not established", reason="module"
    )
    assert classify_issue(state, observed_but_uncited) == "uncited"


def test_an_issue_naming_a_never_observed_video_is_starved() -> None:
    """``starved`` has to mean "never looked", not "nothing cited it".

    Naming a video the state cites nowhere is the common case and it is not
    starvation: the VLM looked, the report holds the events, the reasoner just
    never referenced one.  Only a video with nothing in its report at all --
    here v2, whose single batch produced an observation gap and no events --
    leaves no window to inherit.
    """

    reports = (_report("v1", [(1.0, 2.0)]), _unobserved_report("v2"))
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "order unclear", "reason": "r"}
            ],
        },
        reports,
    )

    issue = state.unresolved_issues[0]
    assert classify_issue(state, issue) == "starved"
    assert observed_event_ids(state)["v2"] == set()


def test_cited_event_ids_counts_every_citation_site() -> None:
    state, _ = _three_video_fixture()
    by_video = cited_event_ids(state)
    assert by_video["v1"] == {"v1-e1", "v1-e2"}
    assert by_video["v2"] == {"v2-e1"}
    # v3 has an event, but nothing in the state ever cites it.
    assert by_video["v3"] == set()


def test_observed_event_ids_reads_the_report_not_the_citations() -> None:
    """The two mappings differ exactly for the videos ``uncited`` names."""
    state, _ = _three_video_fixture()
    observed = observed_event_ids(state)
    assert observed["v1"] == {"v1-e1", "v1-e2"}
    assert observed["v2"] == {"v2-e1"}
    # The one video cited nowhere still has an observed event behind it.
    assert observed["v3"] == {"v3-e1"}


# ── mechanical window ────────────────────────────────────────────────


def test_window_is_inherited_from_cited_events_with_margin() -> None:
    state, catalogue = _three_video_fixture()
    window, anchors, clipped = focus_window(
        state, events=catalogue, video_id="v1", duration_seconds=_MEDIUM, config=FocusPlannerConfig(margin_seconds=2.0)
    )
    # v1 cites 10-12 and 30-34, so the union padded by 2 is 8-36.
    assert window == (8.0, 36.0)
    assert anchors == ("v1-e1", "v1-e2")


def test_window_never_starts_before_zero() -> None:
    reports = (_report("v1", [(0.5, 1.0)]), _report("v2", [(0.0, 1.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, _ = focus_window(
        state, events=catalogue, video_id="v1", duration_seconds=_MEDIUM, config=FocusPlannerConfig(margin_seconds=4.0)
    )
    assert window is not None
    assert window[0] == 0.0


def test_window_is_widened_to_the_minimum_width() -> None:
    reports = (_report("v1", [(10.0, 10.1)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, _ = focus_window(
        state,
        events=catalogue,
        video_id="v1", duration_seconds=_MEDIUM,
        config=FocusPlannerConfig(margin_seconds=0.0, minimum_span_seconds=2.0),
    )
    assert window == (10.0, 12.0)


def test_a_single_observation_longer_than_the_budget_widens_the_window() -> None:
    """The cap bounds the anchors, never the window: a window that cannot
    contain its own anchors names evidence the look never frames.

    This test used to assert ``window == (0.0, 30.0)`` for an observation
    spanning 0-500s, i.e. that the anchor fell outside its own window.  The
    assertion was the bug, written down.
    """

    reports = (_report("v1", [(0.0, 500.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, anchors, clipped = focus_window(
        state,
        events=catalogue,
        video_id="v1", duration_seconds=_MEDIUM,
        config=FocusPlannerConfig(margin_seconds=0.0, maximum_span_seconds=30.0),
    )
    assert window == (0.0, 500.0)
    assert anchors == ("v1-e1",)
    assert window[0] <= catalogue["v1-e1"].start_seconds
    assert window[1] >= catalogue["v1-e1"].end_seconds


def test_the_moc_zero_window_contains_its_ninety_two_second_anchor() -> None:
    """The live regression, verbatim.

    MOC-0 cited one event spanning 528-620s.  With the shipped config (margin
    2, cap 60) the old code returned [526.0, 586.0] -- the anchor, and the only
    evidence the look was for, ended 34s outside its own window.
    """

    reports = (_report("v1", [(528.0, 620.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, _ = focus_window(
        state,
        events=catalogue,
        video_id="v1", duration_seconds=_MEDIUM,
        config=FocusPlannerConfig(margin_seconds=2.0, maximum_span_seconds=60.0),
    )
    assert window == (526.0, 622.0)
    assert window[1] - window[0] > 60.0


def test_a_window_whose_padding_exceeds_the_cap_still_contains_its_anchor() -> None:
    """The same defect reached through the margin rather than the event length.

    A cap below twice the margin spends itself on padding alone: with a 4s
    margin and a 3s cap, a one-second observation needs a 9s window.  The old
    code truncated it to 3s and put the anchor outside again.
    """

    reports = (_report("v1", [(10.0, 11.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, _ = focus_window(
        state,
        events=catalogue,
        video_id="v1", duration_seconds=_MEDIUM,
        config=FocusPlannerConfig(
            margin_seconds=4.0, minimum_span_seconds=2.0, maximum_span_seconds=3.0
        ),
    )
    assert window == (6.0, 15.0)


def test_padding_past_the_end_of_the_medium_is_clipped_to_it() -> None:
    """The live CC-474 shape, verbatim.

    The anchor is 110-132s and the clip ends at 133.12s, so a 2s margin asks
    for a window ending at 134s -- a tenth of a second of video that does not
    exist.  ``FocusTool`` clips the same way when it resolves the span, so
    planning it unclipped only guarantees that the plan and its result disagree
    by exactly the margin.
    """

    reports = (_report("v1", [(110.0, 132.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, clipped = focus_window(
        state,
        events=catalogue,
        video_id="v1",
        duration_seconds=133.12,
        config=FocusPlannerConfig(margin_seconds=2.0),
    )
    assert window == (108.0, 133.12)
    assert clipped is True
    # Clipping takes the margin on the far side, never the evidence.
    assert window[0] <= catalogue["v1-e1"].start_seconds
    assert window[1] >= catalogue["v1-e1"].end_seconds


def test_a_window_that_fits_is_not_recorded_as_clipped() -> None:
    reports = (_report("v1", [(110.0, 132.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, clipped = focus_window(
        state,
        events=catalogue,
        video_id="v1",
        duration_seconds=200.0,
        config=FocusPlannerConfig(margin_seconds=2.0),
    )
    assert window == (108.0, 134.0)
    assert clipped is False


def test_the_minimum_width_is_met_from_whichever_side_has_the_room() -> None:
    """A minimum width is padding too, so the medium bounds it like padding.

    The anchor sits at 4.0-4.5s of a 6.0s clip and the minimum asks for 30s.
    The right gives what it has, the left gives the rest, and a clip shorter
    than the minimum yields the whole clip -- what a minimum width is for is
    context around the anchor, and a window is not better for being narrow on
    the side that had room.
    """

    reports = (_report("v1", [(4.0, 4.5)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, clipped = focus_window(
        state,
        events=catalogue,
        video_id="v1",
        duration_seconds=6.0,
        config=FocusPlannerConfig(margin_seconds=0.0, minimum_span_seconds=30.0),
    )
    assert window == (0.0, 6.0)
    assert clipped is True
    assert window[1] >= catalogue["v1-e1"].end_seconds
    # The medium is not a licence to leave the anchor behind.
    assert window[0] <= catalogue["v1-e1"].start_seconds


def test_an_anchor_past_the_end_of_the_medium_fails_loudly() -> None:
    """Clipping never covers for corrupt input: an event that cites frames the
    file does not have is a wiring failure, and shrinking the window around it
    would quietly turn a broken report into a plausible-looking look."""

    reports = (_report("v1", [(10.0, 900.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    try:
        focus_window(
            state,
            events=catalogue,
            video_id="v1",
            duration_seconds=120.0,
            config=FocusPlannerConfig(margin_seconds=2.0),
        )
    except ValueError as exc:
        assert "frames that do not exist" in str(exc)
        assert "v1" in str(exc)
    else:
        raise AssertionError("an anchor past the end of its video was planned anyway")


def test_rounding_at_the_end_of_the_medium_is_not_a_corruption() -> None:
    """A report's duration comes from the container and its event ends from the
    last decoded frame, so they disagree in the last decimals by construction.
    That gap must not be reported as an event citing frames that do not exist.
    """

    reports = (_report("v1", [(10.0, 120.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, _, clipped = focus_window(
        state,
        events=catalogue,
        video_id="v1",
        duration_seconds=120.0 - 0.001,
        config=FocusPlannerConfig(margin_seconds=2.0),
    )
    assert window is not None
    assert clipped is True
    assert window[1] >= catalogue["v1-e1"].end_seconds


def test_the_placement_contract_holds_on_random_media_and_configurations() -> None:
    """The contract of :func:`_place_window`, over the whole numeric space.

    Every window defect found so far was found by one corpus that happened to
    reach it: a 92s anchor with a 60s budget, a padding wider than the clip, a
    clip ending at 133.12s.  Each was repaired where it appeared, which is how
    the third one survived two rounds of repairs -- the repairs covered the
    instances the corpus had and not the clauses they were instances of.

    This checks the clauses.  Durations span four orders of magnitude, anchors
    are allowed to be longer than their medium, padding may exceed the whole
    clip, and the configuration varies across the ranges the defaults live in,
    so the next instance of the class fails here instead of in whichever
    corpus reaches it first.  The seed is fixed: a property test that fails
    only on some runs is one nobody can act on.
    """

    rng = random.Random(20260928)
    for _ in range(20_000):
        duration = rng.choice(
            (
                rng.uniform(0.1, 5.0),
                rng.uniform(5.0, 600.0),
                rng.uniform(600.0, 20_000.0),
            )
        )
        anchor_start = rng.uniform(0.0, duration)
        union = (
            anchor_start,
            anchor_start
            + rng.choice(
                (
                    rng.uniform(0.0, 1.0),
                    rng.uniform(0.0, duration),
                    rng.uniform(0.0, 5.0 * duration),
                )
            ),
        )
        config = FocusPlannerConfig(
            margin_seconds=rng.uniform(0.0, 10.0),
            minimum_span_seconds=rng.uniform(0.01, 60.0),
            maximum_span_seconds=rng.uniform(60.0, 300.0),
        )

        window, clipped = _place_window(
            union[0], union[1], duration_seconds=duration, config=config
        )
        low, high = window
        case = (union, duration, config, window)

        # The two absolute clauses, asserted raw rather than through the
        # re-derivation below, so a misread re-derivation cannot hide a broken
        # window.
        assert low <= union[0] and high >= union[1], case
        assert 0.0 <= low and high <= max(duration, union[1]), case

        # The preference clauses, re-derived from the contract as written.
        padded_start = max(0.0, union[0] - config.margin_seconds)
        wanted_end = max(
            union[1] + config.margin_seconds,
            padded_start + config.minimum_span_seconds,
        )
        expected_high = min(wanted_end, max(duration, union[1]))
        expected_low = (
            max(0.0, expected_high - config.minimum_span_seconds)
            if expected_high - padded_start < config.minimum_span_seconds
            else padded_start
        )
        assert window == (expected_low, expected_high), case
        # The flag names the medium as the reason the window is narrower than
        # the configuration asked for, and only the medium.
        assert clipped == (expected_high < wanted_end), case


def test_the_planned_window_contract_holds_on_random_media() -> None:
    """The same contract through the public entry point, varying the medium.

    ``focus_window`` adds two things to the placement: it picks the anchor
    union through ``_select_anchors``, and it refuses a report whose anchors
    reach past the medium.  Both meet at the boundary where an anchor stops
    being inside its own file, so that boundary is where this walks -- and it
    asserts the boundary is the tolerance, not approximately the tolerance.
    """

    reports = (_report("v1", [(110.0, 132.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    rng = random.Random(20260928)
    refused = accepted = 0
    for _ in range(2_000):
        duration = rng.uniform(0.0, 200.0)
        try:
            window, anchors, _ = focus_window(
                state,
                events=catalogue,
                video_id="v1",
                duration_seconds=duration,
                config=FocusPlannerConfig(margin_seconds=2.0),
            )
        except ValueError as exc:
            refused += 1
            # Only the corruption check may refuse here, and only when the
            # evidence really does reach past the file.
            assert "frames that do not exist" in str(exc), (duration, exc)
            assert duration < 132.0 - _MEDIUM_TOLERANCE_SECONDS, duration
            continue
        accepted += 1
        assert duration >= 132.0 - _MEDIUM_TOLERANCE_SECONDS, duration
        assert anchors == ("v1-e1",)
        assert window[0] <= 110.0 and window[1] >= 132.0, window
        assert 0.0 <= window[0] and window[1] <= max(duration, 132.0), window
    assert refused > 0 and accepted > 0, (refused, accepted)


def test_planning_a_video_of_unknown_length_fails_loudly() -> None:
    """A look into a video of unknown length is a claim nobody can check."""

    reports = (_report("v1", [(10.0, 12.0)]), _report("v2", [(1.0, 2.0)]))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "gap", "reason": "r"}
            ],
        },
        reports,
    )
    try:
        plan_focus(
            state,
            events=catalogue,
            video_durations={"v1": _MEDIUM},
            builder=DeterministicMockEvidenceNeed(),
        )
    except ValueError as exc:
        assert "v2" in str(exc)
        assert "no medium duration" in str(exc)
    else:
        raise AssertionError("a bundle was planned without a duration for every video")


def test_the_clip_is_recorded_on_the_look_and_in_the_payload() -> None:
    reports = (_report("v1", [(110.0, 132.0)]), _report("v2", [(1.0, 2.0)]))
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1"], "issue": "gap", "reason": "r"}
            ],
        },
        reports,
    )
    payload = plan_focus(
        state,
        events=_catalogue(reports),
        video_durations={"v1": 133.12, "v2": _MEDIUM},
        builder=DeterministicMockEvidenceNeed(),
    ).to_dict()["plan"]
    assert payload is not None
    look = payload["looks"][0]
    assert look["window_clipped_to_medium"] is True
    assert look["time_window"] == [108.0, 133.12]


def test_the_over_budget_window_is_recorded_on_the_look() -> None:
    """The overrun is data, not an accident: a reader can see and count it.

    Nothing prevents the window from being wider than the configured budget --
    the evidence decides that.  What must not happen is the artifact leaving a
    reader to discover it by comparing two fields.
    """

    reports = (_report("v1", [(528.0, 620.0)]), _report("v2", [(1.0, 2.0)]))
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "gap", "reason": "r"}
            ],
        },
        reports,
    )
    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=_catalogue(reports),
        builder=DeterministicMockEvidenceNeed(),
        config=FocusPlannerConfig(margin_seconds=2.0, maximum_span_seconds=60.0),
    )
    assert result.plan is not None
    over = _look(result.plan, "v1")
    assert over.window_budget_exceeded is True
    assert _look(result.plan, "v2").window_budget_exceeded is False
    payload = result.plan.to_dict()
    by_video = {item["video_id"]: item for item in payload["looks"]}
    assert by_video["v1"]["window_budget_exceeded"] is True
    assert by_video["v2"]["window_budget_exceeded"] is False
    assert by_video["v1"]["time_window"] == [526.0, 622.0]


def test_a_video_cited_nowhere_still_inherits_its_report_window() -> None:
    """v3 is cited nowhere, so the window falls back to what the report saw."""
    state, catalogue = _three_video_fixture()
    window, anchors, clipped = focus_window(
        state, events=catalogue, video_id="v3", duration_seconds=_MEDIUM, config=FocusPlannerConfig(margin_seconds=2.0)
    )
    # v3-e1 spans 20-21 and nothing cites it; the report is still enough.
    assert window == (18.0, 23.0)
    assert anchors == ("v3-e1",)


def test_a_never_observed_video_yields_no_window_at_all() -> None:
    """The fallback is a second source, not an escape hatch into guessing."""
    reports = (_report("v1", [(1.0, 2.0)]), _unobserved_report("v2"))
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ]
        },
        reports,
    )
    window, anchors, clipped = focus_window(state, events=_catalogue(reports), video_id="v2", duration_seconds=_MEDIUM)
    assert window is None
    assert anchors == ()


def test_a_cited_event_missing_from_the_catalogue_is_an_error() -> None:
    state, catalogue = _three_video_fixture()
    del catalogue["v1-e2"]
    try:
        focus_window(state, events=catalogue, video_id="v1", duration_seconds=_MEDIUM)
    except ValueError as exc:
        assert "missing from the event catalogue" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an unknown cited event")


# ── planning ─────────────────────────────────────────────────────────


def test_a_plan_covers_every_video_in_scope_least_cited_first() -> None:
    """One round looks at the whole issue, not at one side of it.

    U01 is over (v1, v2): v1 has two citations and v2 has one, so v2 is the
    side nothing in the state has accounted for yet and is looked at first.
    Both are looked at in the same round, each with its own inherited window --
    a bundle of one look per video, never one request holding both.
    """

    state, catalogue = _three_video_fixture()
    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue,
        builder=DeterministicMockEvidenceNeed(),
        config=FocusPlannerConfig(margin_seconds=2.0),
    )
    assert result.plan is not None
    plan = result.plan
    assert plan.issue_id == "U01"
    assert plan.looked_videos == ("v2", "v1")
    assert plan.bundle_videos == ("v2", "v1")
    assert plan.unanchorable_videos == ()
    assert plan.partial is False

    # v2 cites a single 5-6s event, padded by 2.
    assert plan.looks[0].time_window == (3.0, 8.0)
    assert plan.looks[0].anchor_event_ids == ("v2-e1",)
    # v1 cites 10-12 and 30-34; both fit one budget, so one look holds both.
    assert plan.looks[1].time_window == (8.0, 36.0)
    assert plan.looks[1].anchor_event_ids == ("v1-e1", "v1-e2")
    # Both windows come from events the state itself cites.
    assert plan.looks[0].anchor_source == "state_citation"
    assert plan.looks[1].anchor_source == "state_citation"
    # The one model-authored field is present and is a single sentence.
    assert plan.evidence_need.strip()
    assert result.refusals == ()


def test_no_look_holds_another_videos_anchors() -> None:
    """The planner-level half of "never mix videos in one request"."""
    state, catalogue = _three_video_fixture()
    payload = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue, builder=DeterministicMockEvidenceNeed()
    ).to_dict()["plan"]
    assert payload is not None
    for look in payload["looks"]:
        for event_id in look["anchor_event_ids"]:
            assert event_id.startswith(f"{look['video_id']}-"), (
                f"look for {look['video_id']} cites {event_id}"
            )
        start, end = look["time_window"]
        for event_id in look["anchor_event_ids"]:
            event = catalogue[event_id]
            assert start <= event.start_seconds
            assert end >= event.end_seconds


def test_a_partially_starved_issue_is_planned_for_the_videos_it_can_anchor() -> None:
    """A video nothing has ever observed does not close the issue's other half.

    v2 contributed no events, only a gap saying its batch produced nothing
    usable, so no window can be inherited for it -- and none is invented.  v1
    can be anchored, so the round looks at v1 and records v2 as the reason the
    issue stays open.  Refusing the whole issue would throw away a look that
    was available; planning it silently would read as full coverage.
    """

    reports = (_report("v1", [(1.0, 2.0)]), _unobserved_report("v2"))
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {
                    "id": "U01",
                    "scope": ["v1", "v2"],
                    "issue": "the relevant event is not localized",
                    "reason": "v2 contributed no events of its own",
                }
            ],
        },
        reports,
    )
    assert state.coverage_gaps == ()

    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=DeterministicMockEvidenceNeed())

    assert result.refusals == ()
    assert result.plan is not None
    plan = result.plan
    assert plan.issue_class == "starved"
    assert plan.looked_videos == ("v1",)
    # v2 has no citations, so it heads the priority order -- and keeps its
    # place there even though no look could be planned for it.
    assert plan.bundle_videos == ("v2", "v1")
    assert plan.unanchorable_videos == ("v2",)
    assert plan.partial is True
    assert plan.to_dict()["partial"] is True
    assert all(
        event_id.startswith("v1-")
        for look in plan.looks
        for event_id in look.anchor_event_ids
    )


def test_an_issue_whose_whole_bundle_is_unobserved_is_refused() -> None:
    """The remaining refusal: there is nothing in the issue worth looking at.

    Refusing is right exactly when every video the round would look at has
    nothing to inherit a window from.  The refusal names all of them, so the
    reason is as wide as the fact behind it -- here, the whole bundle.
    """

    reports = (
        _report("v1", [(1.0, 2.0)]),
        _unobserved_report("v2"),
        _unobserved_report("v3"),
    )
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {
                    "id": "U01",
                    "scope": ["v2", "v3"],
                    "issue": "the relevant event is not localized",
                    "reason": "neither video contributed events of its own",
                }
            ],
        },
        reports,
    )

    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=DeterministicMockEvidenceNeed())

    assert result.plan is None
    assert len(result.refusals) == 1
    refusal = result.refusals[0]
    assert refusal.issue_class == "starved"
    assert refusal.missing_videos == ("v2", "v3")
    assert "no observed event" in refusal.reason
    assert result.considered == ("U01",)
    assert result.deferred == ()


def test_an_uncited_issue_is_planned_not_refused() -> None:
    """A video the state never cited is not a blind spot if the report saw it.

    This is the case the old rule got wrong: v3 appears in no citation, so an
    issue naming it looked starved and was refused.  But the report holds
    v3-e1, so a real window exists and there is nothing to refuse.
    """

    reports = (
        _report("v1", [(10.0, 12.0)]),
        _report("v2", [(5.0, 6.0)]),
        _report("v3", [(20.0, 21.0)]),
    )
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {
                    "id": "U01",
                    "scope": ["v1", "v3"],
                    "issue": "whether v3 shows the step",
                    "reason": "the state cites nothing of v3",
                }
            ],
        },
        reports,
    )

    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=_catalogue(reports),
        builder=DeterministicMockEvidenceNeed(),
        config=FocusPlannerConfig(margin_seconds=2.0),
    )

    assert result.refusals == ()
    assert result.plan is not None
    plan = result.plan
    assert plan.issue_class == "uncited"
    assert plan.looked_videos == ("v3", "v1")
    # The window is inherited from a real observation rather than invented,
    # and each look records which layer its own window came from instead of
    # passing the report fallback off as a state citation.  One bundle can
    # hold both kinds, which is why the label is per look.
    v3 = _look(plan, "v3")
    assert v3.anchor_event_ids == ("v3-e1",)
    assert v3.anchor_source == "report_observation"
    assert v3.time_window == (18.0, 23.0)
    assert _look(plan, "v1").anchor_source == "state_citation"
    assert _look(plan, "v1").time_window == (8.0, 14.0)
    by_video = {item["video_id"]: item for item in plan.to_dict()["looks"]}
    assert by_video["v3"]["anchor_source"] == "report_observation"
    assert by_video["v1"]["anchor_source"] == "state_citation"


def test_refusals_record_every_issue_passed_over_on_the_way() -> None:
    """A starved issue sorts first, is refused, and the next one still plans.

    Only an issue whose *whole* bundle is unobserved can still be refused, so
    U01 covers two videos that never produced an event between them.
    """
    reports = (
        _report("v1", [(10.0, 11.0)]),
        _report("v2", [(20.0, 21.0)]),
        _unobserved_report("v3"),
        _unobserved_report("v4"),
    )
    catalogue = _catalogue(reports)
    payload = {
        "observed_information": [
            {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"},
            {"id": "I02", "videos": ["v2"], "event_refs": ["v2-e1"], "statement": "y"},
        ],
        # Covers (v1, v2) while citing both, so the pair is closed without
        # another issue appearing.
        "working_relations": [
            {"id": "R01", "scope": ["v1", "v2"], "statement": "s", "supporting_events": ["v1-e1"]}
        ],
        "unresolved_issues": [
            # Neither video of U01 has any event at all -> never observed, so
            # starved, so first, and the refusal is the whole bundle.
            {"id": "U01", "scope": ["v3", "v4"], "issue": "gap one", "reason": "r"},
            {"id": "U02", "scope": ["v1", "v2"], "issue": "gap two", "reason": "r"},
        ],
    }
    state = _state(payload, reports)
    assert state.coverage_gaps == (), "fixture must not need a module repair"
    assert state.unresolved_issues, "fixture should leave issues open"

    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=DeterministicMockEvidenceNeed())

    assert result.plan is not None
    assert result.plan.issue_id == "U02"
    assert [item.issue_id for item in result.refusals] == ["U01"]
    assert result.class_of("U01") == "starved"
    assert result.class_of("U02") == "uninterpreted"
    assert result.refusals[0].missing_videos == ("v3", "v4")
    # Nothing sorted behind the winner, so nothing was deferred.
    assert result.deferred == ()


def test_issues_behind_the_winner_are_deferred_rather_than_dropped() -> None:
    """The ledger closes: considered == refusals + deferred + the winner.

    One request per round is the whole scope of this round, so an issue that
    sorts after the winner is never attempted -- neither planned nor refused.
    Recording that is the difference between a plan that deferred work and a
    plan that quietly lost it.
    """

    reports = (
        _report("v1", [(10.0, 12.0), (30.0, 34.0)]),
        _report("v2", [(5.0, 6.0)]),
    )
    catalogue = _catalogue(reports)
    payload = {
        "observed_information": [
            {
                "id": "I01",
                "videos": ["v1"],
                "event_refs": ["v1-e1", "v1-e2"],
                "statement": "x",
            },
            {"id": "I02", "videos": ["v2"], "event_refs": ["v2-e1"], "statement": "y"},
        ],
        "unresolved_issues": [
            {"id": "U01", "scope": ["v1", "v2"], "issue": "gap one", "reason": "r"},
            {"id": "U02", "scope": ["v1", "v2"], "issue": "gap two", "reason": "r"},
            {"id": "U03", "scope": ["v1", "v2"], "issue": "gap three", "reason": "r"},
        ],
    }
    state = _state(payload, reports)

    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=DeterministicMockEvidenceNeed())

    # All three are the same class, so the tie breaks on issue id.
    assert result.considered == ("U01", "U02", "U03")
    assert result.plan is not None
    assert result.plan.issue_id == "U01"
    assert result.refusals == ()
    assert result.deferred == ("U02", "U03")
    assert result.to_dict()["deferred"] == ["U02", "U03"]
    assert len(result.considered) == len(result.refusals) + len(result.deferred) + 1


def test_uncited_outranks_starved_because_it_can_close_its_issue() -> None:
    """A round that can cover its whole issue outranks one that cannot.

    ``uncited`` means every video in scope was observed, so the round can look
    at all of them and the evidence is already in the reports.  ``starved``
    means at least one video has no observation at all, so that round is
    partial by construction.  Both are open here; the closable one wins, and
    the other is deferred rather than refused.
    """

    reports = (
        _report("v1", [(10.0, 11.0)]),
        _report("v2", [(20.0, 21.0)]),
        _unobserved_report("v3"),
        _unobserved_report("v4"),
    )
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v2"], "event_refs": ["v2-e1"], "statement": "y"},
            ],
            "unresolved_issues": [
                # Nothing ever observed either video: partial by construction.
                {"id": "U01", "scope": ["v3", "v4"], "issue": "gap one", "reason": "r"},
                # v1 was observed, the state just never cited it.
                {"id": "U02", "scope": ["v1", "v2"], "issue": "gap two", "reason": "r"},
            ],
        },
        reports,
    )

    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=DeterministicMockEvidenceNeed())

    assert result.class_of("U01") == "starved"
    assert result.class_of("U02") == "uncited"
    assert result.plan is not None
    assert result.plan.issue_id == "U02"
    assert result.refusals == ()
    assert result.deferred == ("U01",)


def test_a_bundle_records_the_videos_it_could_not_anchor() -> None:
    """The look list and the reason it is short are both on the record."""
    reports = (
        _report("v1", [(10.0, 11.0)]),
        _report("v2", [(20.0, 21.0)]),
        _unobserved_report("v3"),
    )
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"},
                {"id": "I02", "videos": ["v2"], "event_refs": ["v2-e1"], "statement": "y"},
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2", "v3"], "issue": "gap", "reason": "r"}
            ],
        },
        reports,
    )

    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=DeterministicMockEvidenceNeed())

    assert result.plan is not None
    plan = result.plan
    # v3 has no citations, so the priority order puts it first -- and it keeps
    # that place in bundle_videos even though no look could be planned for it.
    assert plan.bundle_videos == ("v3", "v1", "v2")
    assert plan.looked_videos == ("v1", "v2")
    assert plan.unanchorable_videos == ("v3",)
    assert plan.partial is True
    assert set(plan.bundle_videos) == set(plan.looked_videos) | set(plan.unanchorable_videos)
    assert result.refusals == ()


def test_the_plan_shape_rejects_a_bundle_that_does_not_add_up() -> None:
    """Constructed plans are checked too: the ledger is not a convention."""
    sentence = "Is a lid already on the pan?"

    try:
        _built_plan((), sentence)
    except ValueError as exc:
        assert "at least one look" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a plan with no looks")

    try:
        _built_plan((_look_for("v1", sentence), _look_for("v1", sentence)), sentence)
    except ValueError as exc:
        assert "one look per video" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a repeated video")

    try:
        _built_plan((_look_for("v1", "Something else entirely?"),), sentence)
    except ValueError as exc:
        assert "single evidence_need" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a divergent sentence")

    try:
        FocusPlan(
            issue_id="U01",
            issue_class="starved",
            evidence_need=sentence,
            looks=(_look_for("v1", sentence),),
            bundle_videos=("v1", "v2"),
            unanchorable_videos=("v1",),
        )
    except ValueError as exc:
        assert "both looked at and unanchorable" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a video in both halves")

    try:
        FocusPlan(
            issue_id="U01",
            issue_class="starved",
            evidence_need=sentence,
            looks=(_look_for("v1", sentence),),
            bundle_videos=("v1", "v2"),
        )
    except ValueError as exc:
        assert "exactly the looks plus the unanchorable" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a bundle that does not add up")


def _built_plan(looks: Sequence[FocusLook], sentence: str) -> FocusPlan:
    return FocusPlan(
        issue_id="U01",
        issue_class="uncited",
        evidence_need=sentence,
        looks=tuple(looks),
    )


def _look_for(video_id: str, sentence: str) -> FocusLook:
    return FocusLook(
        request=FocusRequest(
            video_id=video_id, evidence_need=sentence, time_span=(1.0, 2.0)
        ),
        anchor_event_ids=(f"{video_id}-e1",),
    )


def test_the_models_own_observable_is_used_instead_of_being_regenerated() -> None:
    """The state already holds the sentence the planner would otherwise ask for.

    The live defect this pins: every one of 27 samples carried a validated
    ``candidate_requests[].observable`` and the planner read none of them,
    regenerating a sentence from the issue text instead.  For issues phrased as
    the exam question, that regeneration was the exam question restated.

    The builder here raises, so "was the sentence reused" is not a matter of
    comparing strings: any consultation at all fails the test.
    """

    reports = (_report("v1", [(10.0, 12.0)]), _report("v2", [(5.0, 6.0)]))
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "gap", "reason": "r"}
            ],
            "candidate_requests": [
                _request("Q01", ["v2"], "Is a lid already on the pan?", "U01")
            ],
        },
        reports,
    )

    result = plan_focus(state, video_durations=_medium_for(state), events=_catalogue(reports), builder=_ExplodingNeed())

    assert result.plan is not None
    plan = result.plan
    assert plan.evidence_need == "Is a lid already on the pan?"
    # The sentence is one field on the plan and every look's request carries
    # it, because each call is a separate visual-model request and the
    # comparison has to reach all of them.
    assert plan.looks[0].request.evidence_need == "Is a lid already on the pan?"
    # The request named v2 alone, so the round is v2 alone.
    assert plan.looked_videos == ("v2",)
    assert plan.uncovered_request_targets == ()
    assert plan.evidence_need_source == "candidate_request"
    assert plan.evidence_need_note == ""
    assert plan.to_dict()["evidence_need_source"] == "candidate_request"


def test_the_request_narrows_the_bundle_and_the_citation_count_orders_it() -> None:
    """The request says which videos matter; the citation count only orders.

    v1 is cited nowhere, so the plain least-cited rule would look at v1 even
    though the request narrowed the field to v2 and v3.  The rule still
    applies inside that set -- v3 has one citation to v2's two -- because
    *which of the named videos to look at first* is a priority, and priorities
    are the algorithm's.
    """

    reports = (
        _report("v1", [(10.0, 11.0)]),
        _report("v2", [(20.0, 21.0), (40.0, 41.0)]),
        _report("v3", [(50.0, 51.0)]),
    )
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v2"], "event_refs": ["v2-e1", "v2-e2"], "statement": "x"},
                {"id": "I02", "videos": ["v3"], "event_refs": ["v3-e1"], "statement": "y"},
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2", "v3"], "issue": "gap", "reason": "r"}
            ],
            "candidate_requests": [
                _request("Q01", ["v2", "v3"], "Is the pan covered before stirring?", "U01")
            ],
        },
        reports,
    )

    result = plan_focus(state, video_durations=_medium_for(state), events=_catalogue(reports), builder=_ExplodingNeed())

    assert result.plan is not None
    plan = result.plan
    # v3 before v2: the request said those two videos are what this issue is
    # about, and inside that set the least-cited side is looked at first.
    assert plan.looked_videos == ("v3", "v2")
    # v1 was dropped by the narrowing rather than deferred, and it is not
    # reported as an uncovered target -- the request never named it.
    assert plan.bundle_videos == ("v3", "v2")
    assert "v1" not in plan.bundle_videos
    assert plan.request_target_videos == ("v2", "v3")
    assert plan.uncovered_request_targets == ()
    assert plan.to_dict()["uncovered_request_targets"] == []


def test_a_request_naming_videos_outside_the_issue_scope_is_ignored() -> None:
    """The state validator checks each field, never that the two agree.

    A request is validated against the real video list and the real issue
    list separately, so nothing stops one from pointing at a video this issue
    never mentions.  Following it would aim the plan at the wrong video; the
    sentence is dropped along with the target.
    """

    reports = (
        _report("v1", [(10.0, 11.0)]),
        _report("v2", [(20.0, 21.0)]),
        _report("v3", [(30.0, 31.0)]),
    )
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"},
                {"id": "I02", "videos": ["v2"], "event_refs": ["v2-e1"], "statement": "y"},
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "gap", "reason": "r"}
            ],
            "candidate_requests": [
                _request("Q01", ["v3"], "Does v3 show anything relevant?", "U01")
            ],
        },
        reports,
    )

    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=_catalogue(reports), builder=_ScriptedNeed("Is the lid on the pan?")
    )

    assert result.plan is not None
    plan = result.plan
    # The request was dropped, so the bundle is the whole scope.
    assert plan.looked_videos == ("v1", "v2")
    assert plan.evidence_need_source == "generated"
    assert "no candidate request" in plan.evidence_need_note
    assert plan.request_target_videos == ()


def test_a_request_whose_wording_presupposes_the_answer_falls_back_and_says_so() -> None:
    """A better source is not a licence to skip the check.

    The request's sentence is model output like any other, so it passes the
    same guard.  When it fails, the plan falls back rather than refusing -- but
    the fallback is recorded, so a request that keeps producing unusable
    wording is visible instead of silent.
    """

    reports = (_report("v1", [(10.0, 12.0)]), _report("v2", [(5.0, 6.0)]))
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "gap", "reason": "r"}
            ],
            "candidate_requests": [
                _request("Q01", ["v2"], "Is v2 the different video?", "U01")
            ],
        },
        reports,
    )

    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=_catalogue(reports), builder=_ScriptedNeed("Is a lid already on the pan?")
    )

    assert result.plan is not None
    plan = result.plan
    assert plan.evidence_need == "Is a lid already on the pan?"
    assert plan.evidence_need_source == "generated"
    assert "Q01" in plan.evidence_need_note
    assert "asserts a conclusion" in plan.evidence_need_note


def test_an_issue_with_no_candidate_request_still_gets_a_generated_sentence() -> None:
    """The builder is the fallback, and the fallback stays recorded as such."""
    state, catalogue = _three_video_fixture()
    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue, builder=_ScriptedNeed("Is a lid already on the pan?")
    )

    assert result.plan is not None
    plan = result.plan
    assert plan.evidence_need_source == "generated"
    assert "no candidate request" in plan.evidence_need_note
    assert plan.request_target_videos == ()
    assert plan.uncovered_request_targets == ()


def test_planning_is_deterministic_and_needs_no_network() -> None:
    state, catalogue = _three_video_fixture()
    builder = DeterministicMockEvidenceNeed()
    first = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=builder)
    second = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=builder)
    assert first.to_dict() == second.to_dict()


def test_the_planned_window_matches_the_anchor_event_span() -> None:
    """The window is inherited, not guessed: it is traceable to real events.

    Checked per look, because the invariant is per look: each window has to
    contain the anchors *that* look carries, and no look may carry another
    video's.
    """
    state, catalogue = _three_video_fixture()
    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue, builder=DeterministicMockEvidenceNeed()
    )
    assert result.plan is not None
    for look in result.plan.looks:
        start, end = look.time_window
        assert look.anchor_event_ids
        for event_id in look.anchor_event_ids:
            event = catalogue[event_id]
            assert start <= event.start_seconds
            assert end >= event.end_seconds


def test_a_capped_window_narrows_its_anchor_set_instead_of_contradicting_it() -> None:
    """The cap must select anchors, not silently truncate the span.

    The live defect this pins: 20 events spread over 190s were unioned into a
    60s window truncated from the right, so the plan named anchors its own
    window never reached -- 18 of 20 in the worst sample.  A plan that asks for
    a look in one place and cites its evidence in another cannot be checked by
    anyone, which is the whole property this module exists to keep.
    """

    reports = (
        _report("v1", [(10.0, 11.0)]),
        _report("v2", [(float(t), float(t) + 1.0) for t in range(0, 200, 10)]),
    )
    catalogue = _catalogue(reports)
    state = _state(
        {
            "observed_information": [
                {"id": "I01", "videos": ["v1"], "event_refs": ["v1-e1"], "statement": "x"}
            ],
            "unresolved_issues": [
                {"id": "U01", "scope": ["v1", "v2"], "issue": "gap", "reason": "r"}
            ],
        },
        reports,
    )

    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=DeterministicMockEvidenceNeed())

    assert result.plan is not None
    plan = result.plan
    assert plan.looked_videos == ("v2", "v1")
    dense = _look(plan, "v2")
    assert dense.anchor_source == "report_observation"
    start, end = dense.time_window
    assert end - start <= 60.0
    assert dense.window_budget_exceeded is False
    for event_id in dense.anchor_event_ids:
        event = catalogue[event_id]
        assert start <= event.start_seconds
        assert end >= event.end_seconds
    # One window cannot hold 20 observations over 190s.  The ones left out are
    # named, so the plan is not read as "this is all there is to see here".
    assert 0 < len(dense.anchor_event_ids) < 20
    assert len(dense.anchor_event_ids) + len(dense.dropped_anchor_event_ids) == 20
    assert set(dense.anchor_event_ids).isdisjoint(dense.dropped_anchor_event_ids)
    payload = {item["video_id"]: item for item in plan.to_dict()["looks"]}
    assert payload["v2"]["dropped_anchor_event_ids"] == list(
        dense.dropped_anchor_event_ids
    )
    # The other video of the same round is unaffected by the budget: its one
    # event fits, so nothing is dropped and nothing is exceeded.
    spare = _look(plan, "v1")
    assert spare.dropped_anchor_event_ids == ()
    assert spare.window_budget_exceeded is False


# ── the evidence_need guard ──────────────────────────────────────────


def test_a_presupposing_sentence_is_refused() -> None:
    state, catalogue = _three_video_fixture()
    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue,
        builder=_ScriptedNeed("Is v2 the different video in this clip?"),
    )
    assert result.plan is None
    assert "wording guard" in _refusal_for(result, "U01").reason


def test_an_empty_sentence_is_refused() -> None:
    state, catalogue = _three_video_fixture()
    result = plan_focus(state, video_durations=_medium_for(state), events=catalogue, builder=_ScriptedNeed("   "))
    assert result.plan is None
    assert "empty" in _refusal_for(result, "U01").reason


def test_an_observable_sentence_passes_the_guard() -> None:
    state, catalogue = _three_video_fixture()
    result = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue,
        builder=_ScriptedNeed("Is a lid already on the pan before the stirring starts?"),
    )
    assert result.plan is not None
    assert result.plan.evidence_need.startswith("Is a lid")


def test_the_planner_reports_carry_no_answer_field() -> None:
    state, catalogue = _three_video_fixture()
    payload = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue, builder=DeterministicMockEvidenceNeed()
    ).to_dict()
    for forbidden in ("best_answer", "final_answer", "confidence"):
        assert forbidden not in payload
    assert payload["contains_answer"] is False


def test_the_serialized_plan_is_per_look_and_has_no_single_video_fields() -> None:
    """A flat ``target_video`` would be a lie the moment a bundle has two.

    Pinned because both readers of the file were rewritten in the same change
    and a stale field would otherwise survive as a silently-ignored key.
    """
    state, catalogue = _three_video_fixture()
    payload = plan_focus(
        state,
        video_durations=_medium_for(state),
        events=catalogue, builder=DeterministicMockEvidenceNeed()
    ).to_dict()["plan"]
    assert payload is not None
    for gone in ("target_video", "time_window", "anchor_event_ids", "anchor_source", "related_issue"):
        assert gone not in payload, f"{gone} still flat on the plan"
    assert payload["partial"] is False
    assert payload["unanchorable_videos"] == []
    assert [look["video_id"] for look in payload["looks"]] == ["v2", "v1"]
    for look in payload["looks"]:
        assert look["request_kind"] == "time_span"
        assert look["window_budget_exceeded"] is False
        assert len(look["time_window"]) == 2
