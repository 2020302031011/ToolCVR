"""Stage 3, minimal version: turn one unresolved issue into one focus bundle.

Scope of this module is deliberately narrow.  It implements the loop

    WorkingState -> pick an unresolved issue -> derive one time window per
    video mechanically -> let the model write one sentence -> a bundle of
    per-video FocusRequests

and nothing else.  Not here, by decision: entailment pruning, the weighted
priority formula, conflict merging, and the stopping rule.  All three are
described in ``INITIAL_STATE_DESIGN.md`` 4.5/4.7 and are deliberately left
unimplemented so that the shape of the first loop can be judged on its own.

A bundle is a *set of videos looked at in one round*, never a request holding
frames from several: ``AGENTS.md`` forbids mixing videos in one visual-model
request, so each look is one video, one window, one call, and the round's
evidence is merged only when the state is updated.  That is also why the one
sentence lives on the plan and not on each look -- it is the question the whole
round is asking, and it is usually a comparison ("X in video 1 against Y in
video 2") that no single look could express on its own.

Three structural facts drive the code below.

1.  ``WorkingState`` records which events were cited, never *when* they
    happened -- ``AddInformation.event_refs`` and
    ``AddRelation.supporting_events`` are id strings and nothing more.  So the
    event catalogue is an explicit input here, not something the state can hand
    over.  That keeps the state layer media-blind, which is the point of it.
    Video durations are an input for the same reason: the window rule needs
    them, and a module that decoded a file to get one would have stopped being
    a planner.

2.  A video can be cited nowhere in the state and still have been observed.
    The report holds its events; the reasoner simply never referenced one of
    them.  The window rule therefore reads the event catalogue, not the
    citations alone: a video with observations gets a real inherited window
    even when the state cites nothing of it, and only a video with neither
    citations nor observations is refused outright.

3.  The one sentence this module needs is usually already written.  4.5 step 5
    says the LLM's entire output in focus planning is ``question``, and the
    state formation stage already asked for exactly that field, under the name
    ``candidate_requests[].observable`` -- bound to its issue by
    ``related_issue`` and checked against the real video list by the state
    validator.  So the builder below is the *fallback*: it runs only when no
    request names the issue, or when the request's own wording fails the
    guard.  Regenerating by default would discard a validated field and route
    the reasoner around the only path into it that has a schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from toolcvr.focus import FocusRequest
from toolcvr.openai_compatible import APIRequestConfig, OpenAICompatibleTransport
from toolcvr.query import parse_json_object
from toolcvr.schemas import (
    CandidateRequest,
    EventRecord,
    UnresolvedIssue,
    WorkingState,
)

# ``uncited`` first: every video in scope was observed -- the state simply
# never referenced any of it -- so the round can look at all of them at once
# and the evidence it needs is already sitting in the reports.  ``starved``
# next: at least one video in scope was never observed at all, so the round can
# only ever be partial, and the video it cannot cover is named in
# ``unanchorable_videos``.  A round that can close its issue outranks one that
# structurally cannot.  This ordering is a rule of thumb, *not* the priority
# formula of 4.5 step 4 -- that one is deferred, and it is a weighted score
# rather than a sort key.
_ISSUE_CLASS_ORDER = ("uncited", "starved", "isolated", "uninterpreted")

# How far past the end of the medium a report's own event may reach before it
# is treated as corrupt rather than rounding.  A report's ``duration_seconds``
# is derived from the container while its event timestamps are decoded frame
# timestamps, so the two disagree in the last decimals by construction; the
# tolerance absorbs that and nothing else.  An event that reaches past it cites
# frames the medium does not have, which is a wiring failure, not a window.
_MEDIUM_TOLERANCE_SECONDS = 0.05

# A shallow lexical net, not a semantic check.  The wording discipline of 4.5
# step 6 lives primarily in the prompt; this only catches the most literal way
# of asserting a conclusion inside the question.
_PRESUPPOSING_PHRASES = (
    "the answer",
    "correct answer",
    "the odd one",
    "which video is different",
    "the different video",
)


@dataclass(frozen=True)
class FocusPlannerConfig:
    """Window padding and the minimum width a focus window may have.

    ``maximum_span_seconds`` is a *selection* budget, not a ceiling on the
    window.  It bounds the unpadded union of the anchors a window may hold, so
    a window is at most this wide plus two margins -- unless a single
    observation is longer than the budget, in which case the window grows to
    contain it and the overrun is recorded on the look rather than paid for by
    truncating.  See :func:`focus_window`.
    """

    margin_seconds: float = 2.0
    minimum_span_seconds: float = 2.0
    maximum_span_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.margin_seconds < 0:
            raise ValueError("focus planner margin must be nonnegative")
        if self.minimum_span_seconds <= 0:
            raise ValueError("focus planner minimum span must be positive")
        if self.maximum_span_seconds < self.minimum_span_seconds:
            raise ValueError("focus planner maximum span is below its minimum")


@dataclass(frozen=True)
class FocusLook:
    """One video, one inherited window: exactly one ``FocusTool`` call.

    Frames from several videos may never share a visual-model request
    (``AGENTS.md``), so a round that needs three videos is three looks.  This
    record is the per-video half of a :class:`FocusPlan`; the sentence the
    round is asking lives on the plan, because it is one question asked of all
    of them.
    """

    request: FocusRequest
    anchor_event_ids: tuple[str, ...]
    # ``state_citation`` when the window came from events the state itself
    # cites; ``report_observation`` when it came from the report because the
    # state cited none of that video's events.  The second is the weaker
    # claim: those events are real observations, but nothing in the reasoning
    # state has said they matter to the question.
    anchor_source: str = "state_citation"
    # Candidates the window does not contain, because the observations for
    # this video do not fit in one budgeted look.  Kept on the record rather
    # than dropped: they are real evidence that this round chose not to cover,
    # and a later round may cover them.
    dropped_anchor_event_ids: tuple[str, ...] = ()
    # True when containing every anchor needed a window wider than
    # ``FocusPlannerConfig.maximum_span_seconds`` -- i.e. one observation is
    # itself longer than the selection budget.  Recorded rather than prevented:
    # the window is the truthful one either way, so what a reader needs is to
    # know that this look was framed by the evidence and not by the config.
    window_budget_exceeded: bool = False
    # True when the padding, not the evidence, ran past the end of the medium
    # and was cut back to it.  The window still contains every anchor; what it
    # lost is the margin on the last side, which is why this is a note and not
    # an error.  The executor clips identically on its own, so a reader
    # comparing the plan's window against the result's ``resolved_time_span``
    # sees agreement here instead of an apparent shortfall.
    window_clipped_to_medium: bool = False

    def __post_init__(self) -> None:
        if not self.anchor_event_ids:
            raise ValueError("a focus look must inherit at least one anchor event")

    @property
    def video_id(self) -> str:
        return self.request.video_id

    @property
    def time_window(self) -> tuple[float, float]:
        if self.request.time_span is not None:
            return self.request.time_span
        if self.request.center_timestamp_seconds is None:
            raise AssertionError("validated focus request has no locator")
        return (
            self.request.center_timestamp_seconds,
            self.request.center_timestamp_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        start, end = self.time_window
        return {
            "video_id": self.video_id,
            "request_kind": "time_span" if self.request.time_span is not None else "timestamp",
            "time_window": [start, end],
            "anchor_event_ids": list(self.anchor_event_ids),
            "anchor_source": self.anchor_source,
            "dropped_anchor_event_ids": list(self.dropped_anchor_event_ids),
            "window_budget_exceeded": self.window_budget_exceeded,
            "window_clipped_to_medium": self.window_clipped_to_medium,
        }


@dataclass(frozen=True)
class FocusPlan:
    """One issue resolved into one round of per-video looks."""

    issue_id: str
    issue_class: str
    # The one sentence, shared by every look.  4.5 step 5 makes it the model's
    # entire output here, and it is bound to the issue rather than to a video:
    # it is usually the comparison itself, so it has to reach every call.
    evidence_need: str
    looks: tuple[FocusLook, ...]
    # Every video this round set out to cover, in the order it would look at
    # them -- least-cited first, so the first look is the video nothing in the
    # state has accounted for yet.  Recorded so the ledger closes:
    # bundle == looks + unanchorable.
    bundle_videos: tuple[str, ...] = ()
    # Videos in the bundle whose report holds no event at all, so there is no
    # window to inherit and nothing to look at.  They are not a refusal: the
    # rest of the bundle is still gathered this round, and the issue stays
    # unresolved.  Named because "this round covered v1 and v3" must not read
    # as "the issue is settled".
    unanchorable_videos: tuple[str, ...] = ()
    # Where the one sentence came from.  ``candidate_request`` means it is the
    # model's own ``observable``, reused verbatim; ``generated`` means the
    # builder wrote a fresh one, and ``evidence_need_note`` says why.
    evidence_need_source: str = "generated"
    evidence_need_note: str = ""
    # Every video the state's candidate request for this issue named, covering
    # or not.  Kept because the request is the model's own statement of where
    # the evidence is, and a round that narrowed it away should be able to say
    # so.
    request_target_videos: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.looks:
            raise ValueError("a focus plan must contain at least one look")
        looked = self.looked_videos
        if len(set(looked)) != len(looked):
            raise ValueError("a focus plan may contain at most one look per video")
        for look in self.looks:
            if look.request.evidence_need != self.evidence_need:
                raise ValueError("every look must carry the plan's single evidence_need")
        if set(looked) & set(self.unanchorable_videos):
            raise ValueError("a video cannot be both looked at and unanchorable")
        if self.bundle_videos and set(self.bundle_videos) != set(looked) | set(
            self.unanchorable_videos
        ):
            raise ValueError(
                "bundle_videos must be exactly the looks plus the unanchorable videos"
            )

    @property
    def looked_videos(self) -> tuple[str, ...]:
        return tuple(look.video_id for look in self.looks)

    @property
    def partial(self) -> bool:
        """True when the round set out to cover a video it could not anchor."""
        return bool(self.unanchorable_videos)

    @property
    def uncovered_request_targets(self) -> tuple[str, ...]:
        """Targets the request named that this round did not even consider.

        A target that was considered and could not be anchored is *not*
        uncovered -- it is in ``unanchorable_videos``, which says why.  What is
        left here is the request naming a video outside the issue scope, so the
        bundle rule dropped it: the state validator checks a request's targets
        against the real video list and its issue against the real issue list,
        but never that the two agree with each other.
        """
        considered = set(self.looked_videos) | set(self.unanchorable_videos)
        return tuple(
            video_id
            for video_id in self.request_target_videos
            if video_id not in considered
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_class": self.issue_class,
            "evidence_need": self.evidence_need,
            "evidence_need_source": self.evidence_need_source,
            "evidence_need_note": self.evidence_need_note,
            "looks": [look.to_dict() for look in self.looks],
            "bundle_videos": list(self.bundle_videos),
            "unanchorable_videos": list(self.unanchorable_videos),
            "partial": self.partial,
            "request_target_videos": list(self.request_target_videos),
            "uncovered_request_targets": list(self.uncovered_request_targets),
        }


@dataclass(frozen=True)
class PlannerRefusal:
    """An issue that was considered and could not be planned, with the reason."""

    issue_id: str
    issue_class: str
    reason: str
    missing_videos: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannerResult:
    """The outcome of one planning call.

    ``plan`` is ``None`` when no issue could be anchored.  ``refusals`` is the
    audit trail for that: every issue that was passed over stays visible
    instead of quietly disappearing, because a planner that can only see half
    the issues must not look like a planner that had nothing to do.
    """

    plan: FocusPlan | None
    refusals: tuple[PlannerRefusal, ...]
    considered: tuple[str, ...]
    # A tuple of pairs rather than a dict: this object is frozen, and the rest
    # of the state layer keeps its mappings immutable too.
    issue_classes: tuple[tuple[str, str], ...] = ()
    # Issues sorted after the winner.  One FocusRequest per call is this
    # round's whole scope, so those were never attempted: neither planned nor
    # refused.  Without this field they would simply vanish from the record.
    # Naming them closes the ledger -- considered == refusals + deferred +
    # the winner.
    deferred: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": None if self.plan is None else self.plan.to_dict(),
            "refusals": [
                {
                    "issue_id": item.issue_id,
                    "issue_class": item.issue_class,
                    "reason": item.reason,
                    "missing_videos": list(item.missing_videos),
                }
                for item in self.refusals
            ],
            "considered": list(self.considered),
            "issue_classes": dict(self.issue_classes),
            "deferred": list(self.deferred),
            "contains_answer": False,
        }

    def class_of(self, issue_id: str) -> str | None:
        for identifier, label in self.issue_classes:
            if identifier == issue_id:
                return label
        return None


class EvidenceNeedBuilder(Protocol):
    """The only model-facing seam in this module."""

    def evidence_need(
        self,
        *,
        question: str,
        issue: UnresolvedIssue,
        target_videos: tuple[str, ...],
        known_events: Sequence[EventRecord],
    ) -> str: ...


# ── mechanical classification ────────────────────────────────────────


def cited_event_ids(state: WorkingState) -> dict[str, set[str]]:
    """Map video id -> the event ids of that video cited anywhere in the state.

    Citations are collected from observed information, working relations and
    unresolved issues, which together are every place the state can reference
    an event.  Counting these is what separates "we looked and found nothing to
    say" from "we never looked".
    """
    video_of = dict(state.event_videos)
    by_video: dict[str, set[str]] = {video_id: set() for video_id in state.video_ids}
    for event_id in _all_cited_event_ids(state):
        video_id = video_of.get(event_id)
        if video_id is not None:
            by_video.setdefault(video_id, set()).add(event_id)
    return by_video


def _all_cited_event_ids(state: WorkingState) -> list[str]:
    cited: list[str] = []
    for information in state.observed_information:
        cited.extend(information.event_refs)
    for relation in state.working_relations:
        cited.extend(relation.supporting_events)
    for issue in state.unresolved_issues:
        cited.extend(issue.related_events)
    return cited


def observed_event_ids(state: WorkingState) -> dict[str, set[str]]:
    """Map video id -> every event its report contributed, cited or not.

    ``cited_event_ids`` answers "did the state use this video"; this one
    answers "did anything ever look at it".  The two differ exactly for the
    videos an apparently starved issue names: the report holds their events,
    the state simply never referenced one of them.  Reading the catalogue
    instead of the citations is what keeps ``starved`` meaning what its name
    says.
    """
    by_video: dict[str, set[str]] = {video_id: set() for video_id in state.video_ids}
    for event_id, video_id in state.event_videos:
        by_video.setdefault(video_id, set()).add(event_id)
    return by_video


def classify_issue(state: WorkingState, issue: UnresolvedIssue) -> str:
    """Label an issue from citation and observation counts.  No model call.

    ``starved``       at least one video in scope was never observed at all:
                      the state cites no event of it and its report has none
    ``uncited``       at least one video in scope has report events, but the
                      state references none of them
    ``isolated``      no video in scope participates in any relation
    ``uninterpreted`` every video in scope is cited, but no relation covers it

    ``starved`` vs ``uncited`` is the distinction that decides how much of the
    issue a round can cover.  Every ``uncited`` video has a window waiting in
    its report, so one round can look at all of them; a ``starved`` issue has
    at least one video with nothing to inherit, so its round is partial by
    construction and says so in ``FocusPlan.unanchorable_videos``.

    The ``conflicting`` class of 4.5 step 1 is not computed here: deciding that
    two relations disagree needs the entailment comparison this round excludes.
    """
    return _classify(cited_event_ids(state), observed_event_ids(state), state, issue)


def _classify(
    by_video: Mapping[str, set[str]],
    observed: Mapping[str, set[str]],
    state: WorkingState,
    issue: UnresolvedIssue,
) -> str:
    in_relation = {video_id for relation in state.working_relations for video_id in relation.scope}
    # Never observed at all: no citation to inherit from and no report event
    # either.  Only this case has to be refused.
    if any(
        not by_video.get(video_id) and not observed.get(video_id)
        for video_id in issue.scope
    ):
        return "starved"
    # Observed but never referenced: a window exists, it just lives in the
    # report rather than in the state.
    if any(not by_video.get(video_id) for video_id in issue.scope):
        return "uncited"
    if not any(video_id in in_relation for video_id in issue.scope):
        return "isolated"
    return "uninterpreted"


# ── mechanical window derivation ─────────────────────────────────────


def requests_for_issue(
    state: WorkingState, issue: UnresolvedIssue
) -> tuple[CandidateRequest, ...]:
    """The candidate requests the state bound to this issue.

    Filtered to those whose targets actually intersect the issue scope.  The
    state validator checks that a request names a real issue and real videos,
    but never that the two agree with each other, so a request pointing at
    videos this issue never mentions is dropped here rather than steering the
    plan toward a video the issue is not about.
    """
    scope = set(issue.scope)
    return tuple(
        request
        for request in state.candidate_requests
        if request.related_issue == issue.issue_id
        and scope.intersection(request.target_video_ids)
    )


def _bundle_videos(
    by_video: Mapping[str, set[str]],
    issue: UnresolvedIssue,
    request: CandidateRequest | None,
) -> tuple[str, ...]:
    """Every video this round should look at, in the order it looks at them.

    The pool is the issue's own scope, narrowed by the videos a candidate
    request named when there is one: the model already said which videos this
    issue is about, so the citation count only orders *inside* that set instead
    of overriding it.  Ordering is least-cited first, ties on scope order --
    the first look is the video nothing in the state has accounted for yet, and
    which video to look at first is a priority, and priorities are the
    algorithm's rather than the request's listing order.

    Unanchorable videos keep their place in the order; the caller simply plans
    no look for them.  Dropping them here would lose the distinction between
    "this round did not need that video" and "that video has nothing to show".
    """
    order = {video_id: index for index, video_id in enumerate(issue.scope)}
    if request is None:
        pool = tuple(issue.scope)
    else:
        named = set(request.target_video_ids)
        pool = tuple(video_id for video_id in issue.scope if video_id in named)
        if not pool:
            pool = tuple(issue.scope)
    return tuple(
        sorted(
            pool,
            key=lambda video_id: (
                len(by_video.get(video_id, ())),
                order.get(video_id, len(order)),
            ),
        )
    )


def anchor_candidates(
    state: WorkingState,
    *,
    video_id: str,
    cited: Mapping[str, set[str]] | None = None,
) -> tuple[str, ...]:
    """Every event id that could anchor a window for ``video_id``.

    Two sources, in decreasing strength:

    1. the events this state already cites for ``video_id`` -- the reasoner
       named them, so they are the strongest anchor available;
    2. every event the report observed for ``video_id``, cited or not -- the
       VLM did look there, the state just never referenced any of it.  This is
       what gives a video cited nowhere a window built from real observations
       instead of a guessed one.

    The second source is only consulted when the first is empty: mixing them
    would let an uncited event outrank a cited one.  An empty result means the
    video was never observed at all, which is the only case worth refusing.
    """
    by_video = cited_event_ids(state) if cited is None else cited
    anchor_ids = sorted(by_video.get(video_id, ()))
    if not anchor_ids:
        anchor_ids = sorted(observed_event_ids(state).get(video_id, ()))
    return tuple(anchor_ids)


def _select_anchors(
    anchor_ids: Sequence[str],
    span_of: Mapping[str, tuple[float, float]],
    config: FocusPlannerConfig,
) -> tuple[str, ...]:
    """The subset of ``anchor_ids`` one window can hold.

    All of them when they fit.  When they do not, the tightest cluster that
    does -- the placement covering the most observations, earliest on a tie.

    Truncating the *span* instead would leave the plan contradicting itself:
    ``anchor_event_ids`` would name events the window does not contain, and
    for a video with a long observation list that is most of them.  The plan
    would then ask for a look somewhere and cite evidence somewhere else.
    """

    # By span, not by id: ``v2-e10`` sorts before ``v2-e2`` lexically, and the
    # sliding window is only meaningful over a time-ordered list.
    ordered = sorted(anchor_ids, key=lambda item: (*span_of[item], item))
    if not ordered:
        return ()
    budget = config.maximum_span_seconds - 2 * config.margin_seconds
    if budget <= 0:
        # The padding alone spends the whole budget; one anchor still fits.
        return (ordered[0],)
    best_start, best_stop = 0, 1
    for low in range(len(ordered)):
        # Grow the run while its *union* fits.  The union is max(end) minus
        # min(start), not last-minus-first: two events of one video may nest,
        # and reading only the endpoints would let an enclosed long event
        # through a check it should have failed.
        end = span_of[ordered[low]][1]
        high = low
        while high + 1 < len(ordered):
            widened = max(end, span_of[ordered[high + 1]][1])
            if widened - span_of[ordered[low]][0] > budget:
                break
            end = widened
            high += 1
        if high - low + 1 > best_stop - best_start:
            best_start, best_stop = low, high + 1
    return tuple(ordered[best_start:best_stop])


def _place_window(
    union_start: float,
    union_end: float,
    *,
    duration_seconds: float,
    config: FocusPlannerConfig,
) -> tuple[tuple[float, float], bool]:
    """The window around an anchor union, and whether the medium shortened it.

    The contract, in the order the clamps are applied:

    1. **Containment.**  The window contains ``[union_start, union_end]``.
    2. **Bound.**  The window lies inside ``[0, duration_seconds]``.
    3. **Configuration.**  Within 1 and 2, the window is what the
       configuration asks for: the anchors padded by ``margin_seconds`` on
       each side, and at least ``minimum_span_seconds`` wide where the medium
       has the room.  The minimum is taken from the right first and what the
       right cannot give from the left, so a clip too short for the minimum
       yields what the clip has rather than a window that is narrow on the
       side with room left in it.

    Every clamp moves one edge toward the union and never past it, so (1) holds
    by construction rather than by repair -- whatever the configuration and the
    medium do, including a padding wider than the whole medium and a minimum
    width longer than it.  2 and 3 can still disagree: where they do, 2 wins
    and the returned flag says so, because the medium is a fact about the file
    and the configuration is a preference about our own budget.

    ``duration_seconds`` may fall a few thousandths short of the union: a
    report measures its duration from the container and its event ends from the
    last decoded frame, so inside ``_MEDIUM_TOLERANCE_SECONDS`` the two are the
    same instant measured two ways and the anchor decides.  Beyond that the
    report cites frames the file does not have, and :func:`focus_window`
    refuses it before calling this.
    """
    ceiling = max(duration_seconds, union_end)
    start = max(0.0, union_start - config.margin_seconds)
    wanted_end = max(
        union_end + config.margin_seconds,
        start + config.minimum_span_seconds,
    )
    end = min(wanted_end, ceiling)
    if end - start < config.minimum_span_seconds:
        start = max(0.0, end - config.minimum_span_seconds)
    return (start, end), end < wanted_end


def focus_window(
    state: WorkingState,
    *,
    events: Mapping[str, EventRecord],
    video_id: str,
    duration_seconds: float,
    config: FocusPlannerConfig = FocusPlannerConfig(),
    cited: Mapping[str, set[str]] | None = None,
) -> tuple[tuple[float, float] | None, tuple[str, ...], bool]:
    """Inherit a window from evidence that exists; never invent one.

    The anchors come from :func:`anchor_candidates`; :func:`_select_anchors`
    picks the ones this window will actually contain.  Returns ``(None, ())``
    only when there is no candidate at all, i.e. the video was never observed
    -- the caller must refuse rather than guess.

    Everything else about the window, and the evidence for it, is
    :func:`_place_window`'s contract: contained anchors, a window inside the
    medium, and the configuration applied as far as the medium allows.  This
    function's own job is the two ends of that: refusing a report whose
    anchors reach past the medium, and holding ``_select_anchors`` to the
    budget.

    ``maximum_span_seconds`` is therefore a *selection* budget, not a ceiling
    on the window.  It bounds the unpadded union through
    :func:`_select_anchors`, which means the only anchor set that can exceed it
    is one whose single longest observation is itself longer than the budget.
    A window that cannot contain its own anchors is not a window: the plan
    would name evidence the look never frames, and two fields of the same
    artifact would contradict each other.  So the overrun is *recorded* by the
    caller (``FocusLook.window_budget_exceeded``) instead of being paid for by
    truncation.

    Rejected alternative: drop the over-long anchor and keep the next cluster.
    That turns a plannable sample into a refusal whose reason is a property of
    our own configuration rather than of the evidence, which is the kind of
    refusal 4.5 says cannot be argued down.

    Rejected alternative: clamp the union to the medium before placing the
    window.  It reads as the careful thing to do -- the window must be inside
    the file, the evidence must be inside the window -- and it is silently
    wrong: it satisfies the bound by moving the anchors out of the window, so
    the two absolute clauses of the contract are traded against each other
    instead of one of them being refused.  An anchor past the medium is
    corruption in the report and is refused loudly, never clipped around.
    """
    by_video = cited_event_ids(state) if cited is None else cited
    candidates = anchor_candidates(state, video_id=video_id, cited=by_video)
    span_of: dict[str, tuple[float, float]] = {}
    for event_id in candidates:
        event = events.get(event_id)
        if event is None:
            # An event the catalogue does not know about is a wiring bug, not a
            # reason to widen the window silently.
            raise ValueError(f"event {event_id} is missing from the event catalogue")
        span_of[event_id] = (event.start_seconds, event.end_seconds)
    anchor_ids = _select_anchors(candidates, span_of, config)
    if not anchor_ids:
        return None, (), False

    union_start = min(span_of[item][0] for item in anchor_ids)
    union_end = max(span_of[item][1] for item in anchor_ids)
    if union_end > duration_seconds + _MEDIUM_TOLERANCE_SECONDS:
        raise ValueError(
            f"{video_id}: anchor evidence ends at {union_end:.3f}s but the medium ends "
            f"at {duration_seconds:.3f}s; the report cites frames that do not exist"
        )
    window, clipped = _place_window(
        union_start,
        union_end,
        duration_seconds=duration_seconds,
        config=config,
    )
    if window[0] > union_start or window[1] < union_end:
        # ``_place_window`` cannot produce this; the point of the contract is
        # that any edit which reintroduces truncation fails here instead of
        # shipping a plan whose window and anchors contradict each other.
        raise AssertionError("focus window does not contain its anchors")
    return window, anchor_ids, clipped


def _evidence_need(
    *,
    state: WorkingState,
    issue: UnresolvedIssue,
    request: CandidateRequest | None,
    target_videos: tuple[str, ...],
    known_events: Sequence[EventRecord],
    builder: EvidenceNeedBuilder,
) -> tuple[str, str, str]:
    """The one sentence, preferring what the model already wrote for this issue.

    ``candidate_requests[].observable`` is the field the design asks the focus
    planner's model call to produce -- except it is already written, already
    bound to this issue by ``related_issue``, and already checked against the
    real video list by the state validator.  Regenerating it from the issue
    text throws that away and routes around the only validated path from a
    model into the reasoner.

    The builder is the fallback, not the default.  It runs when no request
    names this issue, and when the request's own wording fails the guard --
    the request is a better *source*, not a licence to skip the check.
    """

    def generated() -> str:
        return builder.evidence_need(
            question=state.question,
            issue=issue,
            target_videos=target_videos,
            known_events=known_events,
        ).strip()

    if request is None:
        return generated(), "generated", "no candidate request references this issue"
    statement = request.observable.strip()
    failure = _check_evidence_need(statement)
    if failure is None:
        return statement, "candidate_request", ""
    return (
        generated(),
        "generated",
        f"candidate request {request.request_id} rejected: {failure}",
    )


def _plan_candidate(
    state: WorkingState,
    issue: UnresolvedIssue,
    *,
    events: Mapping[str, EventRecord],
    durations: Mapping[str, float],
    builder: EvidenceNeedBuilder,
    config: FocusPlannerConfig,
) -> tuple[FocusPlan | None, PlannerRefusal | None]:
    by_video = cited_event_ids(state)
    observed = observed_event_ids(state)
    issue_class = _classify(by_video, observed, state, issue)
    requests = requests_for_issue(state, issue)
    request = requests[0] if requests else None
    bundle = _bundle_videos(by_video, issue, request)
    unknown = [video_id for video_id in bundle if video_id not in durations]
    if unknown:
        raise ValueError(
            f"no medium duration supplied for {unknown}; a window that is not known "
            "to lie inside its video cannot be planned"
        )

    # One window per video the bundle can anchor.  A video with no observation
    # at all is not a reason to refuse the whole issue -- the rest of the
    # bundle is still worth looking at -- so it is recorded and skipped.
    windows: list[
        tuple[str, tuple[float, float], tuple[str, ...], tuple[str, ...], bool]
    ] = []
    unanchorable: list[str] = []
    for video_id in bundle:
        candidates = set(anchor_candidates(state, video_id=video_id, cited=by_video))
        window, anchor_ids, clipped = focus_window(
            state,
            events=events,
            video_id=video_id,
            duration_seconds=durations[video_id],
            config=config,
            cited=by_video,
        )
        if window is None:
            unanchorable.append(video_id)
            continue
        windows.append(
            (
                video_id,
                window,
                anchor_ids,
                # Real observations this window does not cover.  Named so the
                # plan is not read as "this is all there is to see here".
                tuple(sorted(candidates - set(anchor_ids))),
                clipped,
            )
        )

    if not windows:
        return None, PlannerRefusal(
            issue_id=issue.issue_id,
            issue_class=issue_class,
            reason=(
                f"no observed event of any of {', '.join(bundle)} to inherit a time "
                "window from; every one of them was never observed, so any window "
                "would have to be invented rather than inherited"
            ),
            missing_videos=tuple(unanchorable),
        )

    # The sentence is written once for the whole round, so the looks are built
    # in two passes: gather the evidence first, then bind the sentence to it.
    known = [
        events[event_id]
        for _, _, anchor_ids, _, _ in windows
        for event_id in anchor_ids
    ]
    statement, need_source, need_note = _evidence_need(
        state=state,
        issue=issue,
        request=request,
        target_videos=tuple(video_id for video_id, _, _, _, _ in windows),
        known_events=known,
        builder=builder,
    )
    guard_failure = _check_evidence_need(statement)
    if guard_failure is not None:
        return None, PlannerRefusal(
            issue_id=issue.issue_id,
            issue_class=issue_class,
            reason=f"evidence_need rejected by the wording guard: {guard_failure}",
        )

    return (
        FocusPlan(
            issue_id=issue.issue_id,
            issue_class=issue_class,
            evidence_need=statement,
            looks=tuple(
                FocusLook(
                    request=FocusRequest(
                        video_id=video_id,
                        evidence_need=statement,
                        time_span=window,
                    ),
                    anchor_event_ids=anchor_ids,
                    # Which layer the window came from.  A cited video needs no
                    # second source, so the fallback only ever applies to an
                    # uncited one -- and it is per look, because one bundle may
                    # hold both kinds.
                    anchor_source=(
                        "state_citation" if by_video.get(video_id) else "report_observation"
                    ),
                    dropped_anchor_event_ids=dropped,
                    window_budget_exceeded=(
                        window[1] - window[0]
                    ) > config.maximum_span_seconds,
                    window_clipped_to_medium=clipped,
                )
                for video_id, window, anchor_ids, dropped, clipped in windows
            ),
            bundle_videos=bundle,
            unanchorable_videos=tuple(unanchorable),
            evidence_need_source=need_source,
            evidence_need_note=need_note,
            request_target_videos=() if request is None else request.target_video_ids,
        ),
        None,
    )


def _check_evidence_need(statement: str) -> str | None:
    if not statement:
        return "the sentence is empty"
    lowered = statement.lower()
    for phrase in _PRESUPPOSING_PHRASES:
        if phrase in lowered:
            return f"the sentence contains {phrase!r}, which asserts a conclusion"
    return None


# ── the entry point ──────────────────────────────────────────────────


def plan_focus(
    state: WorkingState,
    *,
    events: Mapping[str, EventRecord],
    video_durations: Mapping[str, float],
    builder: EvidenceNeedBuilder,
    config: FocusPlannerConfig = FocusPlannerConfig(),
) -> PlannerResult:
    """Turn the best anchorable unresolved issue into one round of looks.

    Issues are walked in a fixed order (see ``_ISSUE_CLASS_ORDER``) and the
    first one that yields a window wins.  Every issue ahead of it that could
    not be anchored is returned in ``refusals``, and every issue behind it in
    ``deferred``: one issue per call is this round's whole scope, so nothing is
    attempted after the winner and nothing leaves the record either.

    An issue whose bundle holds a video with no observation is planned for the
    rest of the bundle rather than refused, and the video it could not cover is
    named.  Only a bundle where *every* video is unobservable is refused.

    ``video_durations`` is how long each video runs, in seconds.  It is an
    argument rather than something this module looks up because planning stays
    media-free -- nothing here opens a file -- while a window that runs past
    the end of its file is still this function's output to get wrong.  Every
    video of a planned bundle must be present: a look into a video of unknown
    length is a claim the planner cannot check, so it is refused loudly rather
    than planned.

    The function is pure: same state, catalogue, durations and builder give the
    same plan.
    """
    # Classify once: the counts behind ``classify_issue`` are the same for every
    # issue, and a sort key is evaluated more than once per element.
    by_video = cited_event_ids(state)
    observed = observed_event_ids(state)
    labels = {
        issue.issue_id: _classify(by_video, observed, state, issue)
        for issue in state.unresolved_issues
    }
    ordered = sorted(
        state.unresolved_issues,
        key=lambda issue: (
            _ISSUE_CLASS_ORDER.index(labels[issue.issue_id])
            if labels[issue.issue_id] in _ISSUE_CLASS_ORDER
            else len(_ISSUE_CLASS_ORDER),
            issue.issue_id,
        ),
    )
    issue_classes = tuple((issue.issue_id, labels[issue.issue_id]) for issue in ordered)

    refusals: list[PlannerRefusal] = []
    for position, issue in enumerate(ordered):
        plan, refusal = _plan_candidate(
            state,
            issue,
            events=events,
            durations=video_durations,
            builder=builder,
            config=config,
        )
        if plan is not None:
            return PlannerResult(
                plan=plan,
                refusals=tuple(refusals),
                considered=tuple(item.issue_id for item in ordered),
                issue_classes=issue_classes,
                deferred=tuple(item.issue_id for item in ordered[position + 1 :]),
            )
        if refusal is not None:
            refusals.append(refusal)
    return PlannerResult(
        plan=None,
        refusals=tuple(refusals),
        considered=tuple(item.issue_id for item in ordered),
        issue_classes=issue_classes,
    )


# ── evidence_need builders ───────────────────────────────────────────


@dataclass(frozen=True)
class DeterministicMockEvidenceNeed:
    """Renders the need from the issue text.  No model, no network.

    Used by the tests and by any run that needs the loop to be reproducible.
    It is intentionally literal: it turns the issue into a question about what
    can be seen, and never mentions a candidate answer.
    """

    def evidence_need(
        self,
        *,
        question: str,
        issue: UnresolvedIssue,
        target_videos: tuple[str, ...],
        known_events: Sequence[EventRecord],
    ) -> str:
        del question, known_events
        return (
            f"Looking only at {', '.join(target_videos)}: {issue.issue.rstrip('.')}? "
            "Report what is directly visible in these frames."
        )


_FOCUS_SYSTEM_PROMPT = """\
You write one observation request for a video inspection tool.

You are given one unresolved issue about a pair of videos and the videos that \
should be looked at.  Write a single sentence describing what to LOOK FOR.

Rules:
- describe something directly visible in frames
- never mention a candidate answer, "the correct", "the different one", or a conclusion
- do not assert which video is right
- return JSON: {"evidence_need": "..."}
"""


@dataclass(frozen=True)
class OpenAICompatibleEvidenceNeed:
    """The model-facing builder.  Its entire output is one sentence."""

    base_url: str
    model: str
    api_key_env: str = "LLM_API_KEY"
    timeout_seconds: float = 120.0
    maximum_tokens: int = 256

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("evidence need builder requires a base URL and a model")
        if self.maximum_tokens <= 0:
            raise ValueError("evidence need maximum_tokens must be positive")

    def evidence_need(
        self,
        *,
        question: str,
        issue: UnresolvedIssue,
        target_videos: tuple[str, ...],
        known_events: Sequence[EventRecord],
    ) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.maximum_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _FOCUS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "unresolved_issue": {
                                "id": issue.issue_id,
                                "scope": list(issue.scope),
                                "issue": issue.issue,
                                "reason": issue.reason,
                            },
                            "look_at_videos": list(target_videos),
                            "already_known_events": [
                                {
                                    "event_id": event.event_id,
                                    "time_span": [event.start_seconds, event.end_seconds],
                                    "description": event.description,
                                    "inferred_interpretation": event.interpretation,
                                }
                                for event in known_events
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        raw = self._post(payload)
        data = parse_json_object(raw)
        statement = data.get("evidence_need")
        if not isinstance(statement, str):
            raise ValueError("evidence need response has no string 'evidence_need'")
        return statement

    def _post(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("evidence need request requires a messages list")
        return OpenAICompatibleTransport(
            base_url=self.base_url,
            model=self.model,
            api_key_env=self.api_key_env,
        ).complete(
            role="focus_evidence_need",
            messages=messages,
            config=APIRequestConfig(
                maximum_tokens=self.maximum_tokens,
                timeout_seconds=self.timeout_seconds,
            ),
        )
