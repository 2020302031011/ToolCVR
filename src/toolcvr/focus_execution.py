"""Stage 4: execute a focus plan, one ``FocusTool`` call per look.

This is the bridge between the two agents, which is why it is its own module
rather than part of either: the planner must stay media-free, and ``focus.py``
cannot import the plan it is being driven by without a cycle.

The one rule this module exists to keep is that a round is a *sequence of
single-video looks*, never one request holding frames from several videos
(``AGENTS.md``).  Each look is one video, one window, one call, and the
evidence is merged only afterwards, when the state is updated -- which is why
nothing here concatenates reports into one, and why the plan's single sentence
is passed to every call unchanged.

What is deliberately *not* here: review, state-delta generation, and the merge
itself.  Those are stage 5.  The merge point already exists --
``state_update.apply_state_delta`` accepts an ``AddInformation`` naming several
videos -- so what a round owes it is exactly this bundle, with every event id
namespaced by video first (see ``FocusEvidenceBundle.repeated_event_ids``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from toolcvr.focus import FocusResult, FocusTool, split_complete_zoom
from toolcvr.focus_planner import FocusPlan
from toolcvr.schemas import EventRecord, VideoInput


@dataclass(frozen=True)
class FocusEvidenceBundle:
    """One executed round: one result per look, in plan order.

    ``unanchorable_videos`` is carried through from the plan on purpose.  The
    results alone would read as "this round looked at v1 and v3", and for an
    issue whose scope also holds a video nothing has ever observed, that is
    half a fact: the issue is still open, and this field is what says why.
    """

    issue_id: str
    issue_class: str
    evidence_need: str
    results: tuple[FocusResult, ...]
    unanchorable_videos: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("a focus bundle must carry at least one result")
        for result in self.results:
            if any(frame.video_id != result.video_id for frame in result.frame_manifest):
                raise ValueError("a focus result may hold frames from one video only")

    @property
    def videos(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(result.video_id for result in self.results))

    @property
    def events(self) -> tuple[EventRecord, ...]:
        """Every event the round produced, look order preserved."""
        return tuple(
            event for result in self.results for event in result.observation.events
        )

    @property
    def partial(self) -> bool:
        return bool(self.unanchorable_videos)

    @property
    def repeated_event_ids(self) -> tuple[str, ...]:
        """Ids the round's evidence uses more than once.

        Reporters number their events within one batch (``e1``, ``e2``, ...),
        so two looks can legitimately return the same id for different events.
        ``report_generation.py`` prefixes each id with its video before the
        events reach a sample-wide namespace; a focus round bypasses that, and
        the merge into ``WorkingState.event_videos`` is keyed by event id, so a
        collision there would silently drop one video's evidence -- a
        provenance loss with no error attached to it.

        Recorded rather than repaired: renaming here would desynchronize
        ``report_text``, which embeds the ids verbatim.  The rename belongs to
        the boundary that lifts a round into the sample-wide namespace.
        """
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.event_id] = counts.get(event.event_id, 0) + 1
        return tuple(sorted(item for item, count in counts.items() if count > 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_class": self.issue_class,
            "evidence_need": self.evidence_need,
            "partial": self.partial,
            "unanchorable_videos": list(self.unanchorable_videos),
            "repeated_event_ids": list(self.repeated_event_ids),
            "looks": [
                {
                    "video_id": result.video_id,
                    "request_kind": result.request_kind,
                    "resolved_time_span": list(result.resolved_time_span),
                    "segment_id": _segment_id(result),
                    "frame_count": len(result.frame_manifest),
                    # Keep the complete ObserverAgent result.  An events-only
                    # serialization would turn a failed observation into an
                    # apparent absence at the state-update boundary.
                    "observation": {
                        "events": [
                            event.to_mapping() for event in result.observation.events
                        ],
                        "query_assessment": result.observation.query_assessment.to_mapping(),
                        "observation_gaps": [
                            gap.to_mapping()
                            for gap in result.observation.observation_gaps
                        ],
                    },
                    "report_text": result.report_text,
                }
                for result in self.results
            ],
        }


def _segment_id(result: FocusResult) -> str:
    """The single segment a look resolves to.

    ``FocusTool`` builds every frame of a call with one segment id, so more
    than one here means the frames came from somewhere else -- which is the
    provenance failure the id exists to detect.
    """
    segments = {frame.segment_id for frame in result.frame_manifest}
    if len(segments) != 1:
        raise ValueError(
            f"a focus result must resolve exactly one segment, got {sorted(segments)}"
        )
    return segments.pop()


def execute_focus_plan(
    *,
    plan: FocusPlan,
    videos: Sequence[VideoInput],
    tool: FocusTool,
    question: str,
    report_hints: Sequence[str] = (),
) -> FocusEvidenceBundle:
    """Run the plan: one call per look, in plan order, one video each.

    ``question`` is the sample's question and ``report_hints`` the reasoner's
    standing hints; the plan's own sentence is appended to both by
    ``FocusTool`` itself, so a comparison the round is asking is visible to
    every call without this function having to restate it.

    A missing video input or a duplicate one is an error, not a partial round:
    the plan was built against a fixed video list, so either means the caller
    handed over a different sample.  Nothing is swallowed -- one look that the
    reporter cannot turn into events fails the whole round, because a bundle
    missing one of its videos is not the evidence the plan asked for.
    """
    if not plan.looks:
        # Unreachable through FocusPlan's constructor.  Asserted anyway: a
        # round that executes nothing must not come back looking like a round
        # that found nothing.
        raise AssertionError("focus plan contains no looks")
    if not question.strip():
        raise ValueError("focus execution question must be nonempty")
    by_id: dict[str, VideoInput] = {}
    for video in videos:
        if video.video_id in by_id:
            raise ValueError(f"duplicate video input for {video.video_id}")
        by_id[video.video_id] = video
    missing = [look.video_id for look in plan.looks if look.video_id not in by_id]
    if missing:
        raise ValueError(f"focus plan names videos with no video input: {missing}")

    return FocusEvidenceBundle(
        issue_id=plan.issue_id,
        issue_class=plan.issue_class,
        evidence_need=plan.evidence_need,
        results=tuple(
            result
            for look in plan.looks
            for result in (
                tool.run(
                    video=by_id[look.video_id], request=request, question=question,
                    report_hints=report_hints,
                )
                for request in split_complete_zoom(
                    video_id=look.video_id, time_span=look.request.time_span
                    if look.request.time_span is not None else _timestamp_span(look.request, tool),
                    evidence_need=look.request.evidence_need,
                    frames_per_call=tool.config.frames_per_call,
                )
            )
        ),
        unanchorable_videos=plan.unanchorable_videos,
    )


def _timestamp_span(request, tool: FocusTool) -> tuple[float, float]:
    if request.center_timestamp_seconds is None:
        raise AssertionError("validated focus request has no locator")
    return (
        max(0.0, request.center_timestamp_seconds - tool.config.default_radius_seconds),
        request.center_timestamp_seconds + tool.config.default_radius_seconds,
    )
