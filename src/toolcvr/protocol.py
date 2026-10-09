"""Build task-agnostic inference semantics from normalized sample metadata.

The adapter is the only layer that understands dataset storage conventions.
This module turns those conventions into explicit, compact facts consumed by
the shared Reasoner loop; it never selects a task-specific solver.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal, Sequence

from toolcvr.schemas import VideoInput


def build_inference_protocol(
    *,
    question: str,
    options: Sequence[str],
    videos: Sequence[VideoInput],
    adapter_notes: Sequence[str] = (),
    answer_kind: Literal["choice", "sequence", "interval", "free_text"] | None = None,
    required_answer_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Describe identities, clocks, answer format, and one-cycle semantics.

    The returned rules are derived from normalized inputs rather than a task
    label, so every CrossVid sample still follows the same reasoning path.
    """
    if not question.strip() or not videos:
        raise ValueError("inference protocol requires a question and videos")

    notes = [
        "Video IDs identify the normalized input clips/views and must never be inferred from event or slot IDs.",
        "A reasoning cycle is PLAN -> local FOCUS observation -> memory revision; Reasoner ALIGN may then rebuild the evidence synthesis when useful. Completing one cycle does not imply sufficient evidence.",
        "FOCUS evidence is valid only for its cited video and time span; it cannot prove whole-video or cross-video absence.",
        "Count physical entities or transitions, not frames, mentions, or duplicate observations of the same event.",
    ]
    resolved_kind = answer_kind or ("choice" if options else "free_text")
    if resolved_kind == "choice":
        notes.extend((
            "The final answer must be exactly one option label from the supplied choices, never a video ID, option text, refusal, or unknown.",
            "If residual uncertainty remains when answering, choose the best-supported option label and retain the uncertainty in the summary.",
        ))
    elif resolved_kind == "sequence":
        required = tuple(required_answer_ids) or tuple(video.video_id for video in videos)
        notes.append(
            "The final answer is one ->-joined sequence containing exactly these input video IDs once each: "
            + ", ".join(required)
            + ". Slot IDs and evidence IDs are invalid answer items."
        )
    elif resolved_kind == "interval":
        notes.append(
            "The final answer is exactly one ordered numeric interval [start_seconds, end_seconds] with start_seconds < end_seconds."
        )
    else:
        notes.append("The final answer is nonempty free text in the format requested by the question.")

    synchronized: dict[str, list[str]] = defaultdict(list)
    for video in videos:
        if video.synchronization_group is not None:
            synchronized[video.synchronization_group].append(video.video_id)
    for group, video_ids in sorted(synchronized.items()):
        if len(video_ids) < 2:
            continue
        notes.append(
            f"Videos {', '.join(video_ids)} share clock {group}: equal timestamps/source-frame positions are comparable across these views."
        )

    by_source: dict[Path, list[VideoInput]] = defaultdict(list)
    for video in videos:
        by_source[video.path.resolve()].append(video)
    for shared in by_source.values():
        if len(shared) < 2 or any(item.synchronization_group for item in shared):
            continue
        video_ids = ", ".join(item.video_id for item in shared)
        if all(item.model_clock == "local" for item in shared):
            notes.append(
                f"Videos {video_ids} are independent logical clips with clip-local clocks. "
                "Their timestamps and frame positions are not comparable across clips and "
                "contain no original-source ordering information; infer chronology only "
                "from visible state transitions."
            )
        else:
            notes.append(
                f"Videos {video_ids} are normalized inputs from one source asset; retain "
                "each input identity. Compare timestamps only when another protocol rule "
                "explicitly declares a shared clock."
            )

    notes.extend(note.strip() for note in adapter_notes if note.strip())
    # Stable de-duplication keeps the protocol compact and deterministic.
    return tuple(dict.fromkeys(notes))
