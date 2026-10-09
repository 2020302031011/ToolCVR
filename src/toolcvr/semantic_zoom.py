"""Bounded, replace-in-place temporal memory for complete visual zooms.

The evidence vault is append-only (the caller's :class:`EvidenceLedger`).  This
module deliberately stores only the *current* representation of a region that
has been completely re-observed.  A zoom is atomic: a partially decoded region
cannot replace its coarse anchors.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence


@dataclass(frozen=True)
class ZoomSlot:
    slot_id: str
    video_id: str
    time_span: tuple[float, float]
    resolution: Literal["coarse", "fine", "coverage"]
    content: str
    evidence_refs: tuple[str, ...]
    revision: int = 1
    uncertainty: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        start, end = self.time_span
        if not self.slot_id or not self.video_id or start < 0 or end < start:
            raise ValueError("zoom slot requires id, video and ordered positive span")
        if self.resolution == "fine" and end <= start:
            raise ValueError("fine zoom slot requires a nonzero time span")
        if not self.content.strip():
            raise ValueError("zoom slot requires rendered content")
        if self.resolution != "coverage" and not self.evidence_refs:
            raise ValueError("observed zoom slots require evidence provenance")
        if any(not item.strip() for item in self.uncertainty):
            raise ValueError("zoom-slot uncertainty entries must be nonempty")


@dataclass(frozen=True)
class ZoomDraft:
    """One dense child of an all-or-nothing requested zoom region."""

    zoom_id: str
    video_id: str
    region_span: tuple[float, float]
    slot_span: tuple[float, float]
    parent_evidence_refs: tuple[str, ...]
    parent_slot_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for start, end in (self.region_span, self.slot_span):
            if start < 0 or end <= start:
                raise ValueError("zoom draft spans must be ordered and positive")
        if not self.zoom_id or not self.video_id:
            raise ValueError("zoom draft needs identity")


@dataclass(frozen=True)
class SemanticZoomMemory:
    """Reasoner-facing current coarse/fine slots; history stays outside."""

    slots: tuple[ZoomSlot, ...] = ()
    superseded_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [item.slot_id for item in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("zoom slot ids must be unique")

    def by_ids(self, slot_ids: Sequence[str]) -> tuple[ZoomSlot, ...]:
        by_id = {item.slot_id: item for item in self.slots}
        missing = [slot_id for slot_id in slot_ids if slot_id not in by_id]
        if missing:
            raise ValueError(f"focus intent cites slots absent from current memory: {missing}")
        return tuple(by_id[slot_id] for slot_id in slot_ids)

    @property
    def focusable_slots(self) -> tuple[ZoomSlot, ...]:
        """Every current address may be refined; coverage addresses recover omissions."""
        return self.slots

    def apply_complete_zoom(
        self,
        *,
        drafts: Sequence[ZoomDraft],
        contents: Sequence[str],
        evidence_refs: Sequence[Sequence[str]],
        uncertainties: Sequence[Sequence[str]] | None = None,
    ) -> "SemanticZoomMemory":
        """Commit one complete zoom only after every dense child succeeded."""
        if not drafts or len(drafts) != len(contents) or len(drafts) != len(evidence_refs):
            raise ValueError("complete zoom requires aligned nonempty draft results")
        if uncertainties is None:
            uncertainties = tuple(() for _ in drafts)
        if len(drafts) != len(uncertainties):
            raise ValueError("complete zoom requires aligned uncertainty results")
        first = drafts[0]
        if any((item.zoom_id, item.video_id, item.region_span, item.parent_evidence_refs,
                item.parent_slot_ids) !=
               (first.zoom_id, first.video_id, first.region_span, first.parent_evidence_refs,
                first.parent_slot_ids)
               for item in drafts):
            raise ValueError("a zoom commit may contain children from exactly one region")
        ordered = sorted(drafts, key=lambda item: item.slot_span)
        if tuple(ordered) != tuple(drafts):
            raise ValueError("zoom children must be supplied in temporal order")
        if ordered[0].slot_span[0] != first.region_span[0] or ordered[-1].slot_span[1] != first.region_span[1]:
            raise ValueError("zoom children must completely cover the requested region")
        for left, right in zip(ordered, ordered[1:]):
            # Adjacent dense clips intentionally share their boundary frame.
            if abs(left.slot_span[1] - right.slot_span[0]) > 1e-6:
                raise ValueError("zoom children must be temporally contiguous")
        prefix = f"{first.video_id}.zoom.{first.zoom_id}"
        parent_ids = set(first.parent_slot_ids)
        parent_revision = max(
            (item.revision for item in self.slots if item.slot_id in parent_ids),
            default=0,
        )
        new_slots = tuple(
            ZoomSlot(
                slot_id=f"{prefix}.s{index:02d}", video_id=draft.video_id,
                time_span=draft.slot_span, resolution="fine", content=content,
                evidence_refs=tuple(refs), uncertainty=tuple(uncertainty),
                revision=parent_revision + 1,
            )
            for index, (draft, content, refs, uncertainty) in enumerate(
                zip(drafts, contents, evidence_refs, uncertainties), start=1
            )
        )
        # A later complete zoom of the same region replaces its visible slots,
        # rather than appending another history layer.
        if first.parent_slot_ids:
            retained = tuple(item for item in self.slots if item.slot_id not in parent_ids)
        else:
            # Compatibility for independently constructed drafts: region
            # replacement remains deterministic, but the normal execution
            # path always supplies exact parent slot identities.
            retained = tuple(item for item in self.slots if not (
                item.video_id == first.video_id and item.time_span[0] >= first.region_span[0]
                and item.time_span[1] <= first.region_span[1]
            ))
        hidden = tuple(sorted(set((*self.superseded_evidence_refs, *first.parent_evidence_refs))))
        return replace(self, slots=(*retained, *new_slots), superseded_evidence_refs=hidden)

    def render(self) -> list[dict[str, object]]:
        return [
            {"slot_id": item.slot_id, "video": item.video_id, "time": item.time_span,
             "resolution": item.resolution, "content": item.content,
             "evidence": list(item.evidence_refs), "uncertainty": list(item.uncertainty),
             "revision": item.revision, "focusable": True}
            for item in sorted(self.slots, key=lambda item: (item.video_id, item.time_span, item.slot_id))
        ]
