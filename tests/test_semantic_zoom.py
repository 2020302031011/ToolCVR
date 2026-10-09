from toolcvr.semantic_zoom import SemanticZoomMemory, ZoomDraft, ZoomSlot


def _draft(span: tuple[float, float]) -> ZoomDraft:
    return ZoomDraft("z1", "v2", (10.0, 25.0), span, ("v2.scan.r0.b1.e1",))


def test_complete_zoom_creates_current_fine_slots_and_hides_parent() -> None:
    memory = SemanticZoomMemory().apply_complete_zoom(
        drafts=(_draft((10.0, 20.0)), _draft((20.0, 25.0))),
        contents=("first dense observation", "second dense observation"),
        evidence_refs=(("v2.focus.r1.l1.e1",), ("v2.focus.r1.l2.e1",)),
    )
    assert [item.time_span for item in memory.slots] == [(10.0, 20.0), (20.0, 25.0)]
    assert all(item.resolution == "fine" for item in memory.slots)
    assert memory.superseded_evidence_refs == ("v2.scan.r0.b1.e1",)
    assert len(memory.render()) == 2


def test_incomplete_zoom_is_rejected_without_a_partial_memory_write() -> None:
    try:
        SemanticZoomMemory().apply_complete_zoom(
            drafts=(_draft((10.0, 20.0)),), contents=("only part",),
            evidence_refs=(("v2.focus.r1.l1.e1",),),
        )
    except ValueError as exc:
        assert "completely cover" in str(exc)
    else:
        raise AssertionError("partial zoom replaced a coarse region")


def test_noncontiguous_zoom_children_are_rejected() -> None:
    try:
        SemanticZoomMemory().apply_complete_zoom(
            drafts=(_draft((10.0, 19.0)), _draft((20.0, 25.0))),
            contents=("a", "b"), evidence_refs=(("a",), ("b",)),
        )
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("gapped children were accepted")


def test_exact_parent_slot_is_replaced_while_neighbouring_coarse_slot_survives() -> None:
    memory = SemanticZoomMemory(slots=(
        ZoomSlot("v2.c1", "v2", (10.0, 25.0), "coarse", "target", ("e1",), 0),
        ZoomSlot("v2.c2", "v2", (20.0, 30.0), "coarse", "overlap", ("e2",), 0),
    ))
    drafts = (
        ZoomDraft("z1", "v2", (10.0, 25.0), (10.0, 20.0), ("e1",), ("v2.c1",)),
        ZoomDraft("z1", "v2", (10.0, 25.0), (20.0, 25.0), ("e1",), ("v2.c1",)),
    )
    updated = memory.apply_complete_zoom(
        drafts=drafts, contents=("fine a", "fine b"), evidence_refs=(("f1",), ("f2",)),
    )
    ids = {slot.slot_id for slot in updated.slots}
    assert "v2.c1" not in ids
    assert "v2.c2" in ids
    assert len(updated.focusable_slots) == 3


def test_complete_zoom_preserves_uncertainty_and_allows_fine_refocus() -> None:
    memory = SemanticZoomMemory(slots=(
        ZoomSlot("v1.c001", "v1", (0.0, 5.0), "coarse", "coarse action", ("e1",), 0),
    ))
    draft = ZoomDraft("z2", "v1", (0.0, 5.0), (0.0, 5.0), ("e1",), ("v1.c001",))
    updated = memory.apply_complete_zoom(
        drafts=(draft,), contents=("hand approaches object",),
        evidence_refs=(("f1",),),
        uncertainties=(("contact is not visible",),),
    )
    assert updated.slots[0].resolution == "fine"
    assert updated.slots[0].uncertainty == ("contact is not visible",)
    assert updated.focusable_slots == updated.slots


def test_unobserved_coverage_slot_may_exist_without_visual_evidence() -> None:
    slot = ZoomSlot(
        "v1.u001", "v1", (5.0, 10.0), "coverage",
        "Unobserved warm-scan interval.", (), 0,
        ("No warm observation was made.",),
    )
    assert slot.evidence_refs == ()
    assert slot in SemanticZoomMemory((slot,)).focusable_slots
