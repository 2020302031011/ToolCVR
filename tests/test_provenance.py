"""The rule that artifacts name their inputs and consumers check them.

Both halves are tested here rather than at the call sites: the records are the
same shape on every artifact, so the check is the same function, and a rule
tested once is a rule that does not drift between the artifacts it governs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from toolcvr.provenance import (
    ProvenanceError,
    file_digest,
    local_qwen3vl_identity_suffix,
    path_identity,
    reuse_decision,
    verify_recorded_input,
)


def _record(path: Path, prefix: str) -> dict[str, str]:
    return {f"{prefix}_source": str(path), f"{prefix}_sha256": file_digest(path)}


def test_a_recorded_input_verifies_against_the_file_it_was_taken_from(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"question": "which video?"}', encoding="utf-8")
    verify_recorded_input(_record(report, "report"), prefix="report", path=report)


def test_a_file_edited_after_the_record_was_written_is_refused(tmp_path) -> None:
    """The case a path check alone waves through.

    A file rewritten in place keeps its name, so an artifacts tree can be
    internally consistent by path and wrong by content -- which is the whole
    reason the digest is recorded next to the path.
    """

    report = tmp_path / "report.json"
    report.write_text('{"question": "which video?"}', encoding="utf-8")
    record = _record(report, "report")
    report.write_text('{"question": "which video?", "answer": "the third one"}', encoding="utf-8")

    try:
        verify_recorded_input(record, prefix="report", path=report)
    except ProvenanceError as exc:
        assert str(report) in str(exc)
    else:
        raise AssertionError("a file edited in place after the record was written verified")


def test_a_run_pointed_at_another_samples_file_is_refused(tmp_path) -> None:
    planned = tmp_path / "CC-474.json"
    planned.write_text('{"question": "A"}', encoding="utf-8")
    other = tmp_path / "CC-475.json"
    other.write_text('{"question": "B"}', encoding="utf-8")

    try:
        verify_recorded_input(_record(planned, "report"), prefix="report", path=other)
    except ProvenanceError as exc:
        # Both paths, because which one is right depends on the run.
        assert str(planned) in str(exc) and str(other) in str(exc)
    else:
        raise AssertionError("a run pointed at another sample's report verified")


def test_a_missing_input_is_refused_by_name(tmp_path) -> None:
    gone = tmp_path / "moved-away.json"
    try:
        verify_recorded_input(
            {"report_source": str(gone), "report_sha256": "0" * 64},
            prefix="report",
            path=gone,
        )
    except ProvenanceError as exc:
        assert "moved-away.json" in str(exc)
    else:
        raise AssertionError("an input that is not on disk verified")


def test_an_unrecorded_input_is_refused_rather_than_assumed(tmp_path) -> None:
    """An artifact that names no input is not an artifact with nothing to check.

    Reading silence as consent is how the missing half of a record becomes a
    passed check, so an incomplete pair is refused outright -- including the
    half-written cases, where one of the two keys is present.
    """

    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    for run in ({}, {"report_source": str(report)}, {"report_sha256": file_digest(report)}):
        try:
            verify_recorded_input(run, prefix="report", path=report)
        except ProvenanceError:
            continue
        raise AssertionError(f"an incomplete record was accepted: {run}")


def test_the_check_is_on_content_so_a_moved_tree_still_verifies(tmp_path) -> None:
    """Locations change; bytes are the invariant.

    An outputs tree that was moved intact is the same artifact tree, and
    refusing it would make the check something to work around rather than
    something to rely on.
    """

    original = tmp_path / "before" / "report.json"
    original.parent.mkdir()
    original.write_text('{"question": "which video?"}', encoding="utf-8")
    record = _record(original, "report")
    moved = tmp_path / "after" / "report.json"
    moved.parent.mkdir()
    moved.write_text('{"question": "which video?"}', encoding="utf-8")

    verify_recorded_input(record, prefix="report", path=moved)


def _fingerprint_the_way_the_reports_did(files: list[tuple[str, int, int]]) -> str:
    """The algorithm, spelled out independently of the implementation.

    Every scenario in this repository names its model as ``<path>@<16 hex>``,
    and that number was produced by this expression.  Written out here so the
    test fails when the implementation changes rather than agreeing with it by
    construction -- which is the only way a pinned value is worth anything.
    """
    rows = [{"path": name, "size": size, "mtime_ns": mtime} for name, size, mtime in files]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def test_the_model_identity_is_pinned_by_value(tmp_path) -> None:
    """It is compared across frozen artifacts, so it cannot be free to change.

    A model directory is named by a short digest of its file list.  The
    reports on disk carry identities produced by one particular way of writing
    that list down -- the fields, their order, the JSON separators -- and a
    change to any of them makes every identity written afterwards
    incomparable with every identity already recorded, silently, for a reason
    that has nothing to do with the model.  So the recipe is pinned here by
    value: this test is the thing that fails when someone reaches for
    ``json.dumps`` without the separators.
    """
    model = tmp_path / "Qwen-VL"
    model.mkdir()
    (model / "config.json").write_bytes(b'{"arch": "qwen3vl"}')
    (model / "model.safetensors").write_bytes(b"\x00" * 32)

    files = [
        (
            name,
            (model / name).stat().st_size,
            (model / name).stat().st_mtime_ns,
        )
        for name in ("config.json", "model.safetensors")
    ]
    assert path_identity(model) == _fingerprint_the_way_the_reports_did(files)

    # A file path is fingerprinted the same way, as a one-file tree keyed by
    # its own name, so an adapter passed as a single file still identifies.
    adapter = tmp_path / "adapter.bin"
    adapter.write_bytes(b"weights")
    assert path_identity(adapter) == _fingerprint_the_way_the_reports_did(
        [("adapter.bin", adapter.stat().st_size, adapter.stat().st_mtime_ns)]
    )


def test_the_identity_changes_when_the_weights_do(tmp_path) -> None:
    """What the digest is for: "the same weights" versus "other weights"."""

    model = tmp_path / "model"
    model.mkdir()
    (model / "a.bin").write_bytes(b"first")
    before = path_identity(model)
    (model / "b.bin").write_bytes(b"second")
    assert path_identity(model) != before

    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        path_identity(empty)
    except ValueError as exc:
        assert str(empty) in str(exc)
    else:
        raise AssertionError("an empty model directory was given an identity")


def test_both_stages_name_a_reporter_in_the_same_words(tmp_path) -> None:
    """The identity is only worth writing if the two stages agree on its shape.

    ``reporter_identity`` and ``observation_reporter_identity`` sit next to
    each other in every evidence bundle, so a reader will compare them.  Two
    copies of the string-building had already stopped agreeing: one used
    compact JSON separators and the other did not, giving the same model
    directory two different digests, and one named no overlay setting at all.
    Neither raised anything.  One recipe, so a difference between the two
    strings in a bundle means a difference in the reporter.
    """

    model = tmp_path / "model"
    model.mkdir()
    (model / "a.bin").write_bytes(b"weights")
    kwargs = {
        "model": model,
        "adapter": None,
        "maximum_new_tokens": 2048,
        "maximum_events": 8,
        "overlay_timestamps": True,
    }
    suffix = local_qwen3vl_identity_suffix(**kwargs)
    assert suffix == (
        f"model={model}@{path_identity(model)}:adapter=none@none:"
        f"tokens=2048:events=8:overlay=True"
    )
    # Every parameter that changes what the model is shown must move the
    # string, or a run that asked a different question looks like the same
    # observer.
    for field, other in (
        ("maximum_new_tokens", 4096),
        ("maximum_events", 16),
        ("overlay_timestamps", False),
    ):
        assert local_qwen3vl_identity_suffix(**{**kwargs, field: other}) != suffix


def test_reuse_is_decided_by_the_input_and_the_producer_together() -> None:
    """Neither half is enough on its own.

    Same state, different planning rules: every digest matches and the plan on
    disk was made by rules that no longer exist.  Checking only the input turns
    ``--resume`` into "serve the previous version of the code".
    """

    run = {"state_sha256": "d1", "planner_identity": "focus-planner-v4"}
    inputs = {"state_sha256": ("the initial state", "d1")}
    here = Path("/plans/CC-474/focus_plan.json")

    assert (
        reuse_decision(
            run,
            subject="plan",
            identities={"planner_identity": "focus-planner-v4"},
            inputs=inputs,
            destination=here,
        )
        is None
    )
    assert (
        reuse_decision(
            run,
            subject="plan",
            identities={"planner_identity": "focus-planner-v5"},
            inputs=inputs,
            destination=here,
        )
        == "plan records planner_identity='focus-planner-v4'; "
        "this run uses 'focus-planner-v5'"
    )
    assert (
        reuse_decision(
            run,
            subject="plan",
            identities={"planner_identity": "focus-planner-v4"},
            inputs={"state_sha256": ("the initial state", "d2")},
            destination=here,
        )
        == "the initial state changed"
    )


def test_the_input_is_reported_before_the_producer_when_both_changed() -> None:
    """The reason printed is the one the reader can act on first.

    A changed input means the artifact was made from something else; a changed
    producer means the rules moved.  Reporting the input first keeps the
    message about the data, so a run whose rules changed does not hide a state
    that was rebuilt underneath it.
    """

    reason = reuse_decision(
        {"state_sha256": "d1", "planner_identity": "focus-planner-v4"},
        subject="plan",
        identities={"planner_identity": "focus-planner-v5"},
        inputs={"state_sha256": ("the initial state", "d2")},
        destination=Path("/plans/CC-474/focus_plan.json"),
    )
    assert reason == "the initial state changed"


def test_a_mock_run_may_not_quietly_replace_a_models_work() -> None:
    """The direction matters, and only one of them loses something.

    Rebuilding a plan because the planning rules changed is the whole point of
    ``--resume``.  Rebuilding one *because this run is a mock* is a different
    act: the mock's output is well formed, passes every check the model's did,
    and leaves nothing on disk saying the corpus stopped being an observation.
    This is not hypothetical -- a ``--mock --resume`` over a corpus whose
    states a local Qwen3-14B had built rebuilt all 27 of them, and the
    artifacts overwritten were the only copy.

    So the downgrade stops the run by name, before anything is written, and
    says which flag means the operator meant it.
    """

    real = "openai-compatible-v3-anchored-local-summary:Qwen3-14B"
    mock = "deterministic-mock-v2-local-assessment"
    run = {"report_sha256": "d1", "builder_identity": real}
    here = Path("/states/CC-474/initial_state.json")
    decision = dict(
        subject="state",
        inputs={"report_sha256": ("the report", "d1")},
        destination=here,
    )

    try:
        reuse_decision(run, identities={"builder_identity": mock}, **decision)
    except ProvenanceError as exc:
        message = str(exc)
        # The file, both producers, and the way through.
        assert str(here) in message
        assert real in message and mock in message
        assert "--allow-builder-downgrade" in message
    else:
        raise AssertionError("a mock run overwrote a model's artifact without asking")

    # The name of the flag is not the point; saying so is.
    assert (
        reuse_decision(
            run, identities={"builder_identity": mock}, allow_downgrade=True, **decision
        )
        is not None
    )

    # The same producer, a mock over a mock: an ordinary rebuild.
    on_disk = {"report_sha256": "d1", "builder_identity": mock}
    assert (
        reuse_decision(on_disk, identities={"builder_identity": mock}, **decision) is None
    ), "a mock run refused to reuse its own artifact"

    # And an upgrade is not a downgrade: the model's work replaces the mock's,
    # which is what re-running the pipeline over a mock corpus does.
    assert (
        reuse_decision(on_disk, identities={"builder_identity": real}, **decision)
        is not None
    )


def test_every_producer_of_an_artifact_is_checked_not_just_the_first() -> None:
    """A second producer is the one that gets forgotten, so it is the one tested.

    A focus plan has two: the rules that placed its windows, and the model that
    wrote the ``evidence_need`` sentence.  When only the rules were compared,
    the plan a mock wrote and the plan a model wrote recorded the *same*
    ``planner_identity`` -- so a ``--mock --resume`` rebuilt 27 model-planned
    plans and the two were indistinguishable afterwards.  The rules were named
    in the run block the whole time; they just were not the field that could
    tell the difference.
    """

    real = "openai-compatible:Qwen3-14B"
    mock = "deterministic-mock"
    rules = "focus-planner-v5-window-placement-contract"
    here = Path("/plans/CC-474/focus_plan.json")
    decision = dict(
        subject="plan",
        identities={"planner_identity": rules, "builder_identity": mock},
        inputs={"state_sha256": ("the initial state", "d1")},
        destination=here,
    )

    # A model planned this one; the rules still match and always would.
    on_disk = {
        "state_sha256": "d1",
        "planner_identity": rules,
        "builder_identity": real,
    }
    try:
        reuse_decision(on_disk, **decision)
    except ProvenanceError as exc:
        assert real in str(exc) and mock in str(exc)
    else:
        raise AssertionError(
            "a mock run replaced a model-planned plan while the rules matched"
        )

    # With the flag, it is the ordinary rebuild the operator asked for.
    assert reuse_decision(on_disk, allow_downgrade=True, **decision) is not None

    # A mock over its own plan is an ordinary resume.
    mock_plan = {
        "state_sha256": "d1",
        "planner_identity": rules,
        "builder_identity": mock,
    }
    assert reuse_decision(mock_plan, **decision) is None

    # And the reverse is an upgrade, not a downgrade: a model re-planning a
    # mock's plan, which is what re-running stage 4 over a mock corpus is.
    assert (
        reuse_decision(
            mock_plan,
            subject="plan",
            identities={"planner_identity": rules, "builder_identity": real},
            inputs={"state_sha256": ("the initial state", "d1")},
            destination=here,
        )
        is not None
    )


def test_the_downgrade_check_does_not_reorder_the_ordinary_reasons() -> None:
    """A stale input is still reported as a stale input.

    The refusal is about the producer, so it fires whether or not the inputs
    also moved -- but when nothing is being lost, the reason printed is the one
    the reader can act on, not a lecture about builder classes.
    """

    decision = dict(
        subject="state",
        identities={"builder_identity": "deterministic-mock-v3"},
        inputs={"report_sha256": ("the report", "d2")},
        destination=Path("/states/CC-474/initial_state.json"),
    )
    assert (
        reuse_decision(
            {"report_sha256": "d1", "builder_identity": "deterministic-mock-v2"},
            **decision,
        )
        == "the report changed"
    )
