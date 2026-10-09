"""Artifacts name their inputs; every consumer checks them.

Each artifact in this repository carries a ``run`` block recording, for every
input it consumed, a ``<name>_source`` path and a ``<name>_sha256`` digest, plus
the identities of its producers.  Those records are what ``--resume`` reads to
decide whether a file on disk is still valid, and what a consumer about to use
an artifact reads to check that the file in hand is the file the artifact was
made from.

The artifacts form one chain -- a report is observed, a state is built from it,
a plan is planned from the state, evidence is executed from the plan -- and
every link is checked the same way, by the consumer of the earlier link:

===============  ==========================  ==============================
artifact         records                     checked by
===============  ==========================  ==============================
``state``        ``report_sha256``           ``plan_focus.py``
``plan``         ``state_sha256``            ``plan_focus.py`` (resume)
``plan``         ``report_sha256``           ``execute_focus_plans.py``
``evidence``     ``plan_sha256``             ``execute_focus_plans.py`` (resume)
===============  ==========================  ==============================

A record that is written but never read leaves the chain unverified, and so
does a check re-implemented per caller -- the second is how the first happens
by accident.  Both live here once.

The same records answer the other question the entry points ask: whether the
finished artifact on disk may be *replaced*.  That one has a direction --
rebuilding a mock's output from a model's is an upgrade, and the reverse
throws away the only copy of something a model produced -- so it is refused
unless asked for by name.  See :func:`reuse_decision`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MOCK_REPORTER_COMPATIBLE_IDENTITIES",
    "MOCK_REPORTER_IDENTITY",
    "ProvenanceError",
    "file_digest",
    "local_qwen3vl_identity_suffix",
    "path_identity",
    "reuse_decision",
    "verify_recorded_input",
]

# ---------------------------------------------------------------------------
# Producer identities
#
# Strings that name *who made* an artifact, written into ``run`` blocks and read
# back by the stage that consumes them.  Their only use is a comparison -- in a
# bundle, in a reuse decision, or by a person asking whether two stages used
# the same observer -- so they are defined once here rather than spelled out
# where they happen to be needed.  The refusal to fold a real model into a mock
# is only as good as the predicate below, and the predicate is only as good as
# the convention being one place.
# ---------------------------------------------------------------------------

# What a deterministic mock reporter calls itself.  The observation and the
# focus round both write this, and a bundle records the two side by side, so
# two spellings of it would be a disagreement about which reporter looked --
# the same defect the model identity had, in the same two files.
MOCK_REPORTER_IDENTITY = "deterministic-mock-event-reporter-v5-local-assessment"

# Older names for the same reporter, which a run reading a finished artifact
# may treat as its own: a mock that produces the same frames under a previous
# name is the same observer, so `--resume` reuses rather than rebuilds.  Only
# the observation stage accepts these; the focus executor has no such list,
# because a name it does not recognise is a different reporter to compare
# against, not a reason to reuse.
MOCK_REPORTER_COMPATIBLE_IDENTITIES = ("deterministic-mock-event-reporter-v4-assessment",)


class ProvenanceError(ValueError):
    """An artifact's record of its own inputs does not match the input in hand.

    A subclass of ``ValueError`` because that is what it is; named because a
    caller repairing an outputs tree needs to tell "the file you handed me is
    not the one this artifact came from" apart from "the artifact is malformed".
    """


def file_digest(path: Path) -> str:
    """The digest an artifact records for a file input: sha256 of its bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_identity(path: Path) -> str:
    """A short identity for a path holding files: a model or an adapter.

    Not the digest of the bytes -- a multi-gigabyte model directory would be
    read on every run -- but a digest over the file list with sizes and
    modification times, which is enough to tell "the same weights" from
    "weights re-downloaded or quantized" and cheap enough to compute first.

    It lives here rather than in each entry point that needs it because its
    only use is a *comparison*.  Both the observation and the focus round write
    ``model=<path>@<identity>`` into their artifacts so the two can be read
    against each other, and two copies of this function did drift: one sorted
    its JSON with compact separators and the other without, so the same
    directory identified as ``36ed3aa315d8884d`` in a report and
    ``4f2ef03c6daa3b90`` in the evidence bundle beside it -- a disagreement
    about the model where there was none.
    """
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    if not files:
        raise ValueError(f"model/adapter path contains no files: {path}")
    identity = [
        {
            "path": str(item.relative_to(path) if path.is_dir() else item.name),
            "size": item.stat().st_size,
            "mtime_ns": item.stat().st_mtime_ns,
        }
        for item in files
    ]
    # The separators are part of the algorithm, not a formatting preference:
    # every identity already recorded in this repository's artifacts was
    # produced with them, so changing them would silently make each newly
    # written identity incomparable with each frozen one.
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def local_qwen3vl_identity_suffix(
    *,
    model: Path,
    adapter: Path | None,
    maximum_new_tokens: int,
    maximum_events: int,
    overlay_timestamps: bool,
) -> str:
    """The parameters of a local Qwen3VL reporter, as one comparable string.

    Two entry points build this reporter -- the warm observation and the focus
    round -- and the point of writing it down is that a reader can compare the
    two: a focus round that claims to have looked through the same eyes as the
    report should name the same model bytes, the same adapter, and the same
    decoding parameters, and the only way to see that is if both say so in the
    same words.

    Built here, once, because two copies had already stopped agreeing about the
    model digest and about whether to name the overlay setting at all.  Neither
    disagreement raised anything; both were visible only to a reader who
    compared two strings in two files.

    It does *not* check that the two reporters are the same one: the focus
    round is deliberately allowed to run a deterministic mock against a report
    a visual model wrote, so that the whole path stays CPU-executable.  Making
    the strings comparable is what this function is for; deciding whether they
    must match is a question for the caller, and for the mock path the answer
    is no.
    """
    adapter_identity = path_identity(adapter) if adapter is not None else "none"
    return (
        f"model={model}@{path_identity(model)}:"
        f"adapter={adapter or 'none'}@{adapter_identity}:"
        f"tokens={maximum_new_tokens}:events={maximum_events}:"
        f"overlay={overlay_timestamps}"
    )


def _is_synthetic(identity: Any) -> bool:
    """Whether an identity names a mock rather than a model.

    A naming convention, not a verified property: the mock builders in
    ``scripts/`` all call themselves ``deterministic-mock*`` and the real ones
    name a model.  That is enough for the one decision this feeds -- see
    :func:`reuse_decision` -- and it is deliberately a *class* rather than a
    comparison, because two mocks or two models differ for ordinary reasons
    (rules changed, weights re-downloaded) while the two classes differ for a
    reason that is not about the code at all.
    """
    return isinstance(identity, str) and identity.startswith("deterministic-mock")


def reuse_decision(
    run: Mapping[str, Any],
    *,
    subject: str,
    identities: Mapping[str, str],
    inputs: Mapping[str, tuple[str, str]],
    destination: Path,
    allow_downgrade: bool = False,
) -> str | None:
    """Whether a finished artifact may be reused, and if not, why not.

    ``None`` means reuse it.  A string means rebuild it and say so.  A
    ``ProvenanceError`` means **stop** -- the artifact is not reusable *and*
    replacing it would lose something this run cannot put back.

    Reuse is keyed on the inputs *and* on the producer.  An artifact is a
    function of both, so a change to the rules leaves every input digest intact
    while every artifact on disk becomes wrong -- and a check that covers only
    the inputs turns ``--resume`` into "serve the previous version".

    ``identities`` maps each recorded key to this run's value for it, and it is
    a mapping rather than a pair because an artifact can have more than one
    producer.  A focus plan has two: the rules that placed its windows, and the
    model that wrote its ``evidence_need`` sentence.  Checking one of them is
    how a stage comes to *record* that a mock wrote something and then compare
    only the string that is the same either way -- the plan a mock wrote and
    the plan a model wrote named the same planner, so a mock ``--resume``
    replaced 27 of them and nothing on disk disagreed.

    The exception is the point.  Rebuilding through ``--resume`` is correct
    whenever the difference is a change of rules, and it is not when the
    difference is that this run would *replace a model's work with a mock's*:
    a corpus regenerated that way is well formed in every field, passes every
    provenance check, and is indistinguishable afterwards from one that was
    never observed by a model at all.  That has happened here -- a
    ``--mock --resume`` over a corpus whose states had been built by a local
    Qwen3-14B rebuilt all 27 of them as mock, and the artifacts on disk were
    the only copy.  Reuse decisions are cheap to redo and expensive to get
    wrong in that direction, so the direction is refused by name and the run
    stops before it writes anything.  ``allow_downgrade`` is the operator
    saying they meant it.

    ``inputs`` maps each recorded digest key to ``(label, digest now)``; the
    label is what the reason names, because the key is a field name and the
    reason is read by a person.
    """
    for identity_key, identity in identities.items():
        recorded_identity = run.get(identity_key)
        if _is_synthetic(recorded_identity) or not _is_synthetic(identity):
            continue
        if allow_downgrade:
            continue
        raise ProvenanceError(
            f"{destination} was built by {recorded_identity!r} "
            f"({identity_key}) and this run would rebuild it with "
            f"{identity!r}, which produces synthetic evidence rather than a "
            f"model's: the corpus would stop being an observation and nothing "
            f"on disk would say so. Pass --allow-builder-downgrade if that is "
            f"what you mean."
        )
    for key, (label, digest) in inputs.items():
        if run.get(key) != digest:
            return f"{label} changed"
    for identity_key, identity in identities.items():
        recorded_identity = run.get(identity_key)
        if recorded_identity != identity:
            return (
                f"{subject} records {identity_key}={recorded_identity!r}; "
                f"this run uses {identity!r}"
            )
    return None


def verify_recorded_input(
    run: Mapping[str, Any],
    *,
    prefix: str,
    path: Path,
) -> None:
    """The file in hand must hold the bytes ``run`` recorded for ``prefix``.

    The comparison is on content, not on location, so an outputs tree that was
    moved intact is still verified while a file edited in place -- or a run
    pointed at a different sample's file -- fails.  The message names both the
    path the artifact recorded and the path in hand, because "the digest does
    not match" is not actionable on its own.
    """
    recorded_path = run.get(f"{prefix}_source")
    recorded_digest = run.get(f"{prefix}_sha256")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ProvenanceError(
            f"the artifact records no {prefix} source: its run block has no "
            f"{prefix}_source, so there is nothing to check {path} against"
        )
    if not isinstance(recorded_digest, str) or not recorded_digest:
        raise ProvenanceError(
            f"the artifact records no {prefix} digest: its run block names "
            f"{recorded_path} but carries no {prefix}_sha256, so {path} cannot "
            f"be checked against it"
        )
    if not path.is_file():
        raise ProvenanceError(
            f"the artifact was made from {prefix} {recorded_path}, "
            f"which is not readable at {path}"
        )
    actual = file_digest(path)
    if actual != recorded_digest:
        raise ProvenanceError(
            f"{path} is not the {prefix} this artifact was made from: it digests to "
            f"{actual[:16]}, while the artifact recorded {recorded_digest[:16]} "
            f"for {recorded_path}"
        )
