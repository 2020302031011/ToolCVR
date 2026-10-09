"""The two focus entry points, at the seams the library cannot test.

``plan_focus.py`` and ``execute_focus_plans.py`` are scripts rather than
modules, so nothing imported them and nothing tested them -- which is where the
last two defects lived: a stale plan refused by a bare ``KeyError`` raised in
the middle of a batch, after the line saying it was regenerating, and a plan
executed against whichever report the caller happened to point at.  Loading
them by path is enough to reach the parts that are functions of their
arguments, and those are the parts that decide what a run does to a corpus.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from toolcvr.pipeline import WarmStartConfig
from toolcvr.provenance import (
    MOCK_REPORTER_COMPATIBLE_IDENTITIES,
    MOCK_REPORTER_IDENTITY,
    ProvenanceError,
    file_digest,
    path_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> Any:
    """The script as a module, without it having to be importable.

    The scripts put ``src`` on ``sys.path`` themselves, so this needs nothing
    but the path to the file.
    """
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan_payload(planner_identity: str, look: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": {"planner_identity": planner_identity},
        "plan": {
            "issue_id": "U01",
            "issue_class": "uncited",
            "evidence_need": "Which video blends the herbs?",
            "looks": [look],
            "bundle_videos": ["v1"],
            "unanchorable_videos": [],
            "evidence_need_source": "candidate_request",
            "evidence_need_note": "",
            "request_target_videos": ["v1"],
        },
    }


def test_a_plan_from_an_older_planner_shape_is_refused_by_name() -> None:
    """The refusal that replaces ``KeyError: 'window_clipped_to_medium'``.

    A plan written before the field existed used to die inside a generator
    expression, with nothing in the message saying the file was older than the
    code -- and only after the run had printed that it was regenerating that
    sample.  Nothing was written, because the rebuild happens before the write,
    but a run that stops mid-batch with a ``KeyError`` reads as a run that
    partially succeeded.  Refusing by name costs the same and says which file,
    which planner wrote it, and what to do.
    """

    module = _load("execute_focus_plans")
    look = {
        "video_id": "v1",
        "time_window": [110.0, 132.0],
        "anchor_event_ids": ["v1-e1"],
        "anchor_source": "state_citation",
        "dropped_anchor_event_ids": [],
        "window_budget_exceeded": False,
    }
    stale = _plan_payload("focus-planner-v3-multi-video-bundle", dict(look))

    try:
        module._rebuild_plan(stale, source=Path("/plans/CC-474/focus_plan.json"))
    except ValueError as exc:
        message = str(exc)
        assert "window_clipped_to_medium" in message
        assert "focus-planner-v3-multi-video-bundle" in message
        assert "CC-474" in message
    else:
        raise AssertionError("a plan written by the previous planner was executed")

    # The same payload with the field present still builds, so the check
    # refuses the old shape rather than everything.
    current = _plan_payload("focus-planner-v4-window-in-medium", dict(look))
    current["plan"]["looks"][0]["window_clipped_to_medium"] = True
    plan = module._rebuild_plan(current, source=Path("/plans/CC-474/focus_plan.json"))
    assert plan.looks[0].request.video_id == "v1"
    assert plan.looks[0].window_clipped_to_medium is True


def test_a_plan_planned_from_another_report_is_refused(tmp_path) -> None:
    """The hole that made a wrong run look like a right one.

    The plan names the report its state came from and the executor reads a
    report for the question: point the second at a different sample's file and
    every window still resolves, every event still gets reported, and the
    bundle's provenance is a lie that nothing in it records.
    """

    module = _load("execute_focus_plans")
    planned = tmp_path / "planned.json"
    planned.write_text('{"question": "which video blends?"}', encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text('{"question": "which video chops?"}', encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text('{"run": {}}', encoding="utf-8")
    run = {
        "report_source": str(planned),
        "report_sha256": file_digest(planned),
        "state_source": str(state),
        "state_sha256": file_digest(state),
    }

    module._verify_plan_inputs(
        run, plan_path=tmp_path / "focus_plan.json", report_path=planned
    )
    try:
        module._verify_plan_inputs(
            run, plan_path=tmp_path / "focus_plan.json", report_path=other
        )
    except ProvenanceError as exc:
        assert str(other) in str(exc)
    else:
        raise AssertionError("a plan was executed against another sample's report")


def test_a_plan_whose_state_was_rebuilt_is_refused(tmp_path) -> None:
    """The strongest link available: the anchors came from that state.

    Every window in a plan is inherited from a citation in the state, so a
    state that has been rebuilt since holds evidence the plan never saw.  The
    executor does not read the state, so the digest is the only way it can tell
    -- which is why it is checked rather than taken on trust.
    """

    module = _load("execute_focus_plans")
    report = tmp_path / "report.json"
    report.write_text('{"question": "which video blends?"}', encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text('{"unresolved_issues": [{"id": "U01"}]}', encoding="utf-8")
    run = {
        "report_source": str(report),
        "report_sha256": file_digest(report),
        "state_source": str(state),
        "state_sha256": file_digest(state),
    }
    module._verify_plan_inputs(run, plan_path=tmp_path / "p.json", report_path=report)

    state.write_text('{"unresolved_issues": []}', encoding="utf-8")
    try:
        module._verify_plan_inputs(run, plan_path=tmp_path / "p.json", report_path=report)
    except ProvenanceError as exc:
        assert str(state) in str(exc)
    else:
        raise AssertionError("a plan was executed against a state it was not planned from")


def test_a_plan_that_names_no_state_is_refused(tmp_path) -> None:
    module = _load("execute_focus_plans")
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    run = {"report_source": str(report), "report_sha256": file_digest(report)}
    try:
        module._verify_plan_inputs(run, plan_path=tmp_path / "p.json", report_path=report)
    except ProvenanceError as exc:
        assert "state_source" in str(exc)
    else:
        raise AssertionError("a plan that names no state verified")


def test_the_state_must_name_the_report_the_run_reads(tmp_path) -> None:
    """plan_focus's half of the chain, and the reason it is checked there.

    Planning reads both artifacts at once, so a state paired with another
    observation's report produces a plan whose windows cite events the question
    never mentioned.  Reproduced here with the pairing swapped, which is what a
    run against a mixed-up outputs tree looks like.
    """

    module = _load("plan_focus")
    built_from = tmp_path / "built-from.json"
    built_from.write_text('{"question": "A"}', encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text('{"question": "B"}', encoding="utf-8")
    state_payload = {
        "run": {
            "report_source": str(built_from),
            "report_sha256": file_digest(built_from),
        }
    }

    module._verify_report_inputs(state_payload, report_path=built_from)
    try:
        module._verify_report_inputs(state_payload, report_path=other)
    except ProvenanceError as exc:
        assert str(other) in str(exc)
    else:
        raise AssertionError("a state was planned from another sample's report")


def test_a_report_without_a_usable_duration_fails_loudly() -> None:
    """The number every window bound rests on.

    A missing duration is not "unknown, so skip the bound": the planner cannot
    check a window against a medium it has no length for, and defaulting one
    would put a made-up fact underneath every window in the plan.
    """

    module = _load("plan_focus")
    try:
        module._durations_from_payload(
            {"videos": [{"video_id": "v1", "duration_seconds": 133.12}, {"video_id": "v2"}]},
            Path("report.json"),
        )
    except ValueError as exc:
        assert "v2" in str(exc)
    else:
        raise AssertionError("a report with no duration for a video was planned anyway")

    durations = module._durations_from_payload(
        {"videos": [{"video_id": "v1", "duration_seconds": 133.12}]},
        Path("report.json"),
    )
    assert durations == {"v1": 133.12}


class _RecordingReporter:
    """Stands in for the visual model, which no test can afford to load."""

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        self.kwargs = kwargs


def test_both_entry_points_name_the_same_reporter_in_the_same_words(tmp_path) -> None:
    """The string two stages write beside each other has to be comparable.

    Every evidence bundle records ``reporter_identity`` and
    ``observation_reporter_identity`` next to one another, and comparing them
    is the only thing either field is for.  Two copies of the string-building
    had stopped agreeing -- different JSON separators, so the same model
    directory named two different digests, and one copy named no overlay
    setting at all -- and neither disagreement raised anything.  So this drives
    the real ``_reporter`` of both scripts, with the model load stubbed out,
    and asks whether a run given the same parameters says the same thing.
    """

    reports = _load("generate_reports_from_selection")
    executor = _load("execute_focus_plans")

    model = tmp_path / "Qwen-VL"
    model.mkdir()
    (model / "config.json").write_bytes(b'{"arch": "qwen3vl"}')

    observation_builder = reports._reporter
    focus_builder = executor._reporter
    real_reporter_class = reports.LocalQwen3VLReporter
    reports.LocalQwen3VLReporter = _RecordingReporter  # type: ignore[assignment]
    executor.LocalQwen3VLReporter = _RecordingReporter  # type: ignore[assignment]
    try:
        observation = observation_builder(
            argparse.Namespace(
                mock_reporter=False,
                vlm_model=model,
                vlm_adapter=None,
                vlm_device="cpu",
                max_vlm_tokens=2048,
                maximum_events_per_batch=8,
            ),
            WarmStartConfig(),
        )
        focus = focus_builder(
            argparse.Namespace(
                mock_reporter=False,
                vlm_model=model,
                vlm_adapter=None,
                vlm_device="cpu",
                max_vlm_tokens=2048,
                maximum_events_per_batch=8,
                overlay_timestamps=True,
            )
        )
    finally:
        reports.LocalQwen3VLReporter = real_reporter_class  # type: ignore[assignment]
        executor.LocalQwen3VLReporter = real_reporter_class  # type: ignore[assignment]

    # The same eyes, named the same way: not merely a shared suffix but the
    # whole identity, so a reader comparing the two fields in a bundle finds
    # them equal when the two stages really did use the same reporter.
    assert observation[1] == focus[1]
    assert observation[1] == (
        f"local-qwen3vl-v6-frame-manifest:"
        f"model={model}@{path_identity(model)}:adapter=none@none:"
        f"tokens=2048:events=8:overlay=True"
    )
    # And built the same way, which is what the name is a claim about: the
    # frames a focus round sends are prepared by the reporter, so two runs that
    # name the same reporter must configure it the same.
    assert observation[0].kwargs == focus[0].kwargs


def test_a_focus_round_that_will_not_overlay_says_so(tmp_path) -> None:
    """A divergence from the observation must be readable, not silent.

    ``overlay`` was the field one of the two copies left out entirely, so a
    run that decoded frames differently from the observation it was planned
    from produced an identity that could not be told apart from a matching one.
    """

    executor = _load("execute_focus_plans")
    model = tmp_path / "Qwen-VL"
    model.mkdir()
    (model / "config.json").write_bytes(b'{"arch": "qwen3vl"}')

    def _identity(overlay: bool) -> tuple[str, dict[str, Any]]:
        real = executor.LocalQwen3VLReporter
        executor.LocalQwen3VLReporter = _RecordingReporter  # type: ignore[assignment]
        try:
            reporter, identity = executor._reporter(
                argparse.Namespace(
                    mock_reporter=False,
                    vlm_model=model,
                    vlm_adapter=None,
                    vlm_device="cpu",
                    max_vlm_tokens=2048,
                    maximum_events_per_batch=8,
                    overlay_timestamps=overlay,
                )
            )
        finally:
            executor.LocalQwen3VLReporter = real  # type: ignore[assignment]
        return identity, reporter.kwargs

    overlaid, overlaid_kwargs = _identity(True)
    plain, plain_kwargs = _identity(False)
    assert overlaid != plain
    assert "overlay=True" in overlaid and "overlay=False" in plain
    assert overlaid_kwargs["overlay_timestamps"] is True
    assert plain_kwargs["overlay_timestamps"] is False


def test_both_entry_points_key_reuse_on_the_producer_as_well_as_the_input() -> None:
    """Resume is the one place a wrong answer costs nothing to notice.

    Both scripts ask the same question about their own artifact -- is the file
    on disk still the one this run would produce -- and both must answer with
    the producer's identity as well as the input's digest.  A rule or a
    reporter that changed leaves every digest intact.
    """

    planner = _load("plan_focus")
    executor = _load("execute_focus_plans")
    plan_path = Path("/plans/CC-474/focus_plan.json")
    evidence_path = Path("/evidence/CC-474/focus_evidence.json")
    ordinary = {"destination": plan_path, "allow_builder_downgrade": False}
    builder = "openai-compatible:Qwen3-14B"

    assert planner._reuse_decision(
        {
            "state_sha256": "d1",
            "planner_identity": planner.PLANNER_IDENTITY,
            "builder_identity": builder,
        },
        "d1",
        builder,
        **ordinary,
    ) is None
    assert (
        planner._reuse_decision(
            {"state_sha256": "d1", "planner_identity": "focus-planner-v3"},
            "d1",
            builder,
            **ordinary,
        )
        == "plan records planner_identity='focus-planner-v3'; this run uses "
        f"{planner.PLANNER_IDENTITY!r}"
    )
    assert planner._reuse_decision(
        {
            "state_sha256": "d2",
            "planner_identity": planner.PLANNER_IDENTITY,
            "builder_identity": builder,
        },
        "d1",
        builder,
        **ordinary,
    ) == "the initial state changed"

    executor_args = {
        "destination": evidence_path,
        "allow_builder_downgrade": False,
    }
    assert executor._reuse_decision(
        {
            "plan_sha256": "p1",
            "reporter_identity": executor.MOCK_REPORTER_IDENTITY,
        },
        "p1",
        executor.MOCK_REPORTER_IDENTITY,
        **executor_args,
    ) is None
    assert (
        executor._reuse_decision({"plan_sha256": "p2"}, "p1", "reporter", **executor_args)
        == "the plan changed"
    )
    mock_v4 = "deterministic-mock-event-reporter-v4-assessment"
    mock_v5 = "deterministic-mock-event-reporter-v5-assessment"
    assert (
        executor._reuse_decision(
            {"plan_sha256": "p1", "reporter_identity": mock_v4}, "p1", mock_v5, **executor_args
        )
        == f"evidence records reporter_identity={mock_v4!r}; this run uses {mock_v5!r}"
    )


def test_the_two_mock_reporters_answer_to_one_name() -> None:
    """The other string the two files had a copy of each.

    Same pair of files as the drifted model fingerprint, same shape of defect:
    a name written into artifacts by two stages, compared by a reader, spelled
    out twice.  It agreed on the day it was written and nothing kept it
    agreeing.  Unlike the model path, both mock branches build without loading
    anything, so this compares what the scripts really produce.
    """

    reports = _load("generate_reports_from_selection")
    executor = _load("execute_focus_plans")

    observation = reports._reporter(
        argparse.Namespace(mock_reporter=True, vlm_model=None, vlm_adapter=None),
        WarmStartConfig(),
    )
    focus = executor._reporter(
        argparse.Namespace(mock_reporter=True, vlm_model=None, vlm_adapter=None)
    )

    assert observation[1] == focus[1] == MOCK_REPORTER_IDENTITY
    # And the observation's list of names it will still reuse under is the
    # shared one, so a mock report from the previous version is not rebuilt
    # into the current one for no reason.
    assert observation[2] == MOCK_REPORTER_COMPATIBLE_IDENTITIES


def test_a_mock_resume_stops_before_it_replaces_a_models_artifact(tmp_path) -> None:
    """The incident, reproduced against the wrappers each script actually calls.

    ``tests/test_provenance.py`` pins the rule.  This pins that the rule is
    *wired in*: the two scripts that overwrote a Qwen3-14B corpus must refuse
    the same run that did it, name the file, and mention the flag, rather than
    returning a message that the batch loop prints as "regenerating" on its way
    to the write.
    """

    real_builder = "openai-compatible-v3-anchored-local-summary:Qwen3-14B"
    mock_builder = "deterministic-mock-v2-local-assessment"

    builder = _load("build_initial_state")
    plan_focus = _load("plan_focus")

    # Two stages, and the second one is the pair the incident actually hit:
    # the plan a mock wrote named the same *rules* as the plan a model wrote,
    # so the field that could tell them apart was the sentence-writer's.
    for module, subject, on_disk, this_run in (
        (
            builder,
            "state",
            {"builder_identity": real_builder},
            {"builder_identity": mock_builder},
        ),
        (
            plan_focus,
            "plan",
            {
                "planner_identity": plan_focus.PLANNER_IDENTITY,
                "builder_identity": real_builder,
            },
            {
                "planner_identity": plan_focus.PLANNER_IDENTITY,
                "builder_identity": "deterministic-mock",
            },
        ),
    ):
        destination = tmp_path / f"{subject}.json"
        try:
            module.reuse_decision(
                on_disk,
                subject=subject,
                identities=this_run,
                inputs={},
                destination=destination,
                allow_downgrade=False,
            )
        except ProvenanceError as exc:
            assert str(destination) in str(exc)
            assert "--allow-builder-downgrade" in str(exc)
        else:
            raise AssertionError(f"{module.__name__} let a mock run overwrite a model's {subject}")

    # Every entry point that asks this question offers the way through, so the
    # refusal is a stop rather than a dead end.
    for name in ("build_initial_state", "plan_focus", "execute_focus_plans"):
        parsed = _load(name)._parser().parse_args(
            [
                "--resume",
                "--allow-builder-downgrade",
                *(
                    ["--reports-dir", str(tmp_path), "--output-dir", str(tmp_path)]
                    if name == "build_initial_state"
                    else []
                ),
                *(
                    [
                        "--state-dir",
                        str(tmp_path),
                        "--reports-dir",
                        str(tmp_path),
                        "--output-dir",
                        str(tmp_path),
                    ]
                    if name == "plan_focus"
                    else []
                ),
                *(
                    ["--plan-dir", str(tmp_path), "--reports-dir", str(tmp_path), "--output-dir", str(tmp_path)]
                    if name == "execute_focus_plans"
                    else []
                ),
            ]
        )
        assert parsed.allow_builder_downgrade is True, name
