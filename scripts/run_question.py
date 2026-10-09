#!/usr/bin/env python3
"""Run CrossVid questions through the whole local stack, in one command.

    python scripts/run_question.py --sample-id CC-117 --cuda-device 0
    python scripts/run_question.py --sample-id PSS-62,CC-117,FSA-21 --cuda-device 0

``--sample-id`` takes one id or a comma-separated list; a list is the reason to
use this script rather than the stage scripts directly, because the language
model is loaded once for the whole list and only the per-sample stages repeat.

Four stages, all local, no paid API:

    0  CLIP selection   audit_crossvid_clip_filter.py   (reused when already frozen)
    1  VLM reports      generate_reports_from_selection.py
    2  initial state    build_initial_state.py
    3  legacy focus plan diagnostic        plan_focus.py

This command intentionally stops after the focus plan. Stage 3 emits one
*plan* -- a bundle holding one look per video the issue covers -- and nothing
here executes it; the executor (``toolcvr/focus_execution.py``) and the
multi-round agent-owned tool loops are not wired into this entry point yet.

The language model is served as an OpenAI-compatible endpoint by
``local_llm_server.py`` (transformers, no vLLM needed); stages 2 and 3 talk to
it, stage 1 does not.  The VLM is loaded in-process, so it needs no server of
its own.  Everything the run produced lands under ``--out-root``.

Device policy
-------------
``--cuda-device`` is the single knob and applies to the server, the VLM and
CLIP.  The three can be split with ``--llm-device`` / ``--vlm-device`` /
``--clip-device`` when they do not all fit on one card.  The server is started
once and kept up for the whole run, so by default the VLM has to share a device
with whatever the server holds; if that does not fit, pass ``--sequential`` and
the server is stopped around the VLM stage instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SELECTION_DIR = PROJECT_ROOT / "outputs" / "crossvid-9task-clip-audit-unified-v1"
DEFAULT_CROSSVID_ROOT = Path("/media/data6/xuejj/CrossVid")
DEFAULT_VLM_MODEL = Path("/media/data6/xuejj/Qwen-VL")
DEFAULT_LLM_MODEL = Path("/media/data6/xuejj/Qwen3-14B")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample-id",
        "--sample-ids",
        dest="sample_ids",
        required=True,
        help=(
            "one id or a comma-separated list, e.g. CC-117, or "
            "PSS-62,CC-117,FSA-21 for one question from each task"
        ),
    )
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="physical CUDA index used by the server, the VLM and CLIP unless overridden",
    )
    parser.add_argument("--llm-device", help="physical CUDA index(es) for vLLM, e.g. 0 or 0,1")
    parser.add_argument("--vlm-device", help="torch device string for the VLM, e.g. cuda:0 or cpu")
    parser.add_argument("--clip-device", help="torch device string for CLIP, e.g. cuda:0")

    parser.add_argument("--llm-model", type=Path, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--vlm-model", type=Path, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-adapter", type=Path)
    parser.add_argument(
        "--server",
        choices=("transformers", "vllm"),
        default="transformers",
        help="how to serve the language model; transformers needs no extra packages",
    )
    parser.add_argument("--tensor-parallel-size", type=int, help="vLLM only")
    parser.add_argument(
        "--server-max-tokens",
        type=int,
        default=2048,
        help="default completion cap when a caller does not set one",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="let a hybrid reasoning model emit its thinking block",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--server-timeout-seconds",
        type=float,
        default=900.0,
        help="how long to wait for the vLLM endpoint to answer",
    )
    parser.add_argument(
        "--server-command",
        help="override the server launch, e.g. 'python -m sglang.launch_server ...'",
    )
    parser.add_argument("--keep-server", action="store_true", help="leave the server running on exit")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="stop the server around the VLM stage when both do not fit on one device",
    )

    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--crossvid-root", type=Path, default=DEFAULT_CROSSVID_ROOT)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "one-shot")
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument(
        "--force-selection",
        action="store_true",
        help="recompute CLIP selection even when a frozen result already exists",
    )
    parser.add_argument("--skip-selection", action="store_true")

    parser.add_argument("--mock-vlm", action="store_true", help="stage 1 without the visual model")
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="stages 2 and 3 without a language model; also skips starting the server",
    )
    parser.add_argument("--maximum-events-per-batch", type=int)
    parser.add_argument("--max-vlm-tokens", type=int)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    return parser


def _die(message: str) -> NoReturn:
    print(f"\n[fail] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _stage(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}", flush=True)


def _run(command: list[str], *, stage: str, env: dict[str, str] | None = None) -> None:
    print(f"[cmd] {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)
    if completed.returncode != 0:
        _die(f"{stage} failed with exit code {completed.returncode}")


def _server_command(args: argparse.Namespace, devices: str) -> list[str]:
    if args.server_command:
        return shlex.split(args.server_command)
    if args.server == "transformers":
        # ``CUDA_VISIBLE_DEVICES`` is already set, so the index here is relative
        # to the visible set and ``cuda:0`` means "the first device I was given".
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "local_llm_server.py"),
            "--model",
            str(args.llm_model),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--device",
            "cuda:0" if devices != "cpu" else "cpu",
            "--default-max-tokens",
            str(args.server_max_tokens),
        ]
        if args.enable_thinking:
            command.append("--enable-thinking")
        return command
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.llm_model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        "0.90",
        "--trust-remote-code",
    ]
    if args.tensor_parallel_size:
        command += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    elif "," in devices:
        command += ["--tensor-parallel-size", str(len(devices.split(",")))]
    return command


def _wait_for_server(port: int, timeout_seconds: float, process: subprocess.Popen) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_seconds
    announced = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _die(f"the language model server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                if response.status == 200:
                    print(f"[server] ready at {url}", flush=True)
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            if not announced:
                print(f"[server] waiting for {url} ...", flush=True)
                announced = True
        time.sleep(3.0)
    _die(f"the language model server did not answer within {timeout_seconds:.0f}s")


class _Server:
    """The vLLM process, started once and stopped in a finally block."""

    def __init__(self, args: argparse.Namespace, devices: str) -> None:
        self.args = args
        self.devices = devices
        self.process: subprocess.Popen | None = None
        self.env: dict[str, str] = {}

    def start(self) -> None:
        if self.process is not None:
            return
        self.env = dict(os.environ)
        self.env["CUDA_VISIBLE_DEVICES"] = self.devices
        self.env.setdefault(self.args.api_key_env, "local")
        command = _server_command(self.args, self.devices)
        print(f"[server] CUDA_VISIBLE_DEVICES={self.devices}", flush=True)
        print(f"[cmd] {shlex.join(command)}", flush=True)
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=self.env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )
        _wait_for_server(self.args.port, self.args.server_timeout_seconds, self.process)

    def stop(self) -> None:
        if self.process is None:
            return
        print("[server] stopping", flush=True)
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            self.process = None
            # vLLM leaves worker processes behind if only the parent is signalled.
            time.sleep(5.0)


def _task_of(sample_id: str) -> str:
    return sample_id.rsplit("-", 1)[0]


def _normalize_sample_id(raw: str) -> str:
    """Upper-case the task prefix, leaving the sample number alone.

    The selection tree spells the task in capitals, so without this
    ``--sample-id pss-62`` would look for a directory that cannot exist and
    report a missing selection instead of an obvious typo.
    """
    body = raw.strip()
    if "-" not in body:
        return body
    task = _task_of(body)
    return task.upper() + body[len(task) :]


def _parse_sample_ids(raw: str) -> tuple[str, ...]:
    """Split ``--sample-id`` into ids, keeping the given order and dropping repeats.

    Order matters because it is the order the samples are printed in, and the
    ids are used verbatim in every stage command, so a typo has to be caught
    here rather than after a model has been loaded.
    """
    sample_ids = tuple(
        dict.fromkeys(_normalize_sample_id(item) for item in raw.split(",") if item.strip())
    )
    if not sample_ids:
        _die("--sample-id was given no ids")
    for sample_id in sample_ids:
        if "-" not in sample_id or not _task_of(sample_id):
            _die(f"--sample-id must look like TASK-NUMBER, got {sample_id!r}")

    # A mistyped task is worth catching here rather than in the CLIP audit,
    # which is the first stage that would reject it -- by then the language
    # model has already been loaded, which on CPU is minutes of waiting for a
    # typo.
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from toolcvr.crossvid import SUPPORTED_CROSSVID_TASKS

    unknown = sorted(
        {_task_of(item).upper() for item in sample_ids} - set(SUPPORTED_CROSSVID_TASKS)
    )
    if unknown:
        _die(
            f"unknown task(s) {unknown}; supported: {list(SUPPORTED_CROSSVID_TASKS)}"
        )
    return sample_ids


def _dotenv(path: Path, base_url: str, model: str, key_env: str) -> None:
    path.write_text(
        f"LLM_BASE_URL={base_url}\nLLM_MODEL={model}\n{key_env}=local\n",
        encoding="utf-8",
    )


def _selection_exists(selection_dir: Path, task: str, sample_id: str) -> bool:
    return (selection_dir / task / sample_id / "result.json").is_file()


def _samples_with_artifact(
    root: Path,
    sample_ids: tuple[str, ...],
    filename: str,
) -> tuple[str, ...]:
    """Return requested samples whose preceding stage completed atomically."""
    return tuple(
        sample_id
        for sample_id in sample_ids
        if (root / _task_of(sample_id) / sample_id / filename).is_file()
    )


def _announce_stage_failures(
    stage: str,
    requested: tuple[str, ...],
    completed: tuple[str, ...],
) -> None:
    missing = [sample_id for sample_id in requested if sample_id not in completed]
    if missing:
        print(
            f"[{stage}] incomplete={','.join(missing)}; continuing with "
            f"{len(completed)}/{len(requested)} successful sample(s)",
            flush=True,
        )


def main() -> int:
    args = _parser().parse_args()
    sample_ids = _parse_sample_ids(args.sample_ids)
    tasks = tuple(dict.fromkeys(_task_of(item) for item in sample_ids))
    joined_samples = ",".join(sample_ids)
    joined_tasks = ",".join(tasks)

    llm_devices = args.llm_device or args.cuda_device
    vlm_device = args.vlm_device or f"cuda:{llm_devices.split(',')[0]}"
    clip_device = args.clip_device or f"cuda:{llm_devices.split(',')[0]}"

    out_root = args.out_root.expanduser().resolve()
    reports_root = out_root / "reports"
    state_root = out_root / "state"
    plan_root = out_root / "plan"
    selection_root = args.selection_dir.expanduser().resolve()
    base_url = f"http://127.0.0.1:{args.port}/v1"
    model_name = args.llm_model.name

    print(f"[run] samples={joined_samples} ({len(sample_ids)} over tasks {joined_tasks})")
    print(f"[run] llm_device={llm_devices} vlm_device={vlm_device} clip_device={clip_device}")
    print(f"[run] llm_model={args.llm_model}")
    print(f"[run] vlm_model={args.vlm_model}")
    print(f"[run] output={out_root}")

    if not args.mock_llm and not args.mock_vlm and vlm_device.startswith("cuda"):
        if vlm_device == f"cuda:{llm_devices.split(',')[0]}":
            print(
                "[warn] the VLM shares a device with the language model server. If this "
                "runs out of memory, pass --vlm-device cuda:1 or --sequential.",
                flush=True,
            )

    env = dict(os.environ)
    env.setdefault(args.api_key_env, "local")
    env["LLM_BASE_URL"] = base_url
    env["LLM_MODEL"] = model_name

    server = _Server(args, llm_devices)
    dotenv_path = Path(tempfile.mkdtemp(prefix="toolcvr-run-")) / "llm.env"
    _dotenv(dotenv_path, base_url, model_name, args.api_key_env)

    try:
        if not args.mock_llm:
            _stage("stage 0/3  starting the local language model server")
            server.start()
        else:
            _stage("stage 0/3  language model server skipped (--mock-llm)")

        # ── CLIP selection ───────────────────────────────────────────
        _stage("stage 1/4  CLIP selection")
        if args.skip_selection:
            print("[skip] --skip-selection", flush=True)
        else:
            absent = [
                sample_id
                for sample_id in sample_ids
                if not _selection_exists(selection_root, _task_of(sample_id), sample_id)
            ]
            if not absent and not args.force_selection:
                print(
                    f"[skip] frozen selection already present for all "
                    f"{len(sample_ids)} sample(s) under {selection_root}",
                    flush=True,
                )
                print("[skip] pass --force-selection to recompute it", flush=True)
            else:
                # Only the tasks that are actually missing a sample need the
                # audit; the ones already frozen keep their result.
                audit_tasks = tasks if args.force_selection or not absent else tuple(
                    dict.fromkeys(_task_of(item) for item in absent)
                )
                audit_samples = sample_ids if args.force_selection else tuple(absent)
                print(f"[clip] recomputing selection for tasks {','.join(audit_tasks)}", flush=True)
                audit_command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "audit_crossvid_clip_filter.py"),
                    "--tasks",
                    ",".join(audit_tasks),
                    "--sample-ids",
                    ",".join(audit_samples),
                    "--samples-per-task",
                    str(args.samples_per_task),
                    "--crossvid-root",
                    str(args.crossvid_root),
                    "--output-dir",
                    str(selection_root),
                    "--api-env-file",
                    str(dotenv_path),
                    "--api-base-url",
                    base_url,
                    "--api-model",
                    model_name,
                    "--api-key-env",
                    args.api_key_env,
                    "--clip-device",
                    clip_device,
                ]
                if not args.force_selection:
                    audit_command.append("--resume")
                _run(
                    audit_command,
                    stage="CLIP selection",
                    env=env,
                )
        still_absent = [
            sample_id
            for sample_id in sample_ids
            if not _selection_exists(selection_root, _task_of(sample_id), sample_id)
        ]
        if still_absent:
            _die(
                f"no selection for {','.join(still_absent)} under {selection_root} after "
                "stage 1. The CLIP audit samples per task by duration, so an untargetable "
                "id has to be produced by a full task run or supplied with --selection-dir."
            )

        # ── VLM reports ──────────────────────────────────────────────
        _stage("stage 2/4  VLM observer reports")
        if args.sequential and server.process is not None:
            server.stop()
            print("[server] stopped so the VLM can use the device on its own", flush=True)
        report_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_reports_from_selection.py"),
            "--selection-dir",
            str(selection_root),
            "--output-dir",
            str(reports_root),
            "--crossvid-root",
            str(args.crossvid_root),
            "--sample-ids",
            joined_samples,
            "--vlm-device",
            vlm_device,
            "--resume",
        ]
        # ``--mock-reporter`` refuses to be combined with a model or adapter path,
        # so those are only passed when the VLM is actually going to be loaded.
        if args.mock_vlm:
            report_command.append("--mock-reporter")
        else:
            report_command += ["--vlm-model", str(args.vlm_model)]
            if args.vlm_adapter:
                report_command += ["--vlm-adapter", str(args.vlm_adapter)]
        if args.maximum_events_per_batch is not None:
            report_command += ["--maximum-events-per-batch", str(args.maximum_events_per_batch)]
        if args.max_vlm_tokens is not None:
            report_command += ["--max-vlm-tokens", str(args.max_vlm_tokens)]
        _run(report_command, stage="VLM reports", env=env)

        report_sample_ids = _samples_with_artifact(reports_root, sample_ids, "report.json")
        _announce_stage_failures("reports", sample_ids, report_sample_ids)
        if not report_sample_ids:
            _die("VLM reporting produced no complete report; state formation cannot continue")

        if args.sequential and not args.mock_llm:
            server.start()

        # ── initial state ────────────────────────────────────────────
        _stage("stage 3/4  initial state formation")
        state_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_initial_state.py"),
            "--reports-dir",
            str(reports_root),
            "--output-dir",
            str(state_root),
            "--sample-ids",
            ",".join(report_sample_ids),
            "--resume",
        ]
        if args.mock_llm:
            state_command.append("--mock")
        else:
            state_command += [
                "--api-base-url",
                base_url,
                "--api-model",
                model_name,
                "--api-key-env",
                args.api_key_env,
            ]
        _run(state_command, stage="initial state formation", env=env)

        state_sample_ids = _samples_with_artifact(
            state_root, report_sample_ids, "initial_state.json"
        )
        _announce_stage_failures("state", report_sample_ids, state_sample_ids)
        if not state_sample_ids:
            _die("initial-state formation produced no complete state; planning cannot continue")

        # ── focus plan ───────────────────────────────────────────────
        _stage("stage 4/4  focus plan (one look per video in the issue's bundle)")
        plan_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "plan_focus.py"),
            "--state-dir",
            str(state_root),
            "--reports-dir",
            str(reports_root),
            "--output-dir",
            str(plan_root),
            "--sample-ids",
            ",".join(state_sample_ids),
            "--resume",
        ]
        if args.mock_llm:
            plan_command.append("--mock")
        else:
            plan_command += [
                "--api-base-url",
                base_url,
                "--api-model",
                model_name,
                "--api-key-env",
                args.api_key_env,
            ]
        _run(plan_command, stage="focus planning", env=env)

    finally:
        if not args.keep_server:
            server.stop()
        shutil.rmtree(dotenv_path.parent, ignore_errors=True)

    _stage("done")
    totals = {"planned": 0, "no_plan": 0, "missing": 0, "partial": 0}
    for sample_id in sample_ids:
        task = _task_of(sample_id)
        artifacts = {
            "selection": selection_root / task / sample_id / "result.json",
            "report": reports_root / task / sample_id / "report.json",
            "initial_state": state_root / task / sample_id / "initial_state.json",
            "focus_plan": plan_root / task / sample_id / "focus_plan.json",
        }
        print(f"\n[{sample_id}]", flush=True)
        for label, path in artifacts.items():
            mark = "ok " if path.is_file() else "-- "
            print(f"  {mark}{label:<14} {path}", flush=True)

        plan_path = artifacts["focus_plan"]
        if not plan_path.is_file():
            totals["missing"] += 1
            continue
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        plan = payload.get("plan")
        if plan is None:
            totals["no_plan"] += 1
            print("  [plan] no issue could be anchored; refusals:", flush=True)
            for refusal in payload.get("refusals", []):
                print(
                    f"    {refusal['issue_id']} ({refusal['issue_class']}): {refusal['reason']}",
                    flush=True,
                )
            continue
        totals["planned"] += 1
        looks = plan["looks"]
        print(
            f"  [plan] issue={plan['issue_id']} class={plan['issue_class']} "
            f"looks={len(looks)}",
            flush=True,
        )
        for look in looks:
            start, end = look["time_window"]
            print(
                f"    {look['video_id']} window=[{start:.1f}, {end:.1f}] "
                f"anchor={look['anchor_source']} "
                f"anchors={len(look['anchor_event_ids'])}"
                + (" [over budget]" if look["window_budget_exceeded"] else "")
                + (" [clipped to medium]" if look.get("window_clipped_to_medium") else ""),
                flush=True,
            )
        if plan.get("unanchorable_videos"):
            totals["partial"] += 1
            print(
                f"  [plan] partial: no window for {plan['unanchorable_videos']}",
                flush=True,
            )
        print(f"  [plan] evidence_need: {plan['evidence_need']}", flush=True)
        if payload.get("refusals"):
            print(f"  [plan] {len(payload['refusals'])} issue(s) passed over:", flush=True)
            for refusal in payload["refusals"]:
                print(f"    {refusal['issue_id']} ({refusal['issue_class']})", flush=True)

    print(
        f"\n[done] planned={totals['planned']} partial={totals['partial']} "
        f"no_plan={totals['no_plan']} "
        f"missing={totals['missing']} of {len(sample_ids)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
