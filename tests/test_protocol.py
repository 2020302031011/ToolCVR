from pathlib import Path

from toolcvr.protocol import build_inference_protocol
from toolcvr.schemas import VideoInput


def test_protocol_exposes_synchronized_clock_and_adapter_alias_semantics(tmp_path: Path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.touch()
    second.touch()
    notes = build_inference_protocol(
        question="How many cycles does A1 complete?",
        options=("one", "two"),
        videos=(
            VideoInput("v1", first, synchronization_group="scene-7"),
            VideoInput("v2", second, synchronization_group="scene-7"),
        ),
        adapter_notes=("A1 and B1 denote the same physical track.",),
    )
    text = " ".join(notes)
    assert "share clock scene-7" in text
    assert "A1 and B1" in text
    assert "Count physical entities or transitions" in text
    assert "exactly one option label" in text


def test_protocol_has_no_task_label_input_or_task_routing(tmp_path: Path) -> None:
    video = tmp_path / "same.mp4"
    video.touch()
    notes = build_inference_protocol(
        question="Order the inputs.", options=(),
        videos=(VideoInput("v1", video), VideoInput("v2", video)),
        answer_kind="sequence", required_answer_ids=("v1", "v2"),
    )
    assert any("one source asset" in item for item in notes)
    assert any("exactly these input video IDs once each" in item for item in notes)


def test_protocol_renders_structural_sequence_and_interval_contracts(tmp_path: Path) -> None:
    video = tmp_path / "same.mp4"
    video.touch()
    sequence = build_inference_protocol(
        question="Order the inputs.", options=(),
        videos=(VideoInput("v1", video), VideoInput("v2", video)),
        answer_kind="sequence", required_answer_ids=("v1", "v2"),
    )
    assert any("v1, v2" in item and "Slot IDs" in item for item in sequence)
    interval = build_inference_protocol(
        question="Find the interval.", options=(),
        videos=(VideoInput("v1", video),), answer_kind="interval",
    )
    assert any("[start_seconds, end_seconds]" in item for item in interval)


def test_local_clip_clocks_hide_shared_source_order(tmp_path: Path) -> None:
    video = tmp_path / "same.mp4"
    video.touch()
    notes = build_inference_protocol(
        question="Order the inputs.", options=(),
        videos=(
            VideoInput("v1", video, ((40.0, 50.0),), model_clock="local"),
            VideoInput("v2", video, ((5.0, 15.0),), model_clock="local"),
        ),
        answer_kind="sequence", required_answer_ids=("v1", "v2"),
    )
    text = " ".join(notes)
    assert "clip-local clocks" in text
    assert "not comparable" in text
    assert "declared source interval" not in text
