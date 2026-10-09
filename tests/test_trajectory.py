from __future__ import annotations

import json

from toolcvr.openai_compatible import APICallRecord, APIUsage
from toolcvr.trajectory import JsonlTrajectoryWriter


def test_jsonl_trajectory_writer_persists_complete_model_call(tmp_path) -> None:
    target = tmp_path / "trajectory.jsonl"
    writer = JsonlTrajectoryWriter(target)
    writer.append("run_start", {"sample_id": "CC-1"})
    writer.record_api_call(APICallRecord(
        role="reasoner_action",
        model="flash",
        visual_input_count=0,
        retry_count=1,
        usage=APIUsage(10, 4, 0),
        request_payload={"messages": [{"role": "user", "content": "context"}]},
        raw_response='{"action":{"kind":"ANSWER"}}',
        latency_seconds=1.25,
        transient_errors=("temporary",),
    ))

    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert [item["event_type"] for item in records] == ["run_start", "model_call"]
    assert records[1]["payload"]["request_payload"]["messages"][0]["content"] == "context"
    assert records[1]["payload"]["raw_response"].startswith("{")
    assert records[1]["payload"]["usage"]["input_tokens"] == 10
    assert records[0]["session_id"] == records[1]["session_id"]
