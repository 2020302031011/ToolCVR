"""Crash-safe, append-only model-call traces for distillation and auditing."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from toolcvr.openai_compatible import APICallRecord


class JsonlTrajectoryWriter:
    """Persist one sample's complete API and lifecycle trace incrementally.

    Each invocation gets a session identifier.  Re-running an interrupted
    sample appends a new session instead of overwriting evidence from the
    failed attempt, which makes repair and preference-data construction
    auditable.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self._sequence = 0
        self._lock = threading.Lock()

    def append(self, event_type: str, payload: Mapping[str, object]) -> None:
        if not event_type.strip():
            raise ValueError("trajectory event type must be nonempty")
        with self._lock:
            self._sequence += 1
            record = {
                "schema_version": 1,
                "session_id": self.session_id,
                "sequence": self._sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "payload": dict(payload),
            }
            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def record_api_call(self, call: APICallRecord) -> None:
        payload = asdict(call)
        payload["usage"] = asdict(call.usage)
        self.append("model_call", payload)
