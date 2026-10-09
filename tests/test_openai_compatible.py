import json
import socket
import urllib.error
from unittest.mock import patch

import pytest

from toolcvr.openai_compatible import (
    APIRequestConfig,
    OpenAICompatibleTransport,
    StructuredOutputError,
    extract_json_object,
    redact,
)


class _Response:
    def __init__(self, body: object) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._body


def _body(content: object, *, prompt_tokens: int = 3) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 5,
            "prompt_tokens_details": {"image_tokens": 2},
        },
    }


def test_extract_json_object_accepts_fence_and_light_wrapper() -> None:
    assert extract_json_object("```json\n{\"ok\": true}\n```") == {"ok": True}
    assert extract_json_object("result follows: {\"value\": 4} trailing") == {"value": 4}
    with pytest.raises(StructuredOutputError):
        extract_json_object("[1, 2]")


def test_transport_normalizes_endpoint_content_and_usage(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    captured: dict[str, object] = {}

    def _urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response(_body([{"type": "text", "text": "{\"ok\":true}"}]))

    transport = OpenAICompatibleTransport(
        base_url="https://example.test/v1",
        model="flash",
        api_key_env="TEST_API_KEY",
    )
    with patch("urllib.request.urlopen", _urlopen):
        result = transport.complete(
            role="test",
            messages=[{"role": "user", "content": "hello"}],
            config=APIRequestConfig(maximum_tokens=64, timeout_seconds=7),
            visual_input_count=2,
        )

    assert result == '{"ok":true}'
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert transport.calls[0].visual_input_count == 2
    assert transport.calls[0].usage.visual_tokens == 2
    assert transport.calls[0].request_payload["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert transport.calls[0].latency_seconds >= 0


def test_transport_retries_transient_network_failure(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    outcomes = [urllib.error.URLError("temporary"), _Response(_body('{"ok":true}'))]

    def _urlopen(request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    transport = OpenAICompatibleTransport(
        base_url="https://example.test/v1",
        model="flash",
        api_key_env="TEST_API_KEY",
    )
    with patch("urllib.request.urlopen", _urlopen), patch("time.sleep"):
        result = transport.complete(
            role="test",
            messages=[{"role": "user", "content": "hello"}],
            config=APIRequestConfig(
                maximum_tokens=64,
                timeout_seconds=7,
                maximum_retries=1,
            ),
        )
    assert result == '{"ok":true}'
    assert transport.calls[0].retry_count == 1


def test_transport_retries_socket_timeout_on_python39(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    outcomes = [socket.timeout("read timed out"), _Response(_body('{"ok":true}'))]

    def _urlopen(request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    transport = OpenAICompatibleTransport(
        base_url="https://example.test/v1",
        model="flash",
        api_key_env="TEST_API_KEY",
    )
    with patch("urllib.request.urlopen", _urlopen), patch("time.sleep"):
        result = transport.complete(
            role="test",
            messages=[{"role": "user", "content": "hello"}],
            config=APIRequestConfig(
                maximum_tokens=64,
                timeout_seconds=7,
                maximum_retries=1,
            ),
        )
    assert result == '{"ok":true}'
    assert transport.calls[0].retry_count == 1


def test_structured_completion_repairs_only_the_invalid_structure(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    outcomes = [
        _Response(_body('{"ids":[["bad"]]}')),
        _Response(_body('{"ids":["good"]}')),
    ]

    def _urlopen(request, timeout):
        return outcomes.pop(0)

    def _validator(value: dict[str, object]) -> tuple[str, ...]:
        ids = value.get("ids")
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("ids must be a flat string list")
        return tuple(ids)

    transport = OpenAICompatibleTransport(
        base_url="https://example.test/v1",
        model="flash",
        api_key_env="TEST_API_KEY",
    )
    with patch("urllib.request.urlopen", _urlopen):
        result = transport.complete_structured(
            role="reasoner",
            messages=[{"role": "user", "content": "hello"}],
            config=APIRequestConfig(
                maximum_tokens=64,
                timeout_seconds=7,
                repair_retries=1,
            ),
            validator=_validator,
            schema_hint={"ids": ["string"]},
        )
    assert result == ("good",)
    assert [call.role for call in transport.calls] == ["reasoner", "reasoner.repair"]


def test_redaction_removes_keys_from_diagnostics() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    assert secret not in redact(f"Authorization: Bearer {secret}")
