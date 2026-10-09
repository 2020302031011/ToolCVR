"""One OpenAI-compatible transport for every ToolCVR model role.

Agent mechanisms own prompts and domain schemas.  This module owns only the
stable messaging contract: request construction, transient retries, response
extraction, JSON extraction, bounded structural repair, redaction, and usage
telemetry.  Keeping that boundary in one place prevents a new reasoning
mechanism from accidentally inventing a new, weaker API client.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Generic, Mapping, Sequence, TypeVar


T = TypeVar("T")
_SENSITIVE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\"'\s,}]+|"
    r"\b(?:sk|hf)[-_][A-Za-z0-9._-]{12,}\b"
)


class StructuredOutputError(RuntimeError):
    """A bounded JSON extraction/validation/repair sequence failed."""


@dataclass(frozen=True)
class APIRequestConfig:
    maximum_tokens: int
    timeout_seconds: float
    temperature: float = 0.0
    maximum_retries: int = 3
    repair_retries: int = 1
    enable_thinking: bool | None = None
    thinking_parameter: str = "enable_thinking"

    def __post_init__(self) -> None:
        if self.maximum_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("API token and timeout limits must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("API temperature must lie in [0, 2]")
        if self.maximum_retries < 0 or self.repair_retries < 0:
            raise ValueError("API retry budgets must be nonnegative")
        if self.thinking_parameter not in ("enable_thinking", "thinking"):
            raise ValueError("thinking_parameter must be enable_thinking or thinking")


@dataclass(frozen=True)
class APIUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    visual_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class APICallRecord:
    role: str
    model: str
    visual_input_count: int
    retry_count: int
    usage: APIUsage
    request_payload: Mapping[str, object]
    raw_response: str | None
    latency_seconds: float
    transient_errors: tuple[str, ...] = ()
    error: str | None = None


def redact(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix = next((group for group in match.groups() if group), "")
        return prefix + "[REDACTED]"

    return _SENSITIVE.sub(replace, value)


def extract_json_object(text: str) -> dict[str, object]:
    """Extract one JSON object from plain, fenced, or lightly wrapped output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise StructuredOutputError("model response contains no valid JSON object")
    if not isinstance(value, dict):
        raise StructuredOutputError("structured model response must be a JSON object")
    return value


class OpenAICompatibleTransport(Generic[T]):
    """Thread-safe, role-agnostic OpenAI-compatible messaging backend."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "LLM_API_KEY",
        record_raw_responses: bool = True,
        maximum_raw_chars: int = 65536,
        trace_sink: Callable[[APICallRecord], None] | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip() or not api_key_env.strip():
            raise ValueError("API transport requires base URL, model and key environment")
        if maximum_raw_chars <= 0:
            raise ValueError("maximum_raw_chars must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.record_raw_responses = record_raw_responses
        self.maximum_raw_chars = maximum_raw_chars
        self.trace_sink = trace_sink
        self.calls: list[APICallRecord] = []
        self._calls_lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def complete(
        self,
        *,
        role: str,
        messages: Sequence[dict[str, object]],
        config: APIRequestConfig,
        visual_input_count: int = 0,
        json_response: bool = True,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": config.temperature,
            "max_tokens": config.maximum_tokens,
            "messages": list(messages),
        }
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        if config.enable_thinking is not None:
            if config.thinking_parameter == "enable_thinking":
                payload["enable_thinking"] = config.enable_thinking
            else:
                payload["thinking"] = {
                    "type": "enabled" if config.enable_thinking else "disabled"
                }
        return self._post(
            role=role,
            payload=payload,
            config=config,
            visual_input_count=visual_input_count,
        )

    def complete_structured(
        self,
        *,
        role: str,
        messages: Sequence[dict[str, object]],
        config: APIRequestConfig,
        validator: Callable[[dict[str, object]], T],
        schema_hint: object,
        visual_input_count: int = 0,
    ) -> T:
        raw = self.complete(
            role=role,
            messages=messages,
            config=config,
            visual_input_count=visual_input_count,
        )
        last_error = ""
        for repair_index in range(config.repair_retries + 1):
            try:
                return validator(extract_json_object(raw))
            except (StructuredOutputError, KeyError, TypeError, ValueError) as exc:
                last_error = redact(str(exc))
                if repair_index >= config.repair_retries:
                    break
                repair_messages: list[dict[str, object]] = [
                    {
                        "role": "system",
                        "content": (
                            "Repair the candidate into valid JSON matching the supplied "
                            "contract. Return JSON only and do not add facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "contract": schema_hint,
                                "validation_error": last_error,
                                "candidate": redact(raw)[: self.maximum_raw_chars],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
                raw = self.complete(
                    role=f"{role}.repair",
                    messages=repair_messages,
                    config=config,
                    visual_input_count=0,
                )
        raise StructuredOutputError(
            f"{role} validation failed after {config.repair_retries} repair "
            f"attempt(s): {last_error}"
        )

    def _post(
        self,
        *,
        role: str,
        payload: dict[str, object],
        config: APIRequestConfig,
        visual_input_count: int,
    ) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"API key is unset: {self.api_key_env}")
        retry_count = 0
        raw = ""
        usage = APIUsage()
        transient_errors: list[str] = []
        started = time.monotonic()
        safe_value = _trace_safe_payload(payload)
        if not isinstance(safe_value, dict):
            raise AssertionError("API request trace must remain a mapping")
        safe_request = safe_value
        try:
            for attempt in range(config.maximum_retries + 1):
                request = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=config.timeout_seconds
                    ) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    raw, usage = _response_content(body)
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")[:2000]
                    retryable = exc.code in {408, 409, 429} or exc.code >= 500
                    if not retryable or attempt >= config.maximum_retries:
                        raise RuntimeError(
                            f"API returned HTTP {exc.code}: {redact(detail)}"
                        ) from exc
                    delay = _retry_delay(attempt, exc.headers.get("Retry-After"))
                    transient_errors.append(f"HTTP {exc.code}: {redact(detail)}")
                # ``socket.timeout`` is not a ``TimeoutError`` subclass on the
                # Python 3.9 runtime used by the evaluation environment.  Catch
                # it explicitly so read/connect timeouts follow the same
                # bounded retry contract as other transient transport errors.
                except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                    if attempt >= config.maximum_retries:
                        raise RuntimeError(
                            f"API transport failed after {attempt + 1} attempt(s): {exc}"
                        ) from exc
                    delay = _retry_delay(attempt, None)
                    transient_errors.append(redact(f"{type(exc).__name__}: {exc}"))
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(f"API response has invalid shape: {exc}") from exc
                retry_count += 1
                time.sleep(delay)
            else:
                raise RuntimeError("API retry loop exhausted")
        except Exception as exc:
            self._record(APICallRecord(
                role=role,
                model=self.model,
                visual_input_count=visual_input_count,
                retry_count=retry_count,
                usage=usage,
                request_payload=safe_request,
                raw_response=self._safe_raw(raw, key),
                latency_seconds=time.monotonic() - started,
                transient_errors=tuple(transient_errors),
                error=redact(str(exc)),
            ))
            raise
        self._record(APICallRecord(
            role=role,
            model=self.model,
            visual_input_count=visual_input_count,
            retry_count=retry_count,
            usage=usage,
            request_payload=safe_request,
            raw_response=self._safe_raw(raw, key),
            latency_seconds=time.monotonic() - started,
            transient_errors=tuple(transient_errors),
        ))
        return raw

    def _safe_raw(self, raw: str, key: str) -> str | None:
        if not self.record_raw_responses:
            return None
        return redact(raw.replace(key, "[REDACTED]"))[: self.maximum_raw_chars]

    def _record(self, value: APICallRecord) -> None:
        with self._calls_lock:
            self.calls.append(value)
            if self.trace_sink is not None:
                self.trace_sink(value)


def _trace_safe_payload(value: object) -> object:
    """Remove embedded image bytes while preserving an exact text request.

    Multimodal messages can contain megabytes of base64 data.  Training traces
    retain the surrounding frame labels plus a stable image digest and byte
    count; frames can then be reconstructed from the committed frame manifest
    without duplicating image payloads in every API record.
    """
    if isinstance(value, dict):
        return {str(key): _trace_safe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_trace_safe_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_trace_safe_payload(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        encoded = value.split(",", 1)[1] if "," in value else value
        return {
            "embedded_image": True,
            "encoded_bytes": len(encoded),
            "sha256": sha256(encoded.encode("ascii", errors="ignore")).hexdigest(),
        }
    return value


def _response_content(body: object) -> tuple[str, APIUsage]:
    if not isinstance(body, dict):
        raise TypeError("response root is not an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TypeError("response choices must be a nonempty list")
    first = choices[0]
    if not isinstance(first, dict):
        raise TypeError("response choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise TypeError("response message must be an object")
    raw = message.get("content")
    if isinstance(raw, list):
        raw = "".join(
            item.get("text", "")
            for item in raw
            if isinstance(item, dict) and isinstance(item.get("text", ""), str)
        )
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError("choices[0].message.content is not nonempty text")
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {}
    return raw, APIUsage(
        input_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
        output_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
        visual_tokens=_optional_nonnegative_int(prompt_details.get("image_tokens")),
        reasoning_tokens=_optional_nonnegative_int(
            completion_details.get("reasoning_tokens")
        ),
    )


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 30.0)
        except ValueError:
            pass
    return min(float(2**attempt), 8.0)
