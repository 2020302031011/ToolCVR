"""Direct reasoning over validated warm evidence without an intermediate state.

This module is an ablation boundary, not a second reasoning pipeline for a
particular task.  Every sample uses the same compact evidence schema and the
same single Reasoner operation.  Ground truth is never part of the request.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from toolcvr.openai_compatible import APIRequestConfig, OpenAICompatibleTransport
from toolcvr.query import parse_json_object
from toolcvr.reasoning_cycle import conservative_token_count


@dataclass(frozen=True)
class DirectEvidenceAnswer:
    """Answer outcome and citation audit are deliberately independent.

    The direct-evidence ablation measures whether warm memory is sufficient for
    answering.  Citation compliance is useful diagnostic information, but it
    must not turn a valid answer into a failed sample.  Unknown citations are
    retained separately and are never promoted to grounded evidence.
    """

    status: Literal["ANSWER", "ABSTAIN"]
    answer: object | None
    evidence_refs: tuple[str, ...] = ()
    rejected_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status == "ANSWER":
            if not _has_answer_value(self.answer):
                raise ValueError("ANSWER outcome requires a nonempty JSON answer")
        elif self.status == "ABSTAIN":
            if self.answer is not None:
                raise ValueError("ABSTAIN outcome must not contain an answer")
        else:
            raise ValueError(f"unsupported direct evidence status: {self.status!r}")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("direct evidence citations must not repeat")
        if len(self.rejected_evidence_refs) != len(set(self.rejected_evidence_refs)):
            raise ValueError("rejected direct evidence citations must not repeat")
        overlap = set(self.evidence_refs).intersection(self.rejected_evidence_refs)
        if overlap:
            raise ValueError(f"accepted and rejected citations overlap: {sorted(overlap)}")

    @property
    def citation_status(self) -> Literal["grounded", "missing", "partial", "rejected"]:
        if self.evidence_refs and self.rejected_evidence_refs:
            return "partial"
        if self.evidence_refs:
            return "grounded"
        if self.rejected_evidence_refs:
            return "rejected"
        return "missing"

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "answer": self.answer,
            "evidence_refs": list(self.evidence_refs),
            "rejected_evidence_refs": list(self.rejected_evidence_refs),
            "citation_status": self.citation_status,
        }


class DirectEvidenceBackend(Protocol):
    def answer(self, context: str, *, known_refs: set[str], options: Sequence[str]) -> DirectEvidenceAnswer:
        """Return one answer grounded in the supplied current evidence."""


@dataclass(frozen=True)
class DeterministicMockDirectEvidenceBackend:
    """CPU-only backend for the direct-evidence integration path."""

    def answer(self, context: str, *, known_refs: set[str], options: Sequence[str]) -> DirectEvidenceAnswer:
        payload = json.loads(context)
        if not known_refs:
            raise ValueError("mock direct reasoner requires evidence")
        answer = _option_labels(options)[0] if options else "insufficient-evidence-mock-answer"
        return DirectEvidenceAnswer("ANSWER", answer, (sorted(known_refs)[0],))


@dataclass(frozen=True)
class OpenAICompatibleDirectEvidenceBackend:
    base_url: str
    model: str
    api_key_env: str = "LLM_API_KEY"
    maximum_tokens: int = 768
    timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("direct evidence backend requires URL and model")
        if self.maximum_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("direct evidence generation limits must be positive")

    def answer(self, context: str, *, known_refs: set[str], options: Sequence[str]) -> DirectEvidenceAnswer:
        system = (
            "You answer one multi-video question using only the supplied validated warm evidence. "
            "Return exactly one JSON object with keys answer and evidence_refs. evidence_refs is a "
            "list of at most 8 event IDs copied exactly from the input when supporting events are "
            "available. Citations are audit metadata: use an empty list rather than inventing an "
            "unseen event. If the supplied evidence is genuinely insufficient, set answer to null. "
            "Do not treat omitted video regions as negative evidence. Follow the "
            "output format requested by the question. When options are supplied, answer must be "
            "exactly one option label such as A, B, C, or D, not the option text."
        )
        content = OpenAICompatibleTransport(
            base_url=self.base_url,
            model=self.model,
            api_key_env=self.api_key_env,
        ).complete(
            role="direct_evidence",
            config=APIRequestConfig(
                maximum_tokens=self.maximum_tokens,
                timeout_seconds=self.timeout_seconds,
            ),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
        )
        return parse_direct_evidence_answer(
            content, known_refs=known_refs, options=options,
        )


def build_direct_evidence_context(
    report: Mapping[str, object], *, token_budget: int = 12_000,
) -> tuple[str, set[str], int]:
    """Render every validated warm event in video/time order.

    This ablation intentionally performs no LLM summarization, requirement
    formation, event ranking, or task-dependent routing between the report and
    the Reasoner.  If the complete evidence does not fit, the experiment fails
    loudly rather than silently changing what is being measured.
    """
    if token_budget <= 0:
        raise ValueError("direct evidence token budget must be positive")
    question = report.get("question")
    raw_videos = report.get("videos")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("warm report requires a question")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise ValueError("warm report requires videos")
    options = report.get("options", [])
    if not isinstance(options, list) or any(not isinstance(item, str) for item in options):
        raise ValueError("warm report options must be strings")

    known_refs: set[str] = set()
    videos: list[dict[str, object]] = []
    for raw_video in raw_videos:
        if not isinstance(raw_video, Mapping):
            raise ValueError("warm report video must be an object")
        video_id = raw_video.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("warm report video requires video_id")
        raw_events = raw_video.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("warm report events must be an array")
        events: list[dict[str, object]] = []
        for event in raw_events:
            if not isinstance(event, Mapping):
                raise ValueError("warm event must be an object")
            event_id = event.get("event_id")
            span = event.get("time_span")
            description = event.get("description")
            if (not isinstance(event_id, str) or not event_id
                    or not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(value, (int, float)) for value in span)
                    or float(span[0]) < 0 or float(span[1]) < float(span[0])
                    or not isinstance(description, str) or not description.strip()):
                raise ValueError(f"invalid warm event in {video_id}: {event!r}")
            if event_id in known_refs:
                raise ValueError(f"duplicate warm evidence ID: {event_id}")
            known_refs.add(event_id)
            events.append({
                "id": event_id,
                "time": [round(float(span[0]), 3), round(float(span[1]), 3)],
                "description": description,
                "inferred_interpretation": str(event.get("interpretation", "")),
                "uncertainty": str(event.get("uncertainty", "")),
            })
        events.sort(key=lambda item: (item["time"][0], item["time"][1], item["id"]))
        gaps = raw_video.get("observation_gaps", [])
        gap_count = len(gaps) if isinstance(gaps, list) else 0
        videos.append({
            "video_id": video_id,
            "duration_seconds": raw_video.get("duration_seconds"),
            "coverage": {
                "selected_frames": raw_video.get("selected_frames"),
                "candidate_frames": raw_video.get("clip_candidate_count"),
                "omitted_regions_are_negative_evidence": False,
                "observation_gap_count": gap_count,
            },
            "events": events,
        })
    if not known_refs:
        raise ValueError("direct evidence reasoning requires at least one warm event")
    payload = {
        "question": question,
        "options": options,
        "videos": videos,
        "instruction": (
            "Answer from direct event descriptions and cite supporting event IDs. "
            "inferred_interpretation is a hypothesis, not direct visual evidence."
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    tokens = conservative_token_count(text)
    if tokens > token_budget:
        raise ValueError(f"complete warm evidence exceeds direct context budget: {tokens}>{token_budget}")
    return text, known_refs, tokens


def parse_direct_evidence_answer(
    text: str, *, known_refs: set[str], options: Sequence[str],
) -> DirectEvidenceAnswer:
    value = parse_json_object(text)
    answer = value.get("answer")
    refs = value.get("evidence_refs", [])
    if isinstance(refs, str):
        refs = [refs]
    if refs is None:
        refs = []
    if not isinstance(refs, list):
        raise ValueError("direct evidence evidence_refs must be a string, array, or null")
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError("direct evidence citations must be nonempty strings")
    supplied = tuple(dict.fromkeys(ref.strip() for ref in refs))
    citations = tuple(ref for ref in supplied if ref in known_refs)
    rejected = tuple(ref for ref in supplied if ref not in known_refs)

    normalized = answer.strip() if isinstance(answer, str) else answer
    if (not _has_answer_value(normalized)
            or (isinstance(normalized, str)
                and normalized.upper() in {"ABSTAIN", "INSUFFICIENT_EVIDENCE"})):
        return DirectEvidenceAnswer("ABSTAIN", None, citations, rejected)

    labels = _option_labels(options)
    if labels:
        if not isinstance(normalized, str):
            raise ValueError(
                "multiple-choice answer must be a string option label, "
                f"got {type(normalized).__name__}",
            )
        matched = None
        for label, option in zip(labels, options):
            if (normalized.upper() in {label, f"{label}.", f"{label})", f"{label}:"}
                    or normalized.casefold() == option.strip().casefold()):
                matched = label
                break
        if matched is None:
            raise ValueError(f"multiple-choice answer must identify one of {labels}, got {normalized!r}")
        normalized = matched
    return DirectEvidenceAnswer("ANSWER", normalized, citations, rejected)


def _has_answer_value(value: object | None) -> bool:
    """Return whether a JSON answer carries content without assuming its shape."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return isinstance(value, (bool, int, float))


def _option_labels(options: Sequence[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for index, option in enumerate(options):
        stripped = option.strip()
        expected = chr(ord("A") + index)
        if len(stripped) >= 2 and stripped[0].isalpha() and stripped[1] in ".):":
            labels.append(stripped[0].upper())
        else:
            labels.append(expected)
    return tuple(labels)
