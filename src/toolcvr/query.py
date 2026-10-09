from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from toolcvr.openai_compatible import APIRequestConfig, OpenAICompatibleTransport
from toolcvr.schemas import QueryPlan, RetrievalPhrase


class QueryRewriter(Protocol):
    def rewrite(self, question: str, options: Sequence[str]) -> QueryPlan:
        """Return positively observable phrases and non-filtering report hints."""


_OPTION_PREFIX = re.compile(r"^\s*[A-Z][.)]\s*", re.IGNORECASE)
_VIDEO_PREFIX = re.compile(r"^\s*Video\s*\d+\s*:\s*", re.IGNORECASE)
_ONLY_VIDEO_REFS = re.compile(
    r"^(?:\s*(?:Video|Videos)\s*(?:one|two|three|four|[1-9])(?:\s*(?:and|,|/)\s*)?)+$",
    re.IGNORECASE,
)
_ONLY_NUMBER = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
_PLACEHOLDER = re.compile(r"\{[AB]\d+\}", re.IGNORECASE)
_NONRETRIEVABLE_NEGATION = re.compile(
    r"\b(?:no|not|never|without|absent|absence|skip|skips|skipped|"
    r"does\s+not|did\s+not|doesn't|didn't|cannot|can't)\b",
    re.IGNORECASE,
)


def _clean_option(option: str) -> str:
    value = _OPTION_PREFIX.sub("", option).strip()
    value = _VIDEO_PREFIX.sub("", value).strip()
    return value


def _deduplicate(values: Sequence[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split()).strip(" .")
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if len(result) >= limit:
            break
    return result


class ConservativeQueryRewriter:
    """Deterministic fallback that never invents unseen nouns or actions.

    It deliberately produces broad phrases. A prompted text model is expected
    to perform better, but this fallback keeps the pipeline executable and
    makes failed/flat retrieval visible to the selector.
    """

    def __init__(self, *, maximum_phrases: int = 8) -> None:
        if maximum_phrases <= 0:
            raise ValueError("maximum_phrases must be positive")
        self.maximum_phrases = maximum_phrases

    def rewrite(self, question: str, options: Sequence[str]) -> QueryPlan:
        if not question.strip():
            raise ValueError("question must be nonempty")

        candidates: list[tuple[str, str]] = []
        question_phrase = self._question_phrase(question)
        if question_phrase:
            candidates.append((question_phrase, "question"))

        for index, option in enumerate(options):
            cleaned = _clean_option(option)
            if not cleaned or _ONLY_VIDEO_REFS.fullmatch(cleaned) or _ONLY_NUMBER.fullmatch(cleaned):
                continue
            # A negative or absence claim cannot be retrieved visually. Keep
            # it as a report hint, but do not turn it into a CLIP hard filter.
            lowered = cleaned.casefold()
            if any(token in lowered for token in ("not visible", "skips ", "does not ", "didn't ")):
                continue
            candidates.append((cleaned, f"option_{index + 1}"))

        phrases: list[RetrievalPhrase] = []
        seen: set[str] = set()
        for text, source in candidates:
            text = " ".join(_PLACEHOLDER.sub("", text).split()).strip(" ,.;:")
            key = text.casefold()
            if len(text.split()) < 2 or key in seen:
                continue
            seen.add(key)
            phrases.append(RetrievalPhrase(text=text, source=source))
            if len(phrases) >= self.maximum_phrases:
                break

        hints = _deduplicate([question, *[_clean_option(item) for item in options]], limit=10)
        return QueryPlan(phrases=tuple(phrases), report_hints=tuple(hints))

    @staticmethod
    def _question_phrase(question: str) -> str:
        value = " ".join(_PLACEHOLDER.sub("", question).split()).strip()
        # Remove answer-routing wrappers but retain the actual visible content.
        patterns = (
            r"^Which (?:one|two )?videos? ",
            r"^In which (?:film|video) ",
            r"^How do ",
            r"^What distinguishes ",
            r"^How many ",
            r"^When ",
        )
        for pattern in patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        return value.strip(" ?")


@dataclass(frozen=True)
class OpenAICompatibleQueryRewriter:
    base_url: str
    model: str
    api_key_env: str = "LLM_API_KEY"
    timeout_seconds: float = 120.0
    maximum_phrases: int = 8
    maximum_tokens: int = 1024
    enable_thinking: bool | None = None
    thinking_parameter: str | None = None

    def rewrite(self, question: str, options: Sequence[str]) -> QueryPlan:
        if not question.strip():
            raise ValueError("question must be nonempty")
        if self.maximum_tokens <= 0:
            raise ValueError("query rewrite maximum_tokens must be positive")
        thinking_parameter = self.thinking_parameter or "enable_thinking"
        transport = OpenAICompatibleTransport(
            base_url=self.base_url,
            model=self.model,
            api_key_env=self.api_key_env,
        )
        content = transport.complete(
            role="query_rewrite",
            config=APIRequestConfig(
                maximum_tokens=self.maximum_tokens,
                timeout_seconds=self.timeout_seconds,
                enable_thinking=self.enable_thinking,
                thinking_parameter=thinking_parameter,
            ),
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "options": list(options)},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        data = parse_json_object(str(content))
        raw_phrases = data.get("retrieval_phrases")
        raw_hints = data.get("report_hints")
        if not isinstance(raw_phrases, list) or not isinstance(raw_hints, list):
            raise ValueError("query rewrite JSON requires retrieval_phrases and report_hints lists")

        phrases: list[RetrievalPhrase] = []
        rejected_hints: list[str] = []
        for position, item in enumerate(raw_phrases[: self.maximum_phrases]):
            if not isinstance(item, dict):
                raise ValueError(f"retrieval phrase {position} must be an object")
            targets = item.get("target_video_ids", [])
            if not isinstance(targets, list) or not all(isinstance(value, str) for value in targets):
                raise ValueError("target_video_ids must be a string list")
            text = str(item.get("text", "")).strip()
            if _NONRETRIEVABLE_NEGATION.search(text):
                rejected_hints.append(f"Non-retrievable condition: {text}")
                continue
            phrases.append(
                RetrievalPhrase(
                    text=text,
                    source=str(item.get("source", "rewrite")).strip(),
                    weight=float(item.get("weight", 1.0)),
                    target_video_ids=tuple(targets),
                )
            )
        # Expanded captions improve recall for concrete realizations, but may
        # accidentally narrow an open concept (for example, "pickling" into
        # "placing vegetables in a glass jar"). Whenever the API found at
        # least one usable visual phrase, retain the original unexpanded
        # question and semantic options in the same CLIP text pool. An empty
        # rewrite remains empty so no-signal tasks still take the full-timeline
        # fallback.
        if phrases:
            phrases = _prepend_original_anchors(question, options, phrases)
        if not all(isinstance(item, str) and item.strip() for item in raw_hints):
            raise ValueError("report_hints must contain nonempty strings")
        # The API may summarize away a negative/count option because it is not
        # useful for CLIP retrieval. Preserve the original decision context for
        # the event reporter even when it is intentionally absent from phrases.
        hints = _deduplicate(
            [question, *[_clean_option(item) for item in options], *raw_hints, *rejected_hints],
            limit=24,
        )
        return QueryPlan(phrases=tuple(phrases), report_hints=tuple(hints))


def _prepend_original_anchors(
    question: str,
    options: Sequence[str],
    expansions: Sequence[RetrievalPhrase],
) -> list[RetrievalPhrase]:
    anchors = [RetrievalPhrase(text=question.strip(), source="original_question")]
    for index, option in enumerate(options):
        cleaned = _clean_option(option)
        if (
            not cleaned
            or _ONLY_VIDEO_REFS.fullmatch(cleaned)
            or _ONLY_NUMBER.fullmatch(cleaned)
            or _NONRETRIEVABLE_NEGATION.search(cleaned)
        ):
            continue
        anchors.append(
            RetrievalPhrase(
                text=cleaned,
                source=f"original_option_{chr(ord('A') + index)}",
            )
        )

    result: list[RetrievalPhrase] = []
    seen: set[str] = set()
    for phrase in [*anchors, *expansions]:
        key = " ".join(phrase.text.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(phrase)
    return result


def parse_json_object(text: str) -> dict[str, object]:
    """Decode one JSON object, tolerating a markdown fence around it."""
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {text[:500]}") from exc
    if not isinstance(result, dict):
        raise ValueError("model rewrite response must be a JSON object")
    return result


_REWRITE_SYSTEM_PROMPT = """You compile CLIP image-text retrieval queries for a long-video scanner.
Return one JSON object with exactly these keys:
{
  "retrieval_phrases": [
    {"text": "caption-like description of one visible frame", "source": "question|option_A|...", "weight": 1.0, "target_video_ids": []}
  ],
  "report_hints": ["reasoning requirement that must not be used as a hard image filter"]
}

Rules:
- Produce at most 8 natural-image captions grounded only in the question/options.
- A retrieval phrase is a CLIP caption, not a keyword, category label, answer label,
  or abstract concept. It should normally contain 5-15 words.
- Each phrase should combine the visible subject/material with an action, tool,
  interaction, or observable state. A tool name alone is not sufficient.
- For each positively observable answer candidate, produce 1-2 complementary
  captions: normally one interaction/action view and one visible result/state view.
- Every option-derived caption must express that option as a candidate answer
  to the specific visual relation asked by the question. Do not describe a
  tool's generic use when that use does not realize the questioned relation.
- You may infer a visible interaction directly entailed by combining the
  question and an option, but do not add decorative attributes or unrelated
  functions. For example, if the question asks about removing liquid and an
  option names a sieve, describe liquid draining through the sieve rather than
  the sieve separating generic fine particles.
- Add at most 2 option-neutral question anchors when the question supplies a
  concrete visible action or state. Do not emit an abstract relation such as
  "difference", "correct method", or "moisture removal" by itself.
- Split temporal concepts into separate observable before-state and after-state
  captions; never claim an order inside one static-image phrase.
- Use all answer options symmetrically. Never guess which option is correct.
- Set target_video_ids to [] for every warm-start phrase. Candidate answer content
  must be searched symmetrically in every input video.
- Do not invent an object, tool, action, place, or outcome absent from the input.
- If the affected object/material is unnamed, use a neutral noun such as
  "material" or "object" instead of inventing a specific food or object class.
- Counts, negation, absence, chronology, correctness, causality, and labels such as {A1}/{B5}
  are not directly retrievable by CLIP; put those requirements in report_hints.
- Preserve every answer candidate in report_hints, including candidates that are
  unsuitable for CLIP retrieval.
- A phrase must describe something that could positively appear in one frame.
- Bad: "hammer". Good: "a person striking a metal nail with a hammer".
- Bad: "object removal". Good: "a hand lifting an object out of a container".
- If no phrase is visually grounded, return an empty retrieval_phrases list.
- Output compact JSON only, without markdown or explanatory prose."""
