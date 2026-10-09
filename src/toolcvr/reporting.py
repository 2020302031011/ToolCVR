from __future__ import annotations

import base64
import io
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from toolcvr.openai_compatible import (
    APICallRecord,
    APIRequestConfig,
    OpenAICompatibleTransport,
)
from toolcvr.schemas import (
    EventRecord,
    ObservationBatch,
    ObservationGap,
    QueryAssessment,
    ReportFrame,
)


class EventReporter(Protocol):
    def report(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        """Generate grounded events plus a separate question assessment."""


class DeterministicMockEventReporter:
    def report(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        del question, report_hints
        if not frames:
            raise ValueError("event reporter requires at least one frame")
        grouped: dict[str, list[ReportFrame]] = defaultdict(list)
        for frame in frames:
            grouped[frame.segment_id].append(frame)
        events: list[EventRecord] = []
        for position, (segment_id, values) in enumerate(grouped.items(), start=1):
            values.sort(key=lambda item: item.actual_timestamp_seconds)
            events.append(
                EventRecord(
                    event_id=f"{video_id}-mock-{position}",
                    start_seconds=values[0].actual_timestamp_seconds,
                    end_seconds=values[-1].actual_timestamp_seconds,
                    description=f"Mock event observation for candidate segment {segment_id}.",
                    evidence_frame_ids=tuple(item.frame_id for item in values),
                    confidence=0.5,
                    uncertainty="Mock reporter does not infer visual semantics.",
                )
            )
        return ObservationBatch(
            events=tuple(events),
            query_assessment=QueryAssessment(
                relevance="insufficient",
                evidence_event_ids=tuple(event.event_id for event in events),
                unresolved_needs=("Mock reporter does not infer question relevance.",),
                suggested_tools=("dense_focus",),
            ),
        )


class SelectionAuditReporter:
    """Export CLIP-selected frames without performing semantic VLM inference."""

    def __init__(
        self,
        output_directory: Path,
        *,
        columns: int = 4,
        cell_width: int = 320,
        cell_height: int = 240,
    ) -> None:
        if columns <= 0 or cell_width <= 0 or cell_height <= 0:
            raise ValueError("selection audit grid dimensions must be positive")
        self._output_directory = output_directory.expanduser().resolve()
        self._columns = columns
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._calls: dict[str, int] = defaultdict(int)

    def report(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        del question, report_hints
        if not frames:
            raise ValueError("selection audit requires at least one frame")
        self._calls[video_id] += 1
        batch_index = self._calls[video_id]
        video_directory = self._output_directory / video_id
        video_directory.mkdir(parents=True, exist_ok=True)
        contact_sheet = self._contact_sheet(frames)
        contact_sheet.save(video_directory / f"batch-{batch_index:03d}.jpg", quality=92)

        grouped: dict[str, list[ReportFrame]] = defaultdict(list)
        for frame in frames:
            grouped[frame.segment_id].append(frame)
        events: list[EventRecord] = []
        for position, (segment_id, values) in enumerate(grouped.items(), start=1):
            values.sort(key=lambda item: item.actual_timestamp_seconds)
            events.append(
                EventRecord(
                    event_id=f"audit-b{batch_index}-{position}",
                    start_seconds=values[0].actual_timestamp_seconds,
                    end_seconds=values[-1].actual_timestamp_seconds,
                    description=(
                        "CLIP-selected candidate frames exported for visual audit; "
                        "no VLM interpretation was performed."
                    ),
                    evidence_frame_ids=tuple(item.frame_id for item in values),
                    confidence=0.0,
                    uncertainty=f"Selection proposal only for segment {segment_id}.",
                )
            )
        return ObservationBatch(
            events=tuple(events),
            query_assessment=QueryAssessment(
                relevance="insufficient",
                evidence_event_ids=tuple(event.event_id for event in events),
                unresolved_needs=("Selection audit performs no visual interpretation.",),
                suggested_tools=("dense_focus",),
            ),
        )

    def _contact_sheet(self, frames: Sequence[ReportFrame]) -> Image.Image:
        label_height = 44
        rows = math.ceil(len(frames) / self._columns)
        sheet = Image.new(
            "RGB",
            (self._columns * self._cell_width, rows * (self._cell_height + label_height)),
            color=(20, 20, 20),
        )
        draw = ImageDraw.Draw(sheet)
        for position, frame in enumerate(frames):
            row, column = divmod(position, self._columns)
            x = column * self._cell_width
            y = row * (self._cell_height + label_height)
            image = Image.fromarray(frame.rgb, mode="RGB")
            image.thumbnail((self._cell_width, self._cell_height))
            paste_x = x + (self._cell_width - image.width) // 2
            paste_y = y + (self._cell_height - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
            label = (
                f"{frame.frame_id}  {frame.actual_timestamp_seconds:.3f}s\n"
                f"source_frame={frame.source_frame_index}  segment={frame.segment_id}"
            )
            draw.text((x + 4, y + self._cell_height + 4), label, fill=(255, 255, 0))
        return sheet


def _validated_event_report(
    *,
    video_id: str,
    frames: Sequence[ReportFrame],
    maximum_repairs: int,
    complete: Callable[[str | None], str],
) -> ObservationBatch:
    """Apply one validation/repair contract to every real visual backend."""
    complaint: str | None = None
    repairs = 0
    while True:
        response = complete(complaint)
        try:
            return parse_event_response(response, frames)
        except ValueError as exc:
            if repairs >= maximum_repairs:
                complaint = str(exc)
                print(
                    f"[vlm] {video_id} recorded explicit observation gap after "
                    f"{repairs + 1} rejected output(s): {complaint}",
                    flush=True,
                )
                return ObservationBatch(
                    events=(),
                    observation_gaps=(
                        ObservationGap(
                            gap_id="gap1",
                            start_seconds=frames[0].actual_timestamp_seconds,
                            end_seconds=frames[-1].actual_timestamp_seconds,
                            frame_ids=tuple(frame.frame_id for frame in frames),
                            reason=f"VLM output remained invalid: {complaint}",
                        ),
                    ),
                    query_assessment=QueryAssessment(
                        relevance="insufficient",
                        evidence_event_ids=(),
                        unresolved_needs=(
                            "This local frame batch has no validated visual description.",
                        ),
                        suggested_tools=("dense_focus",),
                    ),
                )
            repairs += 1
            complaint = str(exc)
            print(
                f"[vlm] {video_id} repair {repairs}/{maximum_repairs}: {complaint}",
                flush=True,
            )


class LocalQwen3VLReporter:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        adapter_path: str | None = None,
        maximum_new_tokens: int = 512,
        maximum_events: int = 8,
        overlay_timestamps: bool = True,
        maximum_repairs: int = 2,
    ) -> None:
        if maximum_new_tokens <= 0 or maximum_events <= 0:
            raise ValueError("report generation limits must be positive")
        if maximum_repairs < 0:
            raise ValueError("VLM repair budget must be nonnegative")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("local Qwen3-VL reporting requires torch and transformers") from exc

        self._torch = torch
        self._device = torch.device(device)
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("--vlm-adapter requires peft") from exc
            self._model = PeftModel.from_pretrained(self._model, adapter_path)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._maximum_new_tokens = maximum_new_tokens
        self._maximum_events = maximum_events
        self._overlay_timestamps = overlay_timestamps
        self._maximum_repairs = maximum_repairs

    def report(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        if not frames:
            raise ValueError("event reporter requires at least one frame")
        content: list[dict[str, object]] = []
        images: list[Image.Image] = []
        for frame in frames:
            label = (
                f"FRAME {frame.frame_id} @ {frame.actual_timestamp_seconds:.3f}s "
                f"segment={frame.segment_id}"
            )
            content.append({"type": "text", "text": label})
            image = Image.fromarray(frame.rgb, mode="RGB")
            if self._overlay_timestamps:
                image = _overlay(image, label)
            images.append(image)
            content.append({"type": "image", "image": image})
        content.append(
            {
                "type": "text",
                "text": _report_prompt(
                    video_id=video_id,
                    frames=frames,
                    question=question,
                    report_hints=report_hints,
                    maximum_events=self._maximum_events,
                ),
            }
        )
        def complete(complaint: str | None) -> str:
            blocks = list(content)
            if complaint is not None:
                blocks.append({"type": "text", "text": _repair_prompt(complaint, frames)})
            messages = [{"role": "user", "content": blocks}]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._processor(text=text, images=images, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self._maximum_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            input_length = inputs["input_ids"].shape[1]
            return self._processor.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return _validated_event_report(
            video_id=video_id,
            frames=frames,
            maximum_repairs=self._maximum_repairs,
            complete=complete,
        )


class OpenAICompatibleMultimodalReporter:
    """Remote multimodal Observer with the same validated evidence contract."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "LLM_API_KEY",
        maximum_new_tokens: int = 2048,
        maximum_events: int = 8,
        overlay_timestamps: bool = True,
        maximum_repairs: int = 2,
        timeout_seconds: float = 600.0,
        enable_thinking: bool | None = True,
        jpeg_quality: int = 88,
        maximum_request_retries: int = 3,
        trace_sink: Callable[[APICallRecord], None] | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("API visual reporter requires base_url and model")
        if maximum_new_tokens <= 0 or maximum_events <= 0 or timeout_seconds <= 0:
            raise ValueError("API visual reporter limits must be positive")
        if maximum_repairs < 0 or maximum_request_retries < 0:
            raise ValueError("API visual reporter retry budgets must be nonnegative")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("JPEG quality must lie in [1, 95]")
        self._base_url = base_url
        self._model = model
        self._api_key_env = api_key_env
        self._maximum_new_tokens = maximum_new_tokens
        self._maximum_events = maximum_events
        self._overlay_timestamps = overlay_timestamps
        self._maximum_repairs = maximum_repairs
        self._timeout_seconds = timeout_seconds
        self._enable_thinking = enable_thinking
        self._jpeg_quality = jpeg_quality
        self._maximum_request_retries = maximum_request_retries
        self._transport = OpenAICompatibleTransport(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            trace_sink=trace_sink,
        )
        self._request_config = APIRequestConfig(
            maximum_tokens=maximum_new_tokens,
            timeout_seconds=timeout_seconds,
            maximum_retries=maximum_request_retries,
            enable_thinking=enable_thinking,
        )

    def report(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        if not frames:
            raise ValueError("event reporter requires at least one frame")
        content: list[dict[str, object]] = []
        for frame in frames:
            label = (
                f"FRAME {frame.frame_id} @ {frame.actual_timestamp_seconds:.3f}s "
                f"segment={frame.segment_id}"
            )
            image = Image.fromarray(frame.rgb, mode="RGB")
            if self._overlay_timestamps:
                image = _overlay(image, label)
            content.extend((
                {"type": "text", "text": label},
                {
                    "type": "image_url",
                    "image_url": {"url": _jpeg_data_url(image, self._jpeg_quality)},
                },
            ))
        content.append({
            "type": "text",
            "text": _report_prompt(
                video_id=video_id,
                frames=frames,
                question=question,
                report_hints=report_hints,
                maximum_events=self._maximum_events,
            ),
        })

        def complete(complaint: str | None) -> str:
            blocks = list(content)
            if complaint is not None:
                blocks.append({"type": "text", "text": _repair_prompt(complaint, frames)})
            return self._transport.complete(
                role="observer_report",
                messages=[{"role": "user", "content": blocks}],
                config=self._request_config,
                visual_input_count=len(frames),
            )

        return _validated_event_report(
            video_id=video_id,
            frames=frames,
            maximum_repairs=self._maximum_repairs,
            complete=complete,
        )

def _jpeg_data_url(image: Image.Image, quality: int) -> str:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _overlay(image: Image.Image, label: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    box_height = 22
    draw.rectangle((0, 0, min(result.width, 720), box_height), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 0))
    return result


def _report_prompt(
    *,
    video_id: str,
    frames: Sequence[ReportFrame],
    question: str,
    report_hints: Sequence[str],
    maximum_events: int,
) -> str:
    if not frames:
        raise ValueError("report prompt requires a nonempty frame manifest")
    hints = "\n".join(f"- {item}" for item in report_hints) or "- none"
    manifest = "\n".join(
        f"- {frame.frame_id} @ {frame.actual_timestamp_seconds:.3f}s "
        f"segment={frame.segment_id}"
        for frame in frames
    )
    example_frame_id = frames[0].frame_id
    return f"""You are ObserverAgent performing a query-conditioned sparse scan of video {video_id}.

The frames above are sparse observations arranged in strict logical-video order. Every frame has an
explicit model-visible timestamp. Time gaps may be uneven and the supplied frames are not necessarily continuous.
Describe only evidence supported by the visible frames and their timestamps. Do not invent an event
inside an unobserved gap.

Question context:
{question}

Report hints (these guide observation but are not assumed true):
{hints}

Authoritative frame manifest (the only legal values for evidence_frame_ids):
{manifest}

Return JSON only:
{{
  "events": [
    {{
      "event_id": "e1",
      "time_span": [actual_start_seconds, actual_end_seconds],
      "direct_observation": "concise directly visible attributes, action, or state change",
      "interpretation": "optional inferred identity, role, intent, or causal hypothesis; empty when none",
      "evidence_frame_ids": ["{example_frame_id}"],
      "confidence": 0.0,
      "uncertainty": "what remains unresolved"
    }}
  ],
  "query_assessment": {{
    "relevance": "direct | indirect | insufficient",
    "evidence_event_ids": ["e1"],
    "unresolved_needs": ["question-relevant fact not established by these frames"],
    "suggested_tools": ["temporal_expand"]
  }}
}}

Rules:
- Emit at most {maximum_events} events.
- Keep the JSON compact. Merge adjacent frames that show the same event.
- Keep each direct_observation under 35 words and each interpretation and uncertainty under 20 words.
- Describe events and state changes, not independent frame captions.
- direct_observation is the citable evidence field. Use neutral visible attributes
  instead of guessing a person's role, an object's exact identity, intent, ownership,
  recipe ingredient, or causal purpose. A visible readable label may be transcribed.
- Put every query-conditioned identity, role, intent, ownership, ingredient, or causal
  hypothesis in interpretation, never in direct_observation. When interpretation is
  nonempty, uncertainty must state why that interpretation is not directly established.
- Time spans and evidence IDs must be supported by the supplied frame manifest.
- evidence_frame_ids must contain only exact IDs from the authoritative frame manifest;
  never copy an ID from another request or invent a placeholder.
- Frames with different segment labels are separated by an unobserved source
  discontinuity. Never cite frames from different segments in one event or
  infer motion/state transition across that boundary.
- The events list describes what is visibly present even when it does not answer the question.
- Prefer visible action, object state, tool use, entry/exit, count cue, and ordering cue.
- Do not choose an answer and do not compare against another video.
- Do not claim that an event is absent from the full video: most timeline regions were CLIP-filtered.
- Never write a negative event such as "no target action occurs". Describe only the visible state;
  put "target action not visible in supplied frames" in uncertainty when needed.
- query_assessment describes only this frame batch's local contribution. It must never judge
  whether another video, another batch, or the whole sample has enough evidence.
- If this batch has no direct match, still report its visible state, set relevance to
  "insufficient", and name only a visual detail that could be sought near this video/time.
- Never write "the other video is not provided", "cannot compare videos", or any equivalent
  global conclusion. Cross-video sufficiency is decided later by ReasonerAgent.
- relevance="direct" means this batch directly establishes a requested local fact;
  "indirect" means it provides useful local context; "insufficient" means this batch has no
  validated local match. It does not mean the whole sample is insufficient.
- suggested_tools may contain only: semantic_seek, temporal_expand, dense_focus, boundary_refine,
  before_after, entity_track, visual_count, pair_verify, ocr_read, asr_search.
- Use asr_search when dialogue, identity, authority, intent, or causal narrative cannot be recovered
  from silent frames. Use an empty suggested_tools list when no additional observation is indicated.
"""


def _repair_prompt(complaint: str, frames: Sequence[ReportFrame]) -> str:
    """The message sent back when a reply failed its mechanical checks.

    It repeats the frame timestamps deliberately.  The originals are in the
    conversation already, but as labels drawn over images; a model that just
    wrote an event span outside its own evidence is more likely to get it right
    from a plain list it can read than from the same labels a second time.
    """
    manifest = "\n".join(
        f"{frame.frame_id} @ {frame.actual_timestamp_seconds:.3f}s "
        f"[segment={frame.segment_id}]"
        for frame in frames
    )
    normalized_complaint = complaint.lower()
    if "direct_observation" in normalized_complaint or "interpretation" in normalized_complaint:
        correction = (
            "Every event must separate direct_observation from interpretation. "
            "direct_observation contains only visibly supported attributes, actions, and states; "
            "interpretation contains any inferred identity, role, intent, ingredient, or cause, "
            "or an empty string. A nonempty interpretation requires explicit uncertainty."
        )
    elif "empty evidence" in normalized_complaint:
        correction = (
            "The events list must contain at least one visible observation. Even when "
            "none of the supplied frames resolves the question, describe the visible "
            "object, scene, action, or state in at least one cited frame and record the "
            "remaining limitation in uncertainty. Do not invent a question-relevant event."
        )
    else:
        correction = (
            "An event's time_span must contain every frame the event cites, not just "
            "some of them."
        )
    return (
        f"The checker rejected your JSON: {complaint}\n"
        f"Correction required: {correction}\n"
        "Reply with the corrected JSON object only, same schema.\n"
        "Authoritative frame manifest:\n"
        f"{manifest}"
    )


def parse_event_response(text: str, frames: Sequence[ReportFrame]) -> ObservationBatch:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM event report is not valid JSON: {text[:800]}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise ValueError("VLM event report must be an object containing an events list")
    known = {item.frame_id: item for item in frames}
    events: list[EventRecord] = []
    for position, raw in enumerate(data["events"]):
        if not isinstance(raw, dict):
            raise ValueError(f"event {position} must be an object")
        if "direct_observation" not in raw or "interpretation" not in raw:
            raise ValueError(
                f"event {position} requires direct_observation and interpretation fields"
            )
        if not isinstance(raw["direct_observation"], str) or not isinstance(raw["interpretation"], str):
            raise ValueError(
                f"event {position} direct_observation and interpretation must be strings"
            )
        if "description" in raw:
            raise ValueError(
                f"event {position} must use direct_observation, not the legacy description field"
            )
        normalized = dict(raw)
        normalized["description"] = normalized.pop("direct_observation")
        event = EventRecord.from_mapping(normalized)
        unknown = [item for item in event.evidence_frame_ids if item not in known]
        if unknown:
            raise ValueError(f"event {event.event_id} cites unknown frames: {unknown}")
        evidence_times = [known[item].actual_timestamp_seconds for item in event.evidence_frame_ids]
        evidence_segments = {known[item].segment_id for item in event.evidence_frame_ids}
        if len(evidence_segments) != 1:
            raise ValueError(
                f"event {event.event_id} crosses discontinuous frame segments: "
                f"{sorted(evidence_segments)}"
            )
        tolerance = 0.25
        earliest = min(evidence_times)
        latest = max(evidence_times)
        # The complaint names the numbers, because it is also what gets sent
        # back to the model when a reply is repaired.  "starts after all cited
        # evidence" is enough for a log and useless as an instruction.
        if event.start_seconds > earliest + tolerance:
            raise ValueError(
                f"event {event.event_id} spans [{event.start_seconds:.3f}, "
                f"{event.end_seconds:.3f}]s but its earliest cited frame "
                f"{event.evidence_frame_ids[evidence_times.index(earliest)]} is at "
                f"{earliest:.3f}s; an event span must contain every frame it cites"
            )
        if event.end_seconds < latest - tolerance:
            raise ValueError(
                f"event {event.event_id} spans [{event.start_seconds:.3f}, "
                f"{event.end_seconds:.3f}]s but its latest cited frame "
                f"{event.evidence_frame_ids[evidence_times.index(latest)]} is at "
                f"{latest:.3f}s; an event span must contain every frame it cites"
            )
        events.append(event)
    if not events:
        raise ValueError("VLM event report must not silently return empty evidence")
    raw_assessment = data.get("query_assessment")
    if not isinstance(raw_assessment, dict):
        raise ValueError("VLM event report requires a query_assessment object")
    assessment = QueryAssessment.from_mapping(raw_assessment)
    return ObservationBatch(events=tuple(events), query_assessment=assessment)
