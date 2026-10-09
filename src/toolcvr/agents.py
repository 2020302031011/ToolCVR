from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from toolcvr.reporting import EventReporter
from toolcvr.schemas import (
    EventRecord,
    ObservationBatch,
    ObservationGap,
    QueryAssessment,
    ReportFrame,
    VideoSketch,
)
from toolcvr.tool_runtime import (
    ToolChoiceBackend,
    ToolFunction,
    ToolLoopResult,
    ToolSpec,
    execute_agent_tool_loop,
)


OBSERVER_TOOLS = (
    ToolSpec("semantic_seek", "Locate visually similar moments.", ("query", "video_ids")),
    ToolSpec("temporal_expand", "Expand context around a time anchor.", ("video_id", "anchor_seconds")),
    ToolSpec("dense_focus", "Inspect a short interval densely.", ("video_id", "time_span")),
    ToolSpec("boundary_refine", "Refine an event start and end.", ("video_id", "time_span")),
    ToolSpec("before_after", "Compare state before and after an event.", ("video_id", "time_span")),
    ToolSpec("entity_track", "Track one entity through a local interval.", ("video_id", "time_span", "entity")),
    ToolSpec("visual_count", "Count visible instances without double counting.", ("video_id", "time_span", "target")),
    ToolSpec("pair_verify", "Visually verify a small candidate pair.", ("candidates",)),
    ToolSpec("ocr_read", "Read visible text in a local interval.", ("video_id", "time_span")),
    ToolSpec("asr_search", "Search dialogue or speech evidence.", ("video_id", "time_span", "query")),
    ToolSpec(
        "return_evidence",
        "Return grounded evidence to ReasonerAgent.",
        ("evidence_event_ids", "unresolved_needs"),
        terminal=True,
    ),
)


REASONER_TOOLS = (
    ToolSpec("search_evidence", "Search existing report text and event metadata.", ("query",)),
    ToolSpec("compare_reports", "Compare cited evidence across selected videos.", ("video_ids", "question")),
    ToolSpec("temporal_order", "Infer an order from cited event states.", ("event_ids",)),
    ToolSpec("aggregate_count", "Aggregate source-preserving count evidence.", ("event_ids",)),
    ToolSpec("causal_trace", "Trace a causal narrative through cited evidence.", ("event_ids", "question")),
    ToolSpec("check_counterexample", "Search for evidence against a working claim.", ("claim",)),
    ToolSpec(
        "request_observation",
        "Ask ObserverAgent for missing grounded evidence.",
        ("question", "target_video_ids", "missing_evidence"),
        ("anchor_event_ids", "candidate_times"),
        terminal=True,
    ),
    ToolSpec(
        "answer",
        "Return an answer supported by cited evidence.",
        ("answer", "evidence_event_ids"),
        terminal=True,
    ),
)


class ObserverAgent:
    """Visual agent that converts tool-provided frames into validated evidence."""

    def __init__(self, reporter: EventReporter) -> None:
        self._reporter = reporter

    def run_tools(
        self,
        *,
        observation_request: str,
        chooser: ToolChoiceBackend,
        implementations: Mapping[str, ToolFunction],
        maximum_steps: int = 6,
    ) -> ToolLoopResult:
        return execute_agent_tool_loop(
            role="observer",
            context=observation_request,
            available_tools=OBSERVER_TOOLS,
            chooser=chooser,
            implementations=implementations,
            maximum_steps=maximum_steps,
        )

    def observe(
        self,
        *,
        video_id: str,
        frames: Sequence[ReportFrame],
        question: str,
        report_hints: Sequence[str],
    ) -> ObservationBatch:
        if not video_id.strip():
            raise ValueError("observer video_id must be nonempty")
        if not frames:
            raise ValueError("observer requires at least one visual frame")
        if any(frame.video_id != video_id for frame in frames):
            raise ValueError("observer batch may contain frames from only one video")
        if any(
            right.actual_timestamp_seconds <= left.actual_timestamp_seconds
            for left, right in zip(frames[:-1], frames[1:])
        ):
            raise ValueError("observer frames must be in strict temporal order")
        known_frames = {frame.frame_id: frame for frame in frames}
        if len(known_frames) != len(frames):
            raise ValueError("observer frame IDs must be unique within a batch")

        reported = self._reporter.report(
            video_id=video_id,
            frames=frames,
            question=question,
            report_hints=report_hints,
        )
        if not isinstance(reported, ObservationBatch):
            raise TypeError(
                "observer reporter must return ObservationBatch; event-only reporter "
                "outputs are not a valid observation contract"
            )
        result = reported
        events = result.events
        if not events and not result.observation_gaps:
            raise ValueError("observer must return events or an explicit observation gap")
        event_ids: set[str] = set()
        for event in events:
            if event.event_id in event_ids:
                raise ValueError(f"observer returned duplicate event ID: {event.event_id}")
            event_ids.add(event.event_id)
            unknown = [
                frame_id
                for frame_id in event.evidence_frame_ids
                if frame_id not in known_frames
            ]
            if unknown:
                raise ValueError(
                    f"observer event {event.event_id} cites unknown frames: {unknown}"
                )
            evidence_times = [
                known_frames[frame_id].actual_timestamp_seconds
                for frame_id in event.evidence_frame_ids
            ]
            tolerance = 0.25
            if event.start_seconds > min(evidence_times) + tolerance:
                raise ValueError(
                    f"observer event {event.event_id} starts after its evidence"
                )
            if event.end_seconds < max(evidence_times) - tolerance:
                raise ValueError(
                    f"observer event {event.event_id} ends before its evidence"
                )
        gap_ids: set[str] = set()
        for gap in result.observation_gaps:
            if gap.gap_id in gap_ids:
                raise ValueError(f"observer returned duplicate gap ID: {gap.gap_id}")
            gap_ids.add(gap.gap_id)
            unknown = [frame_id for frame_id in gap.frame_ids if frame_id not in known_frames]
            if unknown:
                raise ValueError(
                    f"observer gap {gap.gap_id} cites unknown frames: {unknown}"
                )
            evidence_times = [
                known_frames[frame_id].actual_timestamp_seconds for frame_id in gap.frame_ids
            ]
            if gap.start_seconds > min(evidence_times) + 0.25:
                raise ValueError(f"observer gap {gap.gap_id} starts after its frames")
            if gap.end_seconds < max(evidence_times) - 0.25:
                raise ValueError(f"observer gap {gap.gap_id} ends before its frames")
        return result

    @staticmethod
    def render_report_text(
        *,
        video_id: str,
        events: Sequence[EventRecord],
        selected_frames: int | None = None,
        candidate_frames: int | None = None,
        scan_stride_seconds: float | None = None,
        selection_mode: str | None = None,
        query_assessments: Sequence[QueryAssessment] = (),
        observation_gaps: Sequence[ObservationGap] = (),
    ) -> str:
        """Render the only report-text format accepted by the reasoner."""

        if not video_id.strip():
            raise ValueError("report video_id must be nonempty")
        if not events and not observation_gaps:
            raise ValueError("cannot render a report without events or explicit gaps")
        ordered = sorted(
            events,
            key=lambda event: (
                event.start_seconds,
                event.end_seconds,
                event.event_id,
            ),
        )
        coverage = (
            selected_frames,
            candidate_frames,
            scan_stride_seconds,
            selection_mode,
        )
        if any(value is not None for value in coverage) and not all(
            value is not None for value in coverage
        ):
            raise ValueError("report coverage metadata must be supplied as one complete set")
        lines = [f"VIDEO_REPORT {video_id}"]
        if all(value is not None for value in coverage):
            if (
                selected_frames is None
                or candidate_frames is None
                or scan_stride_seconds is None
                or selection_mode is None
            ):
                raise AssertionError("complete coverage tuple unexpectedly contains None")
            if not 0 < selected_frames <= candidate_frames or scan_stride_seconds <= 0:
                raise ValueError("report coverage counts/stride are invalid")
            lines.append(
                "COVERAGE "
                f"selected={selected_frames}/{candidate_frames} "
                f"sparse_stride={scan_stride_seconds:.3f}s mode={selection_mode}"
            )
            if selected_frames < candidate_frames:
                lines.append(
                    "COVERAGE_CAVEAT omitted sparse candidates are unobserved, "
                    "not negative evidence"
                )
            else:
                lines.append(
                    "COVERAGE_CAVEAT all sparse candidates retained; moments between "
                    "sampled frames remain unobserved"
                )
        lines.append("EVENT INDEX")
        if ordered:
            for event in ordered:
                evidence = ",".join(event.evidence_frame_ids)
                lines.append(
                    f"- {event.event_id} [{event.start_seconds:.3f}-{event.end_seconds:.3f}s] "
                    f"{event.description.strip()} "
                    f"| evidence={evidence} | confidence={event.confidence:.3f}"
                )
        else:
            lines.append("- none validated")
        lines.append("INFERRED INTERPRETATIONS (not direct visual evidence)")
        interpreted = [event for event in ordered if event.interpretation.strip()]
        if interpreted:
            lines.extend(
                f"- {event.event_id} {event.interpretation.strip()}"
                for event in interpreted
            )
        else:
            lines.append("- none")
        lines.append("OBSERVATION GAPS (not visual evidence)")
        if observation_gaps:
            for gap in observation_gaps:
                frames = ",".join(gap.frame_ids)
                lines.append(
                    f"- {gap.gap_id} [{gap.start_seconds:.3f}-{gap.end_seconds:.3f}s] "
                    f"{gap.reason.strip()} | frames={frames}"
                )
        else:
            lines.append("- none")
        unresolved = [event for event in ordered if event.uncertainty.strip()]
        lines.append("UNCERTAINTY / NEEDS FOCUS")
        if unresolved:
            lines.extend(
                f"- {event.event_id} [{event.start_seconds:.3f}-{event.end_seconds:.3f}s] "
                f"{event.uncertainty.strip()}"
                for event in unresolved
            )
        else:
            lines.append("- none")
        lines.append("QUERY ASSESSMENT")
        if query_assessments:
            for position, assessment in enumerate(query_assessments, start=1):
                evidence = ",".join(assessment.evidence_event_ids) or "none"
                needs = "; ".join(assessment.unresolved_needs) or "none"
                tools = ",".join(assessment.suggested_tools) or "none"
                lines.append(
                    f"- batch={position} relevance={assessment.relevance} "
                    f"evidence={evidence} unresolved={needs} suggested_tools={tools}"
                )
        else:
            lines.append("- unavailable (legacy observer output)")
        return "\n".join(lines)


class ReasonerAgent:
    """Text-only agent boundary for planning over Observer evidence reports."""

    def run_tools(
        self,
        *,
        reasoning_context: str,
        chooser: ToolChoiceBackend,
        implementations: Mapping[str, ToolFunction],
        maximum_steps: int = 6,
    ) -> ToolLoopResult:
        return execute_agent_tool_loop(
            role="reasoner",
            context=reasoning_context,
            available_tools=REASONER_TOOLS,
            chooser=chooser,
            implementations=implementations,
            maximum_steps=maximum_steps,
        )

    def build_evidence_context(self, reports: Sequence[VideoSketch]) -> str:
        if not reports:
            raise ValueError("reasoner requires at least one video report")
        video_ids = [report.video_id for report in reports]
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("reasoner video reports must have unique video IDs")
        missing = [report.video_id for report in reports if not report.report_text.strip()]
        if missing:
            raise ValueError(f"reasoner received empty report text for videos: {missing}")
        ordered = sorted(reports, key=lambda report: report.video_id)
        return "\n\n".join(report.report_text for report in ordered)
