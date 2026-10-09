"""Strict function-call runtime shared by the two logical agents.

This module never chooses, ranks, or rewrites a tool call.  The decision
backend belongs to the calling Agent; the runtime only validates ownership,
executes a selected callable, and records an auditable trace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol


AgentRole = Literal["reasoner", "observer"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...] = ()
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("tool name and description must be nonempty")
        names = (*self.required_arguments, *self.optional_arguments)
        if any(not item.strip() for item in names) or len(set(names)) != len(names):
            raise ValueError(f"tool {self.name} has invalid argument names")

    def validate(self, arguments: Mapping[str, Any]) -> None:
        missing = [item for item in self.required_arguments if item not in arguments]
        if missing:
            raise ValueError(f"tool {self.name} is missing arguments: {missing}")
        allowed = set(self.required_arguments) | set(self.optional_arguments)
        unexpected = sorted(set(arguments) - allowed)
        if unexpected:
            raise ValueError(f"tool {self.name} received unexpected arguments: {unexpected}")

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "required_arguments": list(self.required_arguments),
            "optional_arguments": list(self.optional_arguments),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool call name must be nonempty")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool call arguments must be an object")


@dataclass(frozen=True)
class ToolTrace:
    step: int
    call: ToolCall
    result: Any


@dataclass(frozen=True)
class ToolLoopResult:
    role: AgentRole
    terminal_call: ToolCall
    trace: tuple[ToolTrace, ...]


class ToolChoiceBackend(Protocol):
    def choose(
        self,
        *,
        role: AgentRole,
        context: str,
        available_tools: Sequence[ToolSpec],
        trace: Sequence[ToolTrace],
    ) -> ToolCall:
        """Choose the next tool. The runtime must not alter this decision."""


ToolFunction = Callable[[Mapping[str, Any]], Any]


def execute_agent_tool_loop(
    *,
    role: AgentRole,
    context: str,
    available_tools: Sequence[ToolSpec],
    chooser: ToolChoiceBackend,
    implementations: Mapping[str, ToolFunction],
    maximum_steps: int,
) -> ToolLoopResult:
    if not context.strip():
        raise ValueError(f"{role} tool loop requires nonempty context")
    if maximum_steps <= 0:
        raise ValueError("maximum tool steps must be positive")
    specs = {tool.name: tool for tool in available_tools}
    if len(specs) != len(available_tools):
        raise ValueError(f"{role} tool specifications contain duplicate names")
    trace: list[ToolTrace] = []
    for step in range(1, maximum_steps + 1):
        call = chooser.choose(
            role=role,
            context=context,
            available_tools=available_tools,
            trace=tuple(trace),
        )
        spec = specs.get(call.name)
        if spec is None:
            raise ValueError(f"{role} selected an unavailable tool: {call.name}")
        spec.validate(call.arguments)
        if spec.terminal:
            return ToolLoopResult(
                role=role,
                terminal_call=call,
                trace=tuple(trace),
            )
        implementation = implementations.get(call.name)
        if implementation is None:
            raise KeyError(f"no implementation registered for {role} tool {call.name}")
        result = implementation(call.arguments)
        trace.append(ToolTrace(step=step, call=call, result=result))
    raise RuntimeError(f"{role} exhausted the {maximum_steps}-step tool budget")


class ScriptedToolChoice:
    """Deterministic CPU-only decision backend for integration tests."""

    def __init__(self, calls: Sequence[ToolCall]) -> None:
        if not calls:
            raise ValueError("scripted tool choice requires at least one call")
        self._calls = tuple(calls)
        self._position = 0

    def choose(
        self,
        *,
        role: AgentRole,
        context: str,
        available_tools: Sequence[ToolSpec],
        trace: Sequence[ToolTrace],
    ) -> ToolCall:
        del role, context, available_tools, trace
        if self._position >= len(self._calls):
            raise RuntimeError("scripted tool choice ran out of calls before termination")
        call = self._calls[self._position]
        self._position += 1
        return call
