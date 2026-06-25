"""Shared model backend types."""

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

JsonSchema = Mapping[str, object]


@dataclass(frozen=True)
class Message:
    """A model-independent chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class ToolDefinition:
    """One model-callable function and its JSON argument schema."""

    name: str
    description: str
    parameters: JsonSchema

    def as_openai_tool(self, *, strict: bool = False) -> dict[str, object]:
        function: dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if strict:
            function["strict"] = True
        return {"type": "function", "function": function}


@dataclass(frozen=True)
class ToolCall:
    """One validated function call selected by a model backend."""

    name: str
    arguments: dict[str, Any]


class ModelBackend(Protocol):
    """Interface implemented by every model provider."""

    context_window_tokens: int

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: JsonSchema | None = None,
    ) -> str:
        """Generate a response for a conversation."""
        return ""

    def complete_tool(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ToolCall:
        """Select exactly one tool call for a conversation."""
        raise NotImplementedError


class ModelError(RuntimeError):
    """Raised when a model provider cannot return a valid response."""


def parse_tool_arguments(value: object) -> dict[str, Any]:
    """Normalize provider tool arguments into one JSON object."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ModelError("Model returned invalid tool arguments.") from error
    if not isinstance(value, dict):
        raise ModelError("Model tool arguments must be a JSON object.")
    return value
