"""Ollama model backend."""

from dataclasses import asdict, dataclass
from typing import Sequence

from ..base import (
    JsonSchema,
    Message,
    ModelError,
    ToolCall,
    ToolDefinition,
    parse_tool_arguments,
)
from ..http import post_json
from ..registry import register_backend
from .runtime import ensure_ollama_running


@dataclass
class OllamaBackend:
    """Local backend for the Ollama chat API."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0
    context_window_tokens: int = 131_072

    def prepare(self) -> None:
        """Start an installed local Ollama service when it is not running."""
        ensure_ollama_running(
            self.base_url,
            startup_timeout=min(self.timeout, 15.0),
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: JsonSchema | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
            "stream": False,
        }
        if response_schema is not None:
            payload["format"] = response_schema
        data = post_json(
            url=f"{self.base_url.rstrip('/')}/api/chat",
            payload=payload,
            headers={},
            timeout=self.timeout,
        )
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as error:
            raise ModelError("Ollama returned an unexpected response.") from error

    def complete_tool(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ToolCall:
        """Request one function call and enforce singularity locally."""
        data = post_json(
            url=f"{self.base_url.rstrip('/')}/api/chat",
            payload={
                "model": self.model,
                "messages": [asdict(message) for message in messages],
                "tools": [tool.as_openai_tool() for tool in tools],
                "stream": False,
            },
            headers={},
            timeout=self.timeout,
        )
        try:
            calls = data["message"]["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                raise ModelError("Ollama must return exactly one tool call.")
            function = calls[0]["function"]
            return ToolCall(
                name=str(function["name"]),
                arguments=parse_tool_arguments(function["arguments"]),
            )
        except (KeyError, TypeError) as error:
            raise ModelError("Ollama returned an unexpected tool call.") from error


@register_backend("ollama", default_model="llama3.2")
def create_ollama_backend(
    *,
    model: str,
    base_url: str | None,
    timeout: float,
) -> OllamaBackend:
    """Build an Ollama backend from common factory options."""
    return OllamaBackend(
        model=model,
        base_url=base_url or "http://localhost:11434",
        timeout=timeout,
    )
