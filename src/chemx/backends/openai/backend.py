"""OpenAI-compatible model backend."""

import os
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


@dataclass
class OpenAIBackend:
    """Remote backend for the OpenAI Chat Completions API."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    context_window_tokens: int = 1_047_576

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: JsonSchema | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "chemx_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        data = post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ModelError("OpenAI returned an unexpected response.") from error

    def complete_tool(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ToolCall:
        """Require one strict, non-parallel function call."""
        data = post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            payload={
                "model": self.model,
                "messages": [asdict(message) for message in messages],
                "tools": [tool.as_openai_tool(strict=True) for tool in tools],
                "tool_choice": "required",
                "parallel_tool_calls": False,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            calls = data["choices"][0]["message"]["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                raise ModelError("OpenAI must return exactly one tool call.")
            function = calls[0]["function"]
            return ToolCall(
                name=str(function["name"]),
                arguments=parse_tool_arguments(function["arguments"]),
            )
        except (KeyError, IndexError, TypeError) as error:
            raise ModelError("OpenAI returned an unexpected tool call.") from error


@register_backend("openai", default_model="gpt-4.1-mini")
def create_openai_backend(
    *,
    model: str,
    base_url: str | None,
    timeout: float,
) -> OpenAIBackend:
    """Build an OpenAI backend from common factory options."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENAI_API_KEY must contain an API key.")
    return OpenAIBackend(
        model=model,
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        timeout=timeout,
    )
