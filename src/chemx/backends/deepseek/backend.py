"""DeepSeek OpenAI-compatible chat backend."""

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
class DeepSeekBackend:
    """Remote backend for the DeepSeek Chat Completions API."""

    model: str
    api_key: str
    base_url: str = "https://api.deepseek.com"
    timeout: float = 60.0
    context_window_tokens: int = 128_000

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: JsonSchema | None = None,
    ) -> str:
        """Generate one non-streaming DeepSeek chat response."""
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
            "stream": False,
        }
        if response_schema is not None:
            # DeepSeek documents JSON-object mode but not JSON Schema mode.
            payload["response_format"] = {"type": "json_object"}
        data = post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ModelError("DeepSeek returned an unexpected response.") from error

    def complete_tool(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ToolCall:
        """Request one function call and enforce singularity locally."""
        data = post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            payload={
                "model": self.model,
                "messages": [asdict(message) for message in messages],
                "tools": [tool.as_openai_tool() for tool in tools],
                "tool_choice": "required",
                "thinking": {"type": "disabled"},
                "stream": False,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            calls = data["choices"][0]["message"]["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                raise ModelError("DeepSeek must return exactly one tool call.")
            function = calls[0]["function"]
            return ToolCall(
                name=str(function["name"]),
                arguments=parse_tool_arguments(function["arguments"]),
            )
        except (KeyError, IndexError, TypeError) as error:
            raise ModelError("DeepSeek returned an unexpected tool call.") from error


@register_backend("deepseek", default_model="deepseek-v4-flash")
def create_deepseek_backend(
    *,
    model: str,
    base_url: str | None,
    timeout: float,
) -> DeepSeekBackend:
    """Build a DeepSeek backend from common factory options."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Environment variable DEEPSEEK_API_KEY must contain an API key.")
    return DeepSeekBackend(
        model=model,
        api_key=api_key,
        base_url=base_url or "https://api.deepseek.com",
        timeout=timeout,
    )
