"""OpenAI-compatible model backend."""

import os
from dataclasses import asdict, dataclass
from typing import Sequence

from ..base import Message, ModelError
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

    def complete(self, messages: Sequence[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
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
