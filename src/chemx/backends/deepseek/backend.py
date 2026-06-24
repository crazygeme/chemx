"""DeepSeek OpenAI-compatible chat backend."""

import os
from dataclasses import asdict, dataclass
from typing import Sequence

from ..base import Message, ModelError
from ..http import post_json
from ..registry import register_backend


@dataclass
class DeepSeekBackend:
    """Remote backend for the DeepSeek Chat Completions API."""

    model: str
    api_key: str
    base_url: str = "https://api.deepseek.com"
    timeout: float = 60.0

    def complete(self, messages: Sequence[Message]) -> str:
        """Generate one non-streaming DeepSeek chat response."""
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
            "stream": False,
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
            raise ModelError("DeepSeek returned an unexpected response.") from error


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
