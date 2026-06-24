"""Ollama model backend."""

from dataclasses import asdict, dataclass
from typing import Sequence

from ..base import Message, ModelError
from ..http import post_json
from ..registry import register_backend
from .runtime import ensure_ollama_running


@dataclass
class OllamaBackend:
    """Local backend for the Ollama chat API."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0

    def prepare(self) -> None:
        """Start an installed local Ollama service when it is not running."""
        ensure_ollama_running(
            self.base_url,
            startup_timeout=min(self.timeout, 15.0),
        )

    def complete(self, messages: Sequence[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
            "stream": False,
        }
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
