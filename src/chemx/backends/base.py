"""Shared model backend types."""

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Message:
    """A model-independent chat message."""

    role: str
    content: str


class ModelBackend(Protocol):
    """Interface implemented by every model provider."""

    def complete(self, messages: Sequence[Message]) -> str:
        """Generate a response for a conversation."""
        return ""


class ModelError(RuntimeError):
    """Raised when a model provider cannot return a valid response."""
