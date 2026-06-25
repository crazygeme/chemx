"""Bound model inputs while preserving the most relevant conversation state."""

from dataclasses import dataclass
from math import ceil
from typing import Sequence

from ..backends import Message


_TRUNCATION_MARKER = "\n\n...[content truncated]...\n\n"


@dataclass(frozen=True)
class ContextPolicy:
    """Approximate token budget and retention rules for model requests."""

    context_window_tokens: int = 32_768
    reserved_output_tokens: int | None = None
    recent_turns: int = 8
    characters_per_token: float = 4.0
    max_observation_chars: int = 12_000
    max_observations: int = 8
    max_observation_chars_total: int = 32_000
    max_diff_chars: int = 24_000

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be at least 1.")
        if self.reserved_output_tokens is None:
            object.__setattr__(
                self,
                "reserved_output_tokens",
                calculate_reserved_output_tokens(self.context_window_tokens),
            )
        if not 0 <= self.reserved_output_tokens < self.context_window_tokens:
            raise ValueError(
                "reserved_output_tokens must be non-negative and smaller than "
                "context_window_tokens."
            )
        if self.input_token_budget <= 8:
            raise ValueError(
                "The input token budget must exceed chat message overhead."
            )
        if self.recent_turns < 0:
            raise ValueError("recent_turns cannot be negative.")
        if self.characters_per_token <= 0:
            raise ValueError("characters_per_token must be positive.")
        for name in (
            "max_observation_chars",
            "max_observations",
            "max_observation_chars_total",
            "max_diff_chars",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")

    @property
    def input_token_budget(self) -> int:
        """Return the context space available after reserving model output."""
        assert self.reserved_output_tokens is not None
        return self.context_window_tokens - self.reserved_output_tokens

    @property
    def input_character_budget(self) -> int:
        """Return a conservative character approximation of the input budget."""
        return max(1, int(self.input_token_budget * self.characters_per_token))

    @classmethod
    def from_backend(cls, backend: object) -> "ContextPolicy":
        """Build a policy from context metadata owned by a model backend."""
        try:
            context_window_tokens = int(
                getattr(backend, "context_window_tokens")
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "Model backend must provide a valid context_window_tokens value."
            ) from error
        return cls(context_window_tokens=context_window_tokens)


def calculate_reserved_output_tokens(context_window_tokens: int) -> int:
    """Reserve one eighth of the window, bounded for small and large models."""
    if context_window_tokens < 18:
        raise ValueError("context_window_tokens must be at least 18.")
    calculated = context_window_tokens // 8
    return min(max(calculated, 512), 16_384, context_window_tokens // 2)


def estimate_message_tokens(
    messages: Sequence[Message],
    *,
    characters_per_token: float = 4.0,
) -> int:
    """Estimate chat tokens without requiring a provider-specific tokenizer."""
    character_count = sum(len(message.content) for message in messages)
    content_tokens = ceil(character_count / characters_per_token)
    return content_tokens + (4 * len(messages))


def fit_messages(
    *,
    system_prompt: str,
    history: Sequence[Message],
    current: Message,
    policy: ContextPolicy,
) -> list[Message]:
    """Fit a chat request by retaining the newest complete conversation turns."""
    system = Message(role="system", content=system_prompt)
    budget = policy.input_character_budget
    mandatory_overhead = _message_overhead_chars(2, policy)
    mandatory_content_budget = max(1, budget - mandatory_overhead)

    system_content = system.content
    current_content = current.content
    if len(system_content) + len(current_content) > mandatory_content_budget:
        current_budget = max(
            1,
            mandatory_content_budget - min(
                len(system_content),
                mandatory_content_budget // 2,
            ),
        )
        current_content = truncate_text(current_content, current_budget)
        system_budget = max(1, mandatory_content_budget - len(current_content))
        system_content = truncate_text(system_content, system_budget)

    system = Message(role="system", content=system_content)
    current = Message(role=current.role, content=current_content)
    remaining = budget - _messages_character_cost([system, current], policy)

    candidates = (
        list(history[-policy.recent_turns * 2 :])
        if policy.recent_turns
        else []
    )
    turns = [candidates[index : index + 2] for index in range(0, len(candidates), 2)]
    retained_turns_reversed: list[list[Message]] = []
    for turn in reversed(turns):
        cost = _messages_character_cost(turn, policy)
        if cost > remaining:
            break
        retained_turns_reversed.append(turn)
        remaining -= cost

    retained = [
        message
        for turn in reversed(retained_turns_reversed)
        for message in turn
    ]
    omitted = len(history) - len(retained)
    if omitted:
        marker = Message(
            role="system",
            content=f"[{omitted} earlier conversation message(s) omitted.]",
        )
        marker_cost = _messages_character_cost([marker], policy)
        while retained and marker_cost > remaining:
            removed_turn = retained[:2]
            del retained[:2]
            remaining += _messages_character_cost(removed_turn, policy)
            omitted += len(removed_turn)
            marker = Message(
                role="system",
                content=f"[{omitted} earlier conversation message(s) omitted.]",
            )
            marker_cost = _messages_character_cost([marker], policy)
        if marker_cost <= remaining:
            retained.insert(0, marker)

    return [system, *retained, current]


def truncate_text(text: str, max_chars: int) -> str:
    """Keep both ends of oversized text because diagnostics often end in errors."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_MARKER):
        return text[:max_chars]
    remaining = max_chars - len(_TRUNCATION_MARKER)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + _TRUNCATION_MARKER + text[-tail:]


def _messages_character_cost(
    messages: Sequence[Message],
    policy: ContextPolicy,
) -> int:
    return sum(len(message.content) for message in messages) + (
        _message_overhead_chars(len(messages), policy)
    )


def _message_overhead_chars(count: int, policy: ContextPolicy) -> int:
    return int(4 * policy.characters_per_token * count)
