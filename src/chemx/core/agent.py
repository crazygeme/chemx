"""Provider-independent conversational agent primitive.

This class contains only conversation mechanics: input validation, system
prompt injection, model invocation, and history retention. Specialized agents
compose it with domain-specific state machines and tool execution.
"""

import logging
from dataclasses import dataclass, field

from ..backends import Message, ModelBackend
from .context import ContextPolicy, estimate_message_tokens, fit_messages

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

CONTEXT_COMPACTION_SYSTEM_PROMPT = """Compress prior conversation context.
Preserve decisions, requirements, constraints, factual details, unresolved
questions, and commitments. Remove repetition and social filler. Do not add
facts or answer the current request. Return only the compact context."""


@dataclass
class Agent:
    """Maintain ordered chat history and execute one model turn at a time."""

    model: ModelBackend
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history: list[Message] = field(default_factory=list)
    context_policy: ContextPolicy | None = None
    context_summary: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.context_policy is None:
            self.context_policy = ContextPolicy.from_backend(self.model)

    def run(self, user_input: str) -> str:
        """Run one agent turn and return the assistant response."""
        text = user_input.strip()
        if not text:
            raise ValueError("User input cannot be empty.")

        user_message = Message(role="user", content=text)
        assert self.context_policy is not None
        self._compact_history_if_needed(user_message)
        messages = fit_messages(
            system_prompt=self._system_prompt_with_context(),
            history=self.history,
            current=user_message,
            policy=self.context_policy,
        )

        logger.debug(
            "conversation model turn started history_messages=%d",
            len(self.history),
        )
        response = self.model.complete(messages).strip()
        if not response:
            raise RuntimeError("The model returned an empty response.")

        self.history.extend(
            [
                user_message,
                Message(role="assistant", content=response),
            ]
        )
        logger.debug(
            "conversation model turn completed history_messages=%d",
            len(self.history),
        )
        return response

    def clear(self) -> None:
        """Clear the conversation while preserving agent configuration."""
        logger.info("conversation history cleared messages=%d", len(self.history))
        self.history.clear()
        self.context_summary = None

    def _compact_history_if_needed(self, current: Message) -> None:
        """Use the model to summarize older turns before they would be omitted."""
        assert self.context_policy is not None
        if not self.history:
            return

        recent_limit = self.context_policy.recent_turns * 2
        full_messages = [
            Message(role="system", content=self._system_prompt_with_context()),
            *self.history,
            current,
        ]
        over_budget = (
            estimate_message_tokens(
                full_messages,
                characters_per_token=self.context_policy.characters_per_token,
            )
            > self.context_policy.input_token_budget
        )
        over_retention_limit = len(self.history) > recent_limit
        if not over_budget and not over_retention_limit:
            return

        keep_count = min(len(self.history), recent_limit)
        if over_budget and keep_count == len(self.history):
            keep_count = max(0, keep_count - 2)
        older = self.history[:-keep_count] if keep_count else list(self.history)
        if not older:
            return

        source_parts = []
        if self.context_summary:
            source_parts.append(f"Existing compact context:\n{self.context_summary}")
        source_parts.append(
            "Conversation to incorporate:\n"
            + "\n".join(
                f"{message.role}: {message.content}" for message in older
            )
        )
        prompt = "\n\n".join(source_parts)
        compaction_messages = fit_messages(
            system_prompt=CONTEXT_COMPACTION_SYSTEM_PROMPT,
            history=(),
            current=Message(role="user", content=prompt),
            policy=self.context_policy,
        )
        summary = self.model.complete(compaction_messages).strip()
        if not summary:
            raise RuntimeError("The model returned an empty context summary.")

        self.context_summary = summary
        self.history = self.history[-keep_count:] if keep_count else []
        logger.info(
            "conversation context compacted messages=%d summary_chars=%d",
            len(older),
            len(summary),
        )

    def _system_prompt_with_context(self) -> str:
        if not self.context_summary:
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            "Compact context from earlier conversation:\n"
            f"{self.context_summary}"
        )
