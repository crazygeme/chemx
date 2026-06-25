"""Provider-independent conversational agent primitive.

This class contains only conversation mechanics: input validation, system
prompt injection, model invocation, and history retention. Specialized agents
compose it with domain-specific state machines and tool execution.
"""

import logging
from dataclasses import dataclass, field

from ..backends import Message, ModelBackend
from .context import ContextPolicy, fit_messages

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


@dataclass
class Agent:
    """Maintain ordered chat history and execute one model turn at a time."""

    model: ModelBackend
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history: list[Message] = field(default_factory=list)
    context_policy: ContextPolicy | None = None

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
        messages = fit_messages(
            system_prompt=self.system_prompt,
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
