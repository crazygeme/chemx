"""Workspace-bound interactive sessions for the coding agent."""

import re
from dataclasses import dataclass

from .agent import CodingAgent
from .workspace import CodingWorkspace

_CONVERSATIONAL_MESSAGES = {
    "awesome",
    "good",
    "good job",
    "got it",
    "great",
    "great job",
    "looks good",
    "nice",
    "nice work",
    "ok",
    "okay",
    "perfect",
    "sounds good",
    "thank you",
    "thanks",
    "well done",
}


@dataclass
class CodingSession:
    """Run each interactive prompt as a complete coding workflow."""

    agent: CodingAgent
    workspace: CodingWorkspace
    max_steps: int = 20

    def run(self, user_input: str) -> str:
        """Run conversational acknowledgements or a workspace-backed task."""
        if _is_conversational_message(user_input):
            return self.agent.run(user_input)
        return self.agent.run_workflow(
            user_input,
            self.workspace,
            max_steps=self.max_steps,
        )

    def clear(self) -> None:
        """Clear conversation history retained by the underlying agent."""
        self.agent.clear()


def create_coding_session(
    agent: CodingAgent,
    workspace: CodingWorkspace,
    *,
    max_steps: int = 20,
) -> CodingSession:
    """Bind a coding agent to the policy used for interactive tasks."""
    return CodingSession(
        agent=agent,
        workspace=workspace,
        max_steps=max_steps,
    )


def _is_conversational_message(user_input: str) -> bool:
    """Recognize short social replies without guessing about task-like text."""
    normalized = re.sub(r"[^\w\s]", "", user_input.casefold())
    normalized = " ".join(normalized.split())
    return normalized in _CONVERSATIONAL_MESSAGES
