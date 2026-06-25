"""Workspace-bound interactive sessions for the coding agent."""

from dataclasses import dataclass

from .agent import CodingAgent
from .workspace import CodingWorkspace


@dataclass
class CodingSession:
    """Run each interactive prompt as a complete coding workflow."""

    agent: CodingAgent
    workspace: CodingWorkspace
    max_steps: int = 20

    def run(self, user_input: str) -> str:
        """Execute one workspace-backed coding task."""
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
