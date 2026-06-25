"""Workspace-bound interactive sessions for the coding agent."""

from dataclasses import dataclass

from .agent import CodingAgent
from .router import WorkflowKind, WorkflowRouter
from .workflow import DOCUMENT_WORKFLOW
from .workspace import CodingWorkspace


@dataclass
class CodingSession:
    """Run each interactive prompt as a complete coding workflow."""

    agent: CodingAgent
    workspace: CodingWorkspace
    max_steps: int = 20
    router: WorkflowRouter | None = None
    document_agent: CodingAgent | None = None

    def __post_init__(self) -> None:
        if self.router is None:
            self.router = WorkflowRouter(
                self.agent.model,
                context_policy=self.agent.context_policy,
            )
        if self.document_agent is None:
            self.document_agent = CodingAgent(
                model=self.agent.model,
                system_prompt=DOCUMENT_WORKFLOW.system_prompt,
                workflow=DOCUMENT_WORKFLOW,
                context_policy=self.agent.context_policy,
                progress_output=self.agent.progress_output,
            )

    def run(self, user_input: str) -> str:
        """Classify input and run the selected bounded workflow."""
        assert self.router is not None
        route = self.router.classify(user_input)
        if route.kind is WorkflowKind.CONVERSATION:
            return self.agent.run(route.objective)
        if route.kind is WorkflowKind.DOCUMENT:
            assert self.document_agent is not None
            return self.document_agent.run_workflow(
                route.objective,
                self.workspace,
                max_steps=self.max_steps,
            )
        return self.agent.run_workflow(
            route.objective, self.workspace, max_steps=self.max_steps
        )

    def clear(self) -> None:
        """Clear conversation history retained by the underlying agent."""
        self.agent.clear()
        if self.document_agent is not None:
            self.document_agent.clear()


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
