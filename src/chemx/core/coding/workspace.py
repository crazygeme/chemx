"""Workspace boundary used by the coding-agent workflow."""

from typing import Protocol

from .action import ActionResult, CodingAction


class CodingWorkspace(Protocol):
    """Execution environment for explicit coding actions."""

    def inspect(self, task: str) -> str:
        """Return a bounded initial repository overview."""

    def execute(self, action: CodingAction) -> ActionResult:
        """Execute one structured action and return its observation."""

    def changes(self) -> str:
        """Return a Git diff or filesystem change summary for final review."""
