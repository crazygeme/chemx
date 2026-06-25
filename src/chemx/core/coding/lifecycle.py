"""Stateful action policies for workspace workflows."""

from dataclasses import dataclass, field
from typing import Protocol

from .action import ActionKind, ActionResult, CodingAction


class WorkflowLifecycle(Protocol):
    """Validate proposed actions and observe completed workspace results."""

    def validate(self, action: CodingAction) -> None:
        """Reject an action that violates workflow ordering."""

    def record(self, result: ActionResult) -> None:
        """Update lifecycle state from one completed action."""


@dataclass
class DocumentLifecycle:
    """Enforce research, draft, review, and finish ordering."""

    versions: dict[str, int] = field(default_factory=dict)
    reviewed_versions: dict[str, int] = field(default_factory=dict)

    @property
    def drafting_started(self) -> bool:
        return bool(self.versions)

    @property
    def latest_versions_reviewed(self) -> bool:
        return all(
            self.reviewed_versions.get(path) == version
            for path, version in self.versions.items()
        )

    def validate(self, action: CodingAction) -> None:
        supported = {
            ActionKind.LIST_FILES,
            ActionKind.READ_FILE,
            ActionKind.SEARCH_TEXT,
            ActionKind.REPLACE_TEXT,
            ActionKind.CREATE_FILE,
            ActionKind.WRITE_FILE,
            ActionKind.FINISH,
        }
        if action.kind not in supported:
            raise ValueError(
                f"Action {action.kind.value} is unavailable in document workflows."
            )
        if not self.drafting_started:
            if action.kind is ActionKind.FINISH:
                raise ValueError(
                    "Document workflow cannot finish before a document is drafted."
                )
            return

        if action.kind in {ActionKind.LIST_FILES, ActionKind.SEARCH_TEXT}:
            raise ValueError(
                "Document source research must be completed before drafting."
            )
        if action.kind is ActionKind.READ_FILE:
            path = action.path or ""
            if path not in self.versions:
                raise ValueError(
                    "After drafting, read only a changed document path; "
                    "unrelated source research is closed."
                )
            if self.reviewed_versions.get(path) == self.versions[path]:
                raise ValueError(
                    f"Document {path} has already been reviewed at its current "
                    "version; revise it or finish."
                )
        if action.kind in {ActionKind.WRITE_FILE, ActionKind.REPLACE_TEXT}:
            path = action.path or ""
            if path not in self.versions:
                raise ValueError(
                    "After drafting begins, revisions must target a known "
                    "document path."
                )
        if action.kind is ActionKind.FINISH and not self.latest_versions_reviewed:
            raise ValueError(
                "Document workflow cannot finish until every latest document "
                "version has been read once for review."
            )

    def record(self, result: ActionResult) -> None:
        if not result.success:
            return
        action = result.action
        path = action.path
        if path is None:
            return
        if action.kind in {
            ActionKind.CREATE_FILE,
            ActionKind.WRITE_FILE,
            ActionKind.REPLACE_TEXT,
        }:
            self.versions[path] = self.versions.get(path, 0) + 1
        elif action.kind is ActionKind.READ_FILE and path in self.versions:
            self.reviewed_versions[path] = self.versions[path]
