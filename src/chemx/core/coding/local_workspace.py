"""Local coding workspace composed from reusable, policy-aware tools.

The workspace translates domain-level ``CodingAction`` values into common tool
calls. It supports ordinary directories and Git repositories. Git enriches
inspection and change reporting but is not required for file operations.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ...tools import (
    BashTool,
    CommandTool,
    EditTool,
    GitTool,
    ListTool,
    ReadTool,
    SearchTool,
    WorkspacePaths,
    WriteTool,
)
from .action import ActionKind, ActionResult, CodingAction

logger = logging.getLogger(__name__)


@dataclass
class LocalWorkspace:
    """Execute coding actions inside one bounded local directory.

    Paths are resolved by ``WorkspacePaths`` and cannot escape ``root``.
    Commands must exactly match the allowlist. Bash remains disabled unless the
    caller explicitly enables it. These restrictions belong to the workspace
    rather than the model so execution policy cannot be bypassed by prompting.
    """

    root: Path
    allowed_commands: tuple[tuple[str, ...], ...] = ()
    enable_bash: bool = False
    max_files: int = 100
    max_output_chars: int = 20_000
    command_timeout: float = 120.0
    paths: WorkspacePaths = field(init=False)
    files: ListTool = field(init=False)
    reader: ReadTool = field(init=False)
    writer: WriteTool = field(init=False)
    editor: EditTool = field(init=False)
    searcher: SearchTool = field(init=False)
    commands: CommandTool = field(init=False)
    bash: BashTool = field(init=False)
    git: GitTool = field(init=False)
    _initial_snapshot: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        """Construct tools and capture a baseline for non-Git change reports."""
        self.paths = WorkspacePaths(self.root)
        self.root = self.paths.root
        self.files = ListTool(self.paths)
        self.reader = ReadTool(self.paths, max_chars=self.max_output_chars)
        self.writer = WriteTool(self.paths)
        self.editor = EditTool(self.paths)
        self.searcher = SearchTool(self.paths, max_chars=self.max_output_chars)
        self.commands = CommandTool(
            self.paths,
            allowed_commands=self.allowed_commands,
            timeout=self.command_timeout,
            max_chars=self.max_output_chars,
        )
        self.bash = BashTool(
            self.paths,
            enabled=self.enable_bash,
            timeout=self.command_timeout,
        )
        self.git = GitTool(self.paths, max_chars=self.max_output_chars)
        self._initial_snapshot = self._snapshot()
        logger.info(
            "workspace initialized root=%s type=%s files=%d commands=%d bash=%s",
            self.root,
            "git" if self.git.is_repository() else "directory",
            len(self._initial_snapshot),
            len(self.allowed_commands),
            self.enable_bash,
        )

    def inspect(self, task: str) -> str:
        """Return status and a bounded listing without reading file contents."""
        logger.info("workspace inspection started task=%r", task)
        listed_files = self.files.run(max_files=self.max_files)
        is_git = self.git.is_repository()
        lines = [
            f"Workspace: {self.root}",
            f"Task: {task}",
            f"Workspace type: {'git' if is_git else 'directory'}",
        ]
        if is_git:
            lines.append(f"Git status:\n{self.git.status()}")
        lines.extend(["Files:", *listed_files])
        logger.debug(
            "workspace inspection completed listed_files=%d git=%s",
            len(listed_files),
            is_git,
        )
        return "\n".join(lines)

    def execute(self, action: CodingAction) -> ActionResult:
        """Execute one validated action and convert exceptions to observations."""
        action.validate()
        logger.info("workspace action started kind=%s", action.kind.value)
        logger.debug(
            "workspace action metadata path=%r command=%r argc=%d",
            action.path,
            action.command[0] if action.command else None,
            len(action.command),
        )
        handlers = {
            ActionKind.LIST_FILES: lambda: "\n".join(
                self.files.run(max_files=self.max_files)
            ),
            ActionKind.READ_FILE: lambda: self.reader.run(action.path or ""),
            ActionKind.SEARCH_TEXT: lambda: self.searcher.run(action.query or ""),
            ActionKind.REPLACE_TEXT: lambda: self.editor.replace_exact(
                action.path or "",
                action.old_text or "",
                action.new_text or "",
            ),
            ActionKind.CREATE_FILE: lambda: self.writer.create(
                action.path or "",
                action.content or "",
            ),
            ActionKind.WRITE_FILE: lambda: self.writer.write(
                action.path or "",
                action.content or "",
            ),
            ActionKind.RUN_COMMAND: lambda: self.commands.run(action.command),
            ActionKind.BASH: lambda: self.bash.run(action.script or ""),
            ActionKind.GIT_STATUS: self.git.status,
            ActionKind.FINISH: lambda: action.message or "Finished.",
        }
        try:
            output = handlers[action.kind]()
            result = ActionResult(action, True, self._truncate(output))
            logger.info(
                "workspace action succeeded kind=%s output_chars=%d",
                action.kind.value,
                len(result.output),
            )
            return result
        except (OSError, RuntimeError, ValueError) as error:
            logger.error(
                "workspace action failed kind=%s error=%s",
                action.kind.value,
                error,
            )
            return ActionResult(action, False, str(error))

    def changes(self) -> str:
        """Return Git diff when available, otherwise summarize file changes."""
        if self.git.is_repository():
            changes = self.git.diff()
            logger.debug("workspace Git change report chars=%d", len(changes))
            return changes

        current = self._snapshot()
        created = sorted(current.keys() - self._initial_snapshot.keys())
        deleted = sorted(self._initial_snapshot.keys() - current.keys())
        modified = sorted(
            path
            for path in current.keys() & self._initial_snapshot.keys()
            if current[path] != self._initial_snapshot[path]
        )
        lines = [
            *(f"created: {path}" for path in created),
            *(f"modified: {path}" for path in modified),
            *(f"deleted: {path}" for path in deleted),
        ]
        changes = "\n".join(lines) or "(no changes)"
        logger.debug(
            "workspace filesystem change report created=%d modified=%d deleted=%d",
            len(created),
            len(modified),
            len(deleted),
        )
        return changes

    def _snapshot(self) -> dict[str, str]:
        """Hash visible files for non-Git before/after change detection."""
        snapshot = {}
        for relative_path in self.files.run(max_files=1_000_000):
            path = self.paths.resolve(relative_path)
            try:
                snapshot[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
        return snapshot

    def _truncate(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        return value[: self.max_output_chars] + "\n[truncated]"
