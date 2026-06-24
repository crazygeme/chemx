"""Optional, read-only Git inspection tools."""

import logging
import subprocess
from dataclasses import dataclass

from .filesystem import WorkspacePaths

logger = logging.getLogger(__name__)


@dataclass
class GitTool:
    """Inspect repository status and changes without mutating Git state."""
    paths: WorkspacePaths
    max_chars: int = 40_000

    def is_repository(self) -> bool:
        try:
            return self._run("rev-parse", "--is-inside-work-tree").returncode == 0
        except RuntimeError:
            return False

    def status(self) -> str:
        self._require_repository()
        status = self._run("status", "--short").stdout.strip() or "(clean)"
        logger.debug("Git status read chars=%d", len(status))
        return status

    def diff(self) -> str:
        self._require_repository()
        unstaged = self._run("diff").stdout
        staged = self._run("diff", "--cached").stdout
        output = "\n".join(part for part in (staged, unstaged) if part)
        if len(output) > self.max_chars:
            return output[: self.max_chars] + "\n[truncated]"
        logger.debug("Git diff read chars=%d", len(output))
        return output

    def _require_repository(self) -> None:
        if not self.is_repository():
            raise ValueError(f"Workspace is not a Git repository: {self.paths.root}")

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=self.paths.root,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Command is not installed: git") from error
