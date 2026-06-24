"""Bounded text search implemented with ripgrep."""

import logging
import subprocess
from dataclasses import dataclass

from .filesystem import WorkspacePaths

logger = logging.getLogger(__name__)


@dataclass
class SearchTool:
    paths: WorkspacePaths
    max_chars: int = 20_000

    def run(self, query: str) -> str:
        logger.debug("text search started query=%r", query)
        try:
            result = subprocess.run(
                ["rg", "-n", "--hidden", "--glob", "!.git", query, "."],
                cwd=self.paths.root,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Command is not installed: rg") from error
        if result.returncode not in {0, 1}:
            raise RuntimeError(result.stderr.strip() or "Search failed.")
        output = result.stdout or "(no matches)"
        logger.debug("text search completed output_chars=%d", len(output))
        if len(output) > self.max_chars:
            return output[: self.max_chars] + "\n[truncated]"
        return output
