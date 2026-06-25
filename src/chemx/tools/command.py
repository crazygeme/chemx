"""Allowlisted process execution and explicit opt-in Bash execution."""

import logging
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

from .filesystem import WorkspacePaths

logger = logging.getLogger(__name__)

CommandApproval = Callable[[tuple[str, ...]], bool]


@dataclass
class CommandTool:
    """Execute argument vectors after explicit approval."""
    paths: WorkspacePaths
    approval: CommandApproval | None = None
    timeout: float = 120.0
    max_chars: int = 20_000

    def run(self, command: Sequence[str]) -> str:
        normalized = tuple(command)
        if self.approval is None or not self.approval(normalized):
            raise ValueError(f"Command was not approved: {' '.join(normalized)}")
        logger.info(
            "running approved command executable=%s argc=%d",
            normalized[0],
            len(normalized),
        )
        return self._execute(normalized)

    def _execute(self, command: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                list(command),
                cwd=self.paths.root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(f"Command is not installed: {command[0]}") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Command timed out after {self.timeout:g} seconds: "
                f"{' '.join(command)}"
            ) from error
        output = "\n".join(
            value
            for value in (result.stdout.strip(), result.stderr.strip())
            if value
        ) or f"Command exited with status {result.returncode}."
        if result.returncode != 0:
            logger.error(
                "command failed status=%d executable=%s argc=%d",
                result.returncode,
                command[0],
                len(command),
            )
            raise RuntimeError(output)
        logger.info(
            "command completed status=0 executable=%s argc=%d",
            command[0],
            len(command),
        )
        if len(output) > self.max_chars:
            return output[: self.max_chars] + "\n[truncated]"
        return output


@dataclass
class BashTool:
    """Explicit shell tool, disabled unless the caller opts in."""

    paths: WorkspacePaths
    enabled: bool = False
    timeout: float = 120.0

    def run(self, script: str) -> str:
        if not self.enabled:
            raise ValueError("Bash tool is disabled for this workspace.")
        logger.info("running enabled Bash script chars=%d", len(script))
        command = CommandTool(self.paths, timeout=self.timeout)
        return command._execute(("bash", "-lc", script))
