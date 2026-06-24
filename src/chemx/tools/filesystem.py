"""Filesystem tools constrained to one workspace root.

These tools perform mechanism only. They do not decide which files should be
read or changed, and they never resolve absolute or parent-escaping paths.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolve caller-supplied paths without allowing workspace escape."""

    root: Path

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Workspace does not exist: {root}")
        object.__setattr__(self, "root", root)

    def resolve(self, relative_path: str) -> Path:
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError("Workspace paths must be non-empty and relative.")
        resolved = (self.root / relative_path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {relative_path}")
        return resolved


@dataclass
class ListTool:
    """List regular, non-symlink files beneath a workspace root."""
    paths: WorkspacePaths

    def run(self, *, max_files: int = 100) -> list[str]:
        files = []
        for path in self.paths.root.rglob("*"):
            relative = path.relative_to(self.paths.root)
            if (
                not path.is_file()
                or path.is_symlink()
                or ".git" in relative.parts
            ):
                continue
            files.append(relative.as_posix())
        files.sort()
        logger.debug("listed workspace files count=%d limit=%d", len(files), max_files)
        return files[:max_files]


@dataclass
class ReadTool:
    """Read one UTF-8 text file with bounded output."""
    paths: WorkspacePaths
    max_chars: int = 20_000

    def run(self, relative_path: str) -> str:
        path = self.paths.resolve(relative_path)
        if not path.is_file():
            raise ValueError(f"File does not exist: {relative_path}")
        content = path.read_text(encoding="utf-8")
        if len(content) > self.max_chars:
            return content[: self.max_chars] + "\n[truncated]"
        return content


@dataclass
class WriteTool:
    """Create or deliberately replace UTF-8 text files."""
    paths: WorkspacePaths

    def create(self, relative_path: str, content: str) -> str:
        path = self.paths.resolve(relative_path)
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing file: {relative_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Created {relative_path}."

    def write(self, relative_path: str, content: str) -> str:
        path = self.paths.resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {relative_path}."


@dataclass
class EditTool:
    """Apply narrow edits that require one exact source-text match."""
    paths: WorkspacePaths

    def replace_exact(
        self,
        relative_path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        path = self.paths.resolve(relative_path)
        if not path.is_file():
            raise ValueError(f"File does not exist: {relative_path}")
        content = path.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise ValueError(
                f"Expected exact text once in {relative_path}; found {occurrences}."
            )
        path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Replaced exact text in {relative_path}."
