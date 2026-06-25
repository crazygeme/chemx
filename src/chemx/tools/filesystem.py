"""Filesystem tools constrained to one workspace root.

These tools perform mechanism only. They do not decide which files should be
read or changed, and they never resolve absolute or parent-escaping paths.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IgnoreRule:
    """One compiled rule from a workspace .gitignore file."""

    base: Path
    pattern: re.Pattern[str]
    negated: bool

    def matches(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.base).as_posix()
        except ValueError:
            return False
        return bool(self.pattern.fullmatch(relative))


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

    def run(
        self,
        relative_path: str | None = None,
        *,
        max_files: int = 100,
    ) -> list[str]:
        """List non-ignored files beneath an optional relative directory."""
        base = (
            self.paths.root
            if relative_path is None
            else self.paths.resolve(relative_path)
        )
        if not base.is_dir():
            raise ValueError(f"Directory does not exist: {relative_path}")

        rules = self._load_ignore_rules()
        files = self._walk_files(base, rules)
        files.sort()
        logger.debug("listed workspace files count=%d limit=%d", len(files), max_files)
        return files[:max_files]

    def _walk_files(
        self,
        base: Path,
        rules: list[_IgnoreRule],
    ) -> list[str]:
        files = []
        for path in base.rglob("*"):
            relative = path.relative_to(self.paths.root)
            if (
                not path.is_file()
                or path.is_symlink()
                or ".git" in relative.parts
            ):
                continue
            if self._is_ignored(path, rules):
                continue
            files.append(relative.as_posix())
        return files

    def _load_ignore_rules(self) -> list[_IgnoreRule]:
        rules: list[_IgnoreRule] = []
        ignore_files = sorted(
            self.paths.root.rglob(".gitignore"),
            key=lambda path: (len(path.relative_to(self.paths.root).parts), str(path)),
        )
        for ignore_file in ignore_files:
            if ".git" in ignore_file.relative_to(self.paths.root).parts:
                continue
            if (
                ignore_file.parent != self.paths.root
                and self._is_ignored(ignore_file.parent, rules, check_parents=False)
            ):
                continue
            try:
                lines = ignore_file.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line in lines:
                rule = _parse_ignore_rule(ignore_file.parent, line)
                if rule is not None:
                    rules.append(rule)
        return rules

    def _is_ignored(
        self,
        path: Path,
        rules: list[_IgnoreRule],
        *,
        check_parents: bool = True,
    ) -> bool:
        ignored = _rules_ignore(path, rules)
        if not check_parents or ignored:
            return ignored
        parent = path.parent
        while parent != self.paths.root and self.paths.root in parent.parents:
            if _rules_ignore(parent, rules):
                return True
            parent = parent.parent
        return False


def _parse_ignore_rule(base: Path, line: str) -> _IgnoreRule | None:
    """Parse the common Git ignore syntax used by project .gitignore files."""
    line = line.rstrip()
    if not line or line.startswith("#"):
        return None

    negated = line.startswith("!")
    if negated:
        line = line[1:]
    elif line.startswith(r"\!"):
        line = line[1:]
    if line.startswith(r"\#"):
        line = line[1:]
    if not line:
        return None

    directory_only = line.endswith("/")
    if directory_only:
        line = line.rstrip("/")
    anchored = line.startswith("/") or "/" in line
    line = line.lstrip("/")
    expression = _translate_gitignore_glob(line)
    prefix = "^" if anchored else r"(?:^|.*/)"
    suffix = r"(?:/.*)?$"
    return _IgnoreRule(
        base=base,
        pattern=re.compile(prefix + expression + suffix),
        negated=negated,
    )


def _translate_gitignore_glob(pattern: str) -> str:
    """Translate Git-style *, **, ?, and character classes to a regex."""
    result: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                if index < len(pattern) and pattern[index] == "/":
                    result.append(r"(?:.*/)?")
                    index += 1
                else:
                    result.append(".*")
                continue
            result.append(r"[^/]*")
        elif character == "?":
            result.append(r"[^/]")
        elif character == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                result.append(r"\[")
            else:
                content = pattern[index + 1 : end]
                if content.startswith("!"):
                    content = "^" + content[1:]
                result.append("[" + content.replace("\\", r"\\") + "]")
                index = end
        elif character == "\\" and index + 1 < len(pattern):
            index += 1
            result.append(re.escape(pattern[index]))
        else:
            result.append(re.escape(character))
        index += 1
    return "".join(result)


def _rules_ignore(path: Path, rules: list[_IgnoreRule]) -> bool:
    ignored = False
    for rule in rules:
        if rule.matches(path):
            ignored = not rule.negated
    return ignored


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
