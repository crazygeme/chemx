"""Bounded, pure-Python text search within a workspace."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .filesystem import ListTool, WorkspacePaths

logger = logging.getLogger(__name__)

LineMatcher = Callable[[str], Iterable[re.Match[str] | tuple[int, int]]]


@dataclass
class SearchTool:
    """Search readable text files while honoring workspace ignore rules."""

    paths: WorkspacePaths
    max_chars: int = 20_000

    def run(self, query: str) -> str:
        if not query:
            raise ValueError("Search query cannot be empty.")

        logger.debug("text search started query=%r", query)
        files = ListTool(self.paths).iter_files()
        try:
            pattern = re.compile(query)
        except re.error:
            matches = []
        else:
            matches = self._search(files, pattern.finditer)

        if not matches:
            normalized_query = query.casefold()
            matches = self._search(
                files,
                lambda line: _find_substrings(line.casefold(), normalized_query),
            )

        output = "\n".join(matches) if matches else "(no matches)"
        logger.debug("text search completed output_chars=%d", len(output))
        if len(output) > self.max_chars:
            return output[: self.max_chars] + "\n[truncated]"
        return output

    def _search(
        self,
        files: list[Path],
        find_matches: LineMatcher,
    ) -> list[str]:
        results = []
        for path in files:
            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                    errors="strict",
                ) as source:
                    for line_number, line in enumerate(source, start=1):
                        text = line.rstrip("\r\n")
                        if any(find_matches(text)):
                            relative = path.relative_to(self.paths.root).as_posix()
                            results.append(f"{relative}:{line_number}:{text}")
            except (OSError, UnicodeError):
                logger.debug("skipping unreadable search file path=%s", path)
        return results


def _find_substrings(text: str, query: str) -> Iterable[tuple[int, int]]:
    """Yield non-overlapping literal substring spans."""
    start = 0
    while True:
        index = text.find(query, start)
        if index == -1:
            return
        end = index + len(query)
        yield index, end
        start = end
