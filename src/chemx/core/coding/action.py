"""Structured actions available to a coding agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    """Supported coding workspace operations."""

    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    SEARCH_TEXT = "search_text"
    REPLACE_TEXT = "replace_text"
    CREATE_FILE = "create_file"
    WRITE_FILE = "write_file"
    RUN_COMMAND = "run_command"
    BASH = "bash"
    GIT_STATUS = "git_status"
    FINISH = "finish"


@dataclass(frozen=True)
class CodingAction:
    """One explicit tool operation selected by a model or user."""

    kind: ActionKind
    path: str | None = None
    query: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    content: str | None = None
    command: tuple[str, ...] = ()
    script: str | None = None
    message: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodingAction":
        """Parse and validate an action dictionary."""
        try:
            kind = ActionKind(value["kind"])
        except (KeyError, ValueError) as error:
            raise ValueError("Action requires a supported 'kind'.") from error

        command_value = value.get("command", ())
        if not isinstance(command_value, (list, tuple)):
            raise ValueError("Action command must be a list of arguments.")

        action = cls(
            kind=kind,
            path=_optional_string(value.get("path")),
            query=_optional_string(value.get("query")),
            old_text=_optional_string(value.get("old_text")),
            new_text=_optional_string(value.get("new_text")),
            content=_optional_string(value.get("content")),
            command=tuple(str(item) for item in command_value),
            script=_optional_string(value.get("script")),
            message=_optional_string(value.get("message")),
        )
        action.validate()
        return action

    def validate(self) -> None:
        """Validate fields required by this action kind."""
        required: dict[ActionKind, tuple[str, ...]] = {
            ActionKind.READ_FILE: ("path",),
            ActionKind.SEARCH_TEXT: ("query",),
            ActionKind.REPLACE_TEXT: ("path", "old_text", "new_text"),
            ActionKind.CREATE_FILE: ("path", "content"),
            ActionKind.WRITE_FILE: ("path", "content"),
            ActionKind.RUN_COMMAND: ("command",),
            ActionKind.BASH: ("script",),
            ActionKind.FINISH: ("message",),
        }
        for field_name in required.get(self.kind, ()):
            value = getattr(self, field_name)
            if value is None or value == () or value == "":
                raise ValueError(
                    f"Action {self.kind.value} requires '{field_name}'."
                )


@dataclass(frozen=True)
class ActionResult:
    """Observation produced by one workspace action."""

    action: CodingAction
    success: bool
    output: str


def parse_action(text: str) -> CodingAction:
    """Parse one JSON action, tolerating explanatory text around the object."""
    normalized = text.strip()
    if normalized.startswith("```"):
        first_newline = normalized.find("\n")
        final_fence = normalized.rfind("```")
        if first_newline != -1 and final_fence > first_newline:
            normalized = normalized[first_newline + 1 : final_fence].strip()

    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as error:
        actions = _extract_actions(normalized)
        if len(actions) == 1:
            return actions[0]
        if len(actions) > 1:
            raise ValueError(
                "Model response must contain exactly one JSON action."
            ) from error
        raise ValueError("Model action must contain a valid JSON object.") from error

    if not isinstance(value, dict):
        raise ValueError("Model action must be a JSON object.")
    return CodingAction.from_dict(value)


def _extract_actions(text: str) -> list[CodingAction]:
    """Find valid action objects embedded in otherwise non-JSON text."""
    decoder = json.JSONDecoder()
    actions = []
    offset = 0
    while True:
        object_start = text.find("{", offset)
        if object_start == -1:
            return actions
        try:
            value, consumed = decoder.raw_decode(text, object_start)
        except json.JSONDecodeError:
            offset = object_start + 1
            continue
        offset = consumed
        if not isinstance(value, dict):
            continue
        try:
            actions.append(CodingAction.from_dict(value))
        except ValueError:
            continue


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Action text fields must be strings.")
    return value
