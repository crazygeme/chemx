"""Natural-language coding plans and prompt formatting."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ..context import ContextPolicy, truncate_text
from .action import ActionResult
from .context import format_observations


class PlanSource(str, Enum):
    """Origin of a coding plan."""

    MODEL = "model"
    USER = "user"


@dataclass(frozen=True)
class CodingPlan:
    """A non-executable description of intended coding work."""

    text: str
    source: PlanSource

    def __post_init__(self) -> None:
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Coding plan cannot be empty.")
        object.__setattr__(self, "text", normalized)

    @classmethod
    def from_model(cls, text: str) -> "CodingPlan":
        return cls(text=text, source=PlanSource.MODEL)

    @classmethod
    def from_user(cls, text: str) -> "CodingPlan":
        if not text.strip():
            raise ValueError("User plan cannot be empty.")
        return cls(text=text, source=PlanSource.USER)


def build_plan_prompt(task: str, context: str) -> str:
    """Ask for a concise natural-language implementation plan."""
    return (
        "Create a concise implementation plan. Do not write code yet.\n"
        f"Task:\n{task}\n\n"
        f"Repository overview:\n{context}\n\n"
        "List the files or components to inspect, intended changes, and "
        "verification strategy. Clearly label assumptions."
    )


def build_action_prompt(
    *,
    task: str,
    plan: CodingPlan,
    observations: Sequence[ActionResult],
    context_policy: ContextPolicy | None = None,
    observation_summary: str | None = None,
) -> str:
    """Ask the model to select exactly one structured workspace action."""
    policy = context_policy or ContextPolicy()
    observation_text = format_observations(
        observations,
        policy,
        summary=observation_summary,
    )
    return (
        "Choose exactly one next coding action as a JSON object.\n"
        f"Task:\n{task}\n\n"
        f"Plan:\n{plan.text}\n\n"
        f"Observations:\n{observation_text}\n\n"
        "Supported forms:\n"
        '{"kind":"list_files"}\n'
        '{"kind":"read_file","path":"relative/path.py"}\n'
        '{"kind":"search_text","query":"symbol or text"}\n'
        '{"kind":"replace_text","path":"file.py","old_text":"exact old",'
        '"new_text":"exact new"}\n'
        '{"kind":"create_file","path":"file.py","content":"complete content"}\n'
        '{"kind":"write_file","path":"file.py","content":"complete content"}\n'
        '{"kind":"run_command","command":["python3","-m","unittest"]}\n'
        '{"kind":"git_status"}\n'
        '{"kind":"finish","message":"work and verification are complete"}\n'
        "Return JSON only. Read relevant files before editing. Use exact text "
        "replacement for narrow edits and write_file only when full replacement "
        "is justified. Use run_command with an argument array for compilation or "
        "verification; commands run directly without Bash and require user "
        "approval before execution. Bash is unavailable unless the workspace "
        "explicitly enables it."
    )


def build_summary_prompt(
    *,
    task: str,
    plan: CodingPlan,
    observations: Sequence[ActionResult],
    diff: str,
    context_policy: ContextPolicy | None = None,
    observation_summary: str | None = None,
) -> str:
    """Ask for a factual summary based on actual tool results."""
    policy = context_policy or ContextPolicy()
    results = format_observations(
        observations,
        policy,
        summary=observation_summary,
    )
    bounded_diff = (
        truncate_text(diff, policy.max_diff_chars)
        if diff
        else "(no changes)"
    )
    return (
        "Summarize the completed coding task using only these tool results.\n"
        f"Task:\n{task}\n\nPlan:\n{plan.text}\n\n"
        f"Tool results:\n{results}\n\n"
        f"Final diff:\n{bounded_diff}"
    )
