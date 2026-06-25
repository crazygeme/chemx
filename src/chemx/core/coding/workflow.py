"""Prompt profiles for workspace-backed plan/action workflows."""

from dataclasses import dataclass
from typing import Callable, Sequence

from ..context import ContextPolicy
from .action import ActionResult
from .lifecycle import DocumentLifecycle, WorkflowLifecycle
from .plan import (
    CodingPlan,
    build_action_prompt,
    build_plan_prompt,
    build_summary_prompt,
)


PlanPromptBuilder = Callable[[str, str], str]
ActionPromptBuilder = Callable[
    [str, CodingPlan, Sequence[ActionResult], ContextPolicy, str | None],
    str,
]
SummaryPromptBuilder = Callable[
    [str, CodingPlan, Sequence[ActionResult], str, ContextPolicy, str | None],
    str,
]
LifecycleFactory = Callable[[], WorkflowLifecycle]


@dataclass(frozen=True)
class WorkflowProfile:
    """Prompts and display metadata for one workspace workflow."""

    name: str
    system_prompt: str
    build_plan_prompt: PlanPromptBuilder
    build_action_prompt: ActionPromptBuilder
    build_summary_prompt: SummaryPromptBuilder
    lifecycle_factory: LifecycleFactory | None = None


def _coding_action_prompt(
    task: str,
    plan: CodingPlan,
    observations: Sequence[ActionResult],
    policy: ContextPolicy,
    observation_summary: str | None,
) -> str:
    return build_action_prompt(
        task=task,
        plan=plan,
        observations=observations,
        context_policy=policy,
        observation_summary=observation_summary,
    )


def _coding_summary_prompt(
    task: str,
    plan: CodingPlan,
    observations: Sequence[ActionResult],
    diff: str,
    policy: ContextPolicy,
    observation_summary: str | None,
) -> str:
    return build_summary_prompt(
        task=task,
        plan=plan,
        observations=observations,
        diff=diff,
        context_policy=policy,
        observation_summary=observation_summary,
    )


CODING_WORKFLOW = WorkflowProfile(
    name="Coding",
    system_prompt="""You are a software engineering agent.
Inspect relevant code before editing. Keep plans separate from tool actions.
Make narrow, justified changes, preserve unrelated behavior, and verify work
with appropriate commands. Never claim a file edit or test result unless a
workspace observation confirms it.""",
    build_plan_prompt=build_plan_prompt,
    build_action_prompt=_coding_action_prompt,
    build_summary_prompt=_coding_summary_prompt,
)


def build_document_plan_prompt(task: str, context: str) -> str:
    """Ask for a source-aware document production plan."""
    return (
        "Create a concise document-production plan. Do not draft the document "
        "yet.\n"
        f"Task:\n{task}\n\n"
        f"Workspace overview:\n{context}\n\n"
        "Identify the intended audience, document format and destination, "
        "source files to inspect, required sections, and review strategy. "
        "Clearly label assumptions."
    )


def build_document_action_prompt(
    task: str,
    plan: CodingPlan,
    observations: Sequence[ActionResult],
    policy: ContextPolicy,
    observation_summary: str | None,
) -> str:
    """Select one workspace action for research, drafting, or review."""
    from .context import format_observations

    observation_text = format_observations(
        observations,
        policy,
        summary=observation_summary,
    )
    return (
        "Choose exactly one next document-production action as a JSON object.\n"
        f"Task:\n{task}\n\n"
        f"Plan:\n{plan.text}\n\n"
        f"Observations:\n{observation_text}\n\n"
        "Supported forms:\n"
        '{"kind":"list_files","path":"optional/relative/directory"}\n'
        '{"kind":"read_file","path":"relative/path"}\n'
        '{"kind":"search_text","query":"source text or symbol"}\n'
        '{"kind":"replace_text","path":"document.md","old_text":"exact old",'
        '"new_text":"exact new"}\n'
        '{"kind":"create_file","path":"document.md","content":"complete content"}\n'
        '{"kind":"write_file","path":"document.md","content":"complete content"}\n'
        '{"kind":"finish","message":"document and review are complete"}\n'
        "Return JSON only. Scope list_files to a directory identified by the "
        "plan whenever possible; omit path only when the whole workspace must "
        "be inspected. Follow this lifecycle strictly: research source files, "
        "draft or revise the document, read the changed document once to review "
        "that version, then finish. Complete all source research before the "
        "first create_file, write_file, or replace_text action. After drafting, "
        "do not list, search, or read unrelated source files. Do not read an "
        "unchanged document version more than once. "
        "Use create_file for a new document, replace_text for narrow revisions, "
        "and write_file only when a complete rewrite is justified. Finish only "
        "after the latest document version has been read and checked against "
        "the task and plan."
    )


def build_document_summary_prompt(
    task: str,
    plan: CodingPlan,
    observations: Sequence[ActionResult],
    diff: str,
    policy: ContextPolicy,
    observation_summary: str | None,
) -> str:
    """Summarize document work using observed results only."""
    from ..context import truncate_text
    from .context import format_observations

    results = format_observations(
        observations,
        policy,
        summary=observation_summary,
    )
    return (
        "Summarize the completed document task using only these workspace "
        "results. State the document path and important assumptions when they "
        "are present in the evidence.\n"
        f"Task:\n{task}\n\nPlan:\n{plan.text}\n\n"
        f"Workspace results:\n{results}\n\n"
        f"Final changes:\n{truncate_text(diff or '(no changes)', policy.max_diff_chars)}"
    )


DOCUMENT_WORKFLOW = WorkflowProfile(
    name="Document",
    system_prompt="""You are a technical writing agent.
Research relevant workspace sources before drafting. Keep planning separate
from actions. Produce clear, accurate documents for the stated audience and
format. Preserve unrelated files. Never claim a source was inspected or a file
was written unless a workspace observation confirms it.""",
    build_plan_prompt=build_document_plan_prompt,
    build_action_prompt=build_document_action_prompt,
    build_summary_prompt=build_document_summary_prompt,
    lifecycle_factory=DocumentLifecycle,
)
