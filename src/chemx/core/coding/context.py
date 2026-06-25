"""Context compaction for coding workflow observations."""

from typing import Sequence

from ..context import ContextPolicy, truncate_text
from .action import ActionResult


def build_observation_compaction_prompt(
    previous_summary: str | None,
    observations: Sequence[ActionResult],
) -> str:
    """Ask for a factual, action-oriented compression of older tool results."""
    parts = [
        "Compress coding tool observations into durable working context.\n"
        "Preserve exact file paths, symbols, commands, edits, test outcomes, "
        "errors, constraints, and unresolved work. Distinguish successful and "
        "failed actions. Do not invent results or propose new actions. Return "
        "only the compact context."
    ]
    if previous_summary:
        parts.append(f"Existing compact context:\n{previous_summary}")
    parts.append(
        "Observations to incorporate:\n"
        + "\n\n".join(
            f"{result.action.kind.value} "
            f"({'ok' if result.success else 'failed'}):\n{result.output}"
            for result in observations
        )
    )
    return "\n\n".join(parts)


def format_observations(
    observations: Sequence[ActionResult],
    policy: ContextPolicy,
    summary: str | None = None,
) -> str:
    """Render recent observations within per-result and total character limits."""
    if not observations and not summary:
        return "No actions have run yet."

    candidates = observations[-policy.max_observations :]
    rendered_reversed: list[str] = []
    used = 0
    for result in reversed(candidates):
        output = truncate_text(result.output, policy.max_observation_chars)
        rendered = (
            f"{result.action.kind.value} "
            f"({'ok' if result.success else 'failed'}):\n{output}"
        )
        separator_cost = 2 if rendered_reversed else 0
        remaining = policy.max_observation_chars_total - used - separator_cost
        if remaining <= 0:
            break
        if len(rendered) > remaining:
            rendered = truncate_text(rendered, remaining)
        rendered_reversed.append(rendered)
        used += len(rendered) + separator_cost

    rendered = list(reversed(rendered_reversed))
    if summary:
        rendered.insert(0, f"Compact context from earlier observations:\n{summary}")
    else:
        omitted = len(observations) - len(rendered)
        if omitted:
            rendered.insert(0, f"[{omitted} earlier observation(s) omitted.]")
    return "\n\n".join(rendered)
