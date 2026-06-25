"""Context compaction for coding workflow observations."""

from typing import Sequence

from ..context import ContextPolicy, truncate_text
from .action import ActionResult


def format_observations(
    observations: Sequence[ActionResult],
    policy: ContextPolicy,
) -> str:
    """Render recent observations within per-result and total character limits."""
    if not observations:
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
    omitted = len(observations) - len(rendered)
    if omitted:
        rendered.insert(0, f"[{omitted} earlier observation(s) omitted.]")
    return "\n\n".join(rendered)
