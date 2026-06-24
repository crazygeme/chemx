"""State machine for a coding-agent plan/action/observation lifecycle.

The state machine deliberately contains no model or filesystem logic. It
records durable workflow state and rejects invalid transitions so orchestration
errors fail at their source instead of producing misleading histories.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from .action import ActionResult
from .plan import CodingPlan

logger = logging.getLogger(__name__)


class CodingPhase(str, Enum):
    """Phases in a plan/action/observation coding loop."""

    INITIALIZE = "initialize"
    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class CodingRun:
    """Auditable state accumulated during one coding task.

    ``results`` is append-only. Each entry corresponds to one executed action,
    while ``step`` counts attempted actions. Terminal phases preserve either a
    response or failure reason for frontend reporting.
    """

    task: str
    phase: CodingPhase = CodingPhase.INITIALIZE
    step: int = 0
    context: str | None = None
    plan: CodingPlan | None = None
    results: list[ActionResult] = field(default_factory=list)
    response: str | None = None
    failure_reason: str | None = None


class CodingLoop:
    """Validate and record every coding workflow transition.

    Valid lifecycle:

    ``INITIALIZE -> PLAN -> ACT <-> OBSERVE -> COMPLETE|FAILED``

    The loop allows repeated ``ACT`` and ``OBSERVE`` transitions because a
    coding task normally requires several tool calls. A terminal run cannot be
    resumed.
    """

    def start(self, task: str) -> CodingRun:
        """Create a run after normalizing and validating its objective."""
        normalized = task.strip()
        if not normalized:
            raise ValueError("Coding task cannot be empty.")
        run = CodingRun(task=normalized)
        logger.info("coding run initialized task=%r", normalized)
        return run

    def begin_planning(self, run: CodingRun, context: str) -> None:
        """Attach initial workspace context and enter planning."""
        self._require_phase(run, CodingPhase.INITIALIZE)
        normalized = context.strip()
        if not normalized:
            raise ValueError("Coding context cannot be empty.")
        run.context = normalized
        run.phase = CodingPhase.PLAN
        logger.debug("coding phase transition initialize -> plan")

    def set_plan(self, run: CodingRun, plan: CodingPlan) -> None:
        """Record the selected plan and make the run ready for actions."""
        self._require_phase(run, CodingPhase.PLAN)
        run.plan = plan
        run.phase = CodingPhase.ACT
        logger.info("coding plan accepted source=%s", plan.source.value)
        logger.debug("coding phase transition plan -> act")

    def begin_action(self, run: CodingRun) -> None:
        """Reserve the next action step.

        A new action may begin immediately after plan acceptance or after the
        previous observation has been recorded.
        """
        if run.phase not in {CodingPhase.ACT, CodingPhase.OBSERVE}:
            raise RuntimeError(
                f"Cannot begin action from coding phase {run.phase.value}."
            )
        run.phase = CodingPhase.ACT
        run.step += 1
        logger.debug("coding action step started step=%d", run.step)

    def record_result(self, run: CodingRun, result: ActionResult) -> None:
        """Append one action observation and enter the observation phase."""
        self._require_phase(run, CodingPhase.ACT)
        run.results.append(result)
        run.phase = CodingPhase.OBSERVE
        logger.info(
            "coding action completed step=%d kind=%s success=%s",
            run.step,
            result.action.kind.value,
            result.success,
        )
        logger.debug("coding phase transition act -> observe")

    def complete(self, run: CodingRun, response: str) -> None:
        """Enter the successful terminal state with a non-empty response."""
        if run.phase not in {CodingPhase.ACT, CodingPhase.OBSERVE}:
            raise RuntimeError(
                f"Cannot complete coding run from phase {run.phase.value}."
            )
        normalized = response.strip()
        if not normalized:
            raise ValueError("Coding response cannot be empty.")
        run.response = normalized
        run.phase = CodingPhase.COMPLETE
        logger.info("coding run completed steps=%d", run.step)

    def fail(self, run: CodingRun, reason: str) -> None:
        """Enter the failed terminal state and retain its concrete reason."""
        if run.phase in {CodingPhase.COMPLETE, CodingPhase.FAILED}:
            raise RuntimeError(f"Cannot fail coding run from phase {run.phase.value}.")
        normalized = reason.strip()
        if not normalized:
            raise ValueError("Coding failure requires a reason.")
        run.failure_reason = normalized
        run.phase = CodingPhase.FAILED
        logger.error("coding run failed step=%d reason=%s", run.step, normalized)

    @staticmethod
    def _require_phase(run: CodingRun, expected: CodingPhase) -> None:
        """Enforce one exact transition precondition."""
        if run.phase is not expected:
            raise RuntimeError(
                f"Expected coding phase {expected.value}, got {run.phase.value}."
            )
