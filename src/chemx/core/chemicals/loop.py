"""Safety-gated state machine for chemical-industry workflows.

The workflow separates safety approval, evidence gathering, analysis,
validation, and reporting. Transitions require explicit evidence or
observations so a report cannot accidentally imply that an unperformed review
or validation occurred.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ChemicalsPhase(str, Enum):
    """Phases available to a chemical-industry workflow."""

    INTAKE = "intake"
    SAFETY_REVIEW = "safety_review"
    GATHER_DATA = "gather_data"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    REPORT = "report"
    COMPLETE = "complete"
    BLOCKED = "blocked"


@dataclass
class ChemicalsRun:
    """Auditable state for one chemical-industry task."""

    task: str
    phase: ChemicalsPhase = ChemicalsPhase.INTAKE
    iteration: int = 0
    hazards: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    response: str | None = None
    blocked_reason: str | None = None


class ChemicalsLoop:
    """Control the formal lifecycle of a safety-sensitive task.

    The normal lifecycle is:

    ``INTAKE -> SAFETY_REVIEW -> GATHER_DATA -> ANALYZE -> VALIDATE
    -> REPORT -> COMPLETE``.

    Validation may return to analysis, and any non-terminal phase may enter
    ``BLOCKED`` when safety or evidence requirements cannot be satisfied.
    """

    def start(self, task: str) -> ChemicalsRun:
        """Initialize a workflow and validate its objective."""
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("Chemical-industry task cannot be empty.")
        run = ChemicalsRun(task=normalized_task)
        logger.info("chemicals run initialized task=%r", normalized_task)
        return run

    def begin_safety_review(self, run: ChemicalsRun) -> None:
        """Start hazard and operating-boundary assessment."""
        self._require_phase(run, ChemicalsPhase.INTAKE)
        run.phase = ChemicalsPhase.SAFETY_REVIEW
        run.iteration += 1
        logger.info("chemicals phase intake -> safety_review")

    def approve_safety_review(
        self,
        run: ChemicalsRun,
        hazards: list[str] | None = None,
    ) -> None:
        """Record known hazards and allow evidence collection."""
        self._require_phase(run, ChemicalsPhase.SAFETY_REVIEW)
        run.hazards.extend(self._normalize_items(hazards or []))
        run.phase = ChemicalsPhase.GATHER_DATA
        logger.info(
            "chemicals safety review approved hazards=%d",
            len(run.hazards),
        )

    def begin_analysis(
        self,
        run: ChemicalsRun,
        evidence: list[str],
    ) -> None:
        """Record source data and begin technical analysis."""
        self._require_phase(run, ChemicalsPhase.GATHER_DATA)
        normalized_evidence = self._normalize_items(evidence)
        if not normalized_evidence:
            raise ValueError("Analysis requires at least one evidence item.")
        run.evidence.extend(normalized_evidence)
        run.phase = ChemicalsPhase.ANALYZE
        logger.info("chemicals phase gather_data -> analyze evidence=%d", len(run.evidence))

    def begin_validation(self, run: ChemicalsRun, observation: str) -> None:
        """Record an analytical result and begin validation."""
        self._require_phase(run, ChemicalsPhase.ANALYZE)
        normalized_observation = observation.strip()
        if not normalized_observation:
            raise ValueError("Validation requires an analysis observation.")
        run.observations.append(normalized_observation)
        run.phase = ChemicalsPhase.VALIDATE
        logger.info("chemicals phase analyze -> validate")

    def request_revision(self, run: ChemicalsRun, issue: str) -> None:
        """Record a validation issue and return to analysis."""
        self._require_phase(run, ChemicalsPhase.VALIDATE)
        normalized_issue = issue.strip()
        if not normalized_issue:
            raise ValueError("Revision requires a validation issue.")
        run.observations.append(normalized_issue)
        run.phase = ChemicalsPhase.ANALYZE
        run.iteration += 1
        logger.info("chemicals validation requested revision issue=%s", normalized_issue)

    def begin_reporting(self, run: ChemicalsRun, validation: str) -> None:
        """Record successful validation and begin reporting."""
        self._require_phase(run, ChemicalsPhase.VALIDATE)
        normalized_validation = validation.strip()
        if not normalized_validation:
            raise ValueError("Reporting requires a validation result.")
        run.observations.append(normalized_validation)
        run.phase = ChemicalsPhase.REPORT
        logger.info("chemicals phase validate -> report")

    def complete(self, run: ChemicalsRun, response: str) -> None:
        """Complete a report or an initial safety assessment."""
        if run.phase not in {ChemicalsPhase.SAFETY_REVIEW, ChemicalsPhase.REPORT}:
            raise RuntimeError(
                f"Cannot complete a chemicals workflow from phase {run.phase.value}."
            )
        normalized_response = response.strip()
        if not normalized_response:
            raise ValueError("Chemical-industry response cannot be empty.")
        run.response = normalized_response
        run.phase = ChemicalsPhase.COMPLETE
        logger.info("chemicals run completed iterations=%d", run.iteration)

    def block(self, run: ChemicalsRun, reason: str) -> None:
        """Stop a workflow when safety or evidence requirements are unmet."""
        if run.phase in {ChemicalsPhase.COMPLETE, ChemicalsPhase.BLOCKED}:
            raise RuntimeError(
                f"Cannot block a chemicals workflow from phase {run.phase.value}."
            )
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("A blocked workflow requires a reason.")
        run.blocked_reason = normalized_reason
        run.phase = ChemicalsPhase.BLOCKED
        logger.error("chemicals run blocked reason=%s", normalized_reason)

    @staticmethod
    def _normalize_items(items: list[str]) -> list[str]:
        return [item.strip() for item in items if item.strip()]

    @staticmethod
    def _require_phase(run: ChemicalsRun, expected: ChemicalsPhase) -> None:
        if run.phase is not expected:
            raise RuntimeError(
                f"Expected chemicals phase {expected.value}, got {run.phase.value}."
            )
