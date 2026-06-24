"""Chemical-industry agent with an explicit safety-review boundary."""

import logging
from dataclasses import dataclass, field

from ...backends import ModelBackend
from ..agent import Agent
from .loop import ChemicalsLoop, ChemicalsRun

logger = logging.getLogger(__name__)


CHEMICALS_SYSTEM_PROMPT = """You are an agent for chemical-industry work.
Begin by clarifying the objective, process context, available data, constraints,
and expected deliverable. Treat process safety, worker safety, environmental
impact, material compatibility, data provenance, uncertainty, and regulatory
requirements as explicit concerns. Separate verified facts from assumptions.
Do not claim that laboratory work, plant inspection, simulation, validation,
or regulatory review occurred unless confirmed by evidence or tool results.
Escalate decisions that require qualified engineering, EHS, laboratory, legal,
or regulatory approval."""


@dataclass
class ChemicalsAgent(Agent):
    """Run safety-aware assessments while retaining workflow state.

    The current model-backed entry point performs an initial assessment only.
    It enters ``SAFETY_REVIEW`` and records the response as an assessment; it
    does not advance through data gathering or validation without explicit
    evidence supplied through the workflow API.
    """

    model: ModelBackend
    system_prompt: str = CHEMICALS_SYSTEM_PROMPT
    loop: ChemicalsLoop = field(default_factory=ChemicalsLoop)
    current_run: ChemicalsRun | None = field(default=None, init=False)

    def run(self, user_input: str) -> str:
        """Run an initial safety-aware assessment of an industry task."""
        logger.info("chemical-industry assessment started")
        chemicals_run = self.loop.start(user_input)
        self.current_run = chemicals_run
        self.loop.begin_safety_review(chemicals_run)

        response = super().run(chemicals_run.task)
        self.loop.complete(chemicals_run, response)
        logger.info("chemical-industry assessment completed")
        return response
