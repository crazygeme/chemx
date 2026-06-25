import unittest

from chemx.backends import Message
from chemx.core.chemicals import ChemicalsAgent, ChemicalsLoop, ChemicalsPhase


class StaticModel:
    context_window_tokens = 32_768

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: object = None,
    ) -> str:
        self.calls.append(list(messages))
        return self.response


class ChemicalsLoopTests(unittest.TestCase):
    def test_agent_completes_initial_safety_assessment(self) -> None:
        model = StaticModel("First identify the process conditions and hazards.")
        agent = ChemicalsAgent(model=model)

        response = agent.run("Evaluate a solvent substitution")

        self.assertEqual(
            response,
            "First identify the process conditions and hazards.",
        )
        self.assertIsNotNone(agent.current_run)
        self.assertEqual(agent.current_run.phase, ChemicalsPhase.COMPLETE)
        self.assertEqual(agent.current_run.iteration, 1)
        self.assertIn("chemical-industry work", model.calls[0][0].content)

    def test_full_workflow_reaches_reporting(self) -> None:
        loop = ChemicalsLoop()
        run = loop.start("Compare two separation processes")

        loop.begin_safety_review(run)
        loop.approve_safety_review(run, ["flammable solvent"])
        loop.begin_analysis(run, ["mass balance", "energy estimate"])
        loop.begin_validation(run, "Balances close within tolerance")
        loop.begin_reporting(run, "Independent calculation agrees")
        loop.complete(run, "Process comparison report")

        self.assertEqual(run.phase, ChemicalsPhase.COMPLETE)
        self.assertEqual(run.hazards, ["flammable solvent"])
        self.assertEqual(run.response, "Process comparison report")

    def test_failed_validation_returns_to_analysis(self) -> None:
        loop = ChemicalsLoop()
        run = loop.start("Review reactor yield")

        loop.begin_safety_review(run)
        loop.approve_safety_review(run)
        loop.begin_analysis(run, ["batch records"])
        loop.begin_validation(run, "Yield calculation completed")
        loop.request_revision(run, "Feed concentration is uncertain")

        self.assertEqual(run.phase, ChemicalsPhase.ANALYZE)
        self.assertEqual(run.iteration, 2)
        self.assertIn("Feed concentration is uncertain", run.observations)

    def test_workflow_can_be_blocked_for_safety(self) -> None:
        loop = ChemicalsLoop()
        run = loop.start("Change reactor operating temperature")

        loop.begin_safety_review(run)
        loop.block(run, "Relief-system basis is unavailable")

        self.assertEqual(run.phase, ChemicalsPhase.BLOCKED)
        self.assertEqual(
            run.blocked_reason,
            "Relief-system basis is unavailable",
        )


if __name__ == "__main__":
    unittest.main()
