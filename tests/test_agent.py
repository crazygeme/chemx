import unittest

from chemx.backends import Message
from chemx.core import Agent


class RecordingModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        return next(self.responses)


class AgentTests(unittest.TestCase):
    def test_run_records_a_conversation(self) -> None:
        model = RecordingModel(["Hello.", "Still here."])
        agent = Agent(model=model, system_prompt="Be concise.")

        self.assertEqual(agent.run("Hi"), "Hello.")
        self.assertEqual(agent.run("Are you there?"), "Still here.")

        second_call = model.calls[1]
        self.assertEqual(second_call[0], Message("system", "Be concise."))
        self.assertEqual(second_call[1], Message("user", "Hi"))
        self.assertEqual(second_call[2], Message("assistant", "Hello."))
        self.assertEqual(second_call[3], Message("user", "Are you there?"))

    def test_clear_removes_history(self) -> None:
        model = RecordingModel(["Hello."])
        agent = Agent(model=model)
        agent.run("Hi")

        agent.clear()

        self.assertEqual(agent.history, [])

    def test_empty_input_is_rejected(self) -> None:
        agent = Agent(model=RecordingModel([]))

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            agent.run("  ")


if __name__ == "__main__":
    unittest.main()
