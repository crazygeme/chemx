import unittest

from chemx.backends import Message
from chemx.core import (
    Agent,
    ContextPolicy,
    calculate_reserved_output_tokens,
    truncate_text,
)


class RecordingModel:
    context_window_tokens = 32_768

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: object = None,
    ) -> str:
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

    def test_context_policy_is_derived_from_backend(self) -> None:
        model = RecordingModel(["Hello."])
        model.context_window_tokens = 64_000

        agent = Agent(model=model)

        self.assertEqual(agent.context_policy.context_window_tokens, 64_000)
        self.assertEqual(agent.context_policy.reserved_output_tokens, 8_000)

    def test_backend_without_context_metadata_is_rejected(self) -> None:
        class IncompleteBackend:
            def complete(
                self,
                messages: list[Message],
                *,
                response_schema: object = None,
            ) -> str:
                return "response"

        with self.assertRaisesRegex(ValueError, "context_window_tokens"):
            Agent(model=IncompleteBackend())

    def test_output_reserve_is_bounded(self) -> None:
        self.assertEqual(calculate_reserved_output_tokens(4_096), 512)
        self.assertEqual(calculate_reserved_output_tokens(64_000), 8_000)
        self.assertEqual(calculate_reserved_output_tokens(1_000_000), 16_384)

    def test_clear_removes_history(self) -> None:
        model = RecordingModel(["Hello."])
        agent = Agent(model=model)
        agent.run("Hi")
        agent.context_summary = "Earlier context."

        agent.clear()

        self.assertEqual(agent.history, [])
        self.assertIsNone(agent.context_summary)

    def test_empty_input_is_rejected(self) -> None:
        agent = Agent(model=RecordingModel([]))

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            agent.run("  ")

    def test_context_policy_retains_recent_complete_turns(self) -> None:
        model = RecordingModel(["one", "two", "Earlier: first/one.", "three"])
        agent = Agent(
            model=model,
            system_prompt="system",
            context_policy=ContextPolicy(
                context_window_tokens=1_000,
                reserved_output_tokens=100,
                recent_turns=1,
            ),
        )

        agent.run("first")
        agent.run("second")
        agent.run("third")

        compaction_call = model.calls[2]
        self.assertIn("Compress prior conversation context", compaction_call[0].content)
        self.assertIn("user: first", compaction_call[1].content)

        third_call = model.calls[3]
        self.assertIn("Earlier: first/one.", third_call[0].content)
        self.assertEqual(third_call[1], Message("user", "second"))
        self.assertEqual(third_call[2], Message("assistant", "two"))
        self.assertEqual(third_call[3], Message("user", "third"))
        self.assertEqual(agent.context_summary, "Earlier: first/one.")
        self.assertEqual(agent.history[-2:], [
            Message("user", "third"),
            Message("assistant", "three"),
        ])

    def test_oversized_current_input_is_truncated_to_budget(self) -> None:
        model = RecordingModel(["ok"])
        policy = ContextPolicy(
            context_window_tokens=30,
            reserved_output_tokens=5,
        )
        agent = Agent(model=model, system_prompt="brief", context_policy=policy)

        agent.run("x" * 200)

        total_chars = sum(len(message.content) for message in model.calls[0])
        self.assertLessEqual(total_chars, policy.input_character_budget)
        self.assertIn("content truncated", model.calls[0][-1].content)

    def test_zero_recent_turns_omits_all_history(self) -> None:
        model = RecordingModel(["one", "First exchange summarized.", "two"])
        agent = Agent(
            model=model,
            context_policy=ContextPolicy(recent_turns=0),
        )

        agent.run("first")
        agent.run("second")

        second_call = model.calls[2]
        self.assertIn("First exchange summarized.", second_call[0].content)
        self.assertEqual(second_call[-1], Message("user", "second"))
        self.assertNotIn(Message("user", "first"), second_call)

    def test_truncate_text_preserves_both_ends(self) -> None:
        truncated = truncate_text("start-" + ("x" * 100) + "-end", 50)

        self.assertEqual(len(truncated), 50)
        self.assertTrue(truncated.startswith("start-"))
        self.assertTrue(truncated.endswith("-end"))


if __name__ == "__main__":
    unittest.main()
