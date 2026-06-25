import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from chemx.backends import Message
from chemx.core import ChemicalsAgent, CodingAgent
from chemx.frontends.cli import build_parser, create_agent, main
from chemx.frontends.cli.app import _approve_command, _CodingOutput


class StaticModel:
    context_window_tokens = 32_768

    def complete(self, messages: list[Message]) -> str:
        return "response"


class CliFrontendTests(unittest.TestCase):
    def test_cli_frontend_lists_registered_backends(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--provider", "openai"])

        self.assertEqual(args.provider, "openai")

    def test_cli_accepts_deepseek_provider(self) -> None:
        args = build_parser().parse_args(["--provider", "deepseek"])

        self.assertEqual(args.provider, "deepseek")

    def test_cli_lists_specialized_agents(self) -> None:
        parser = build_parser()

        coding_args = parser.parse_args([])
        chemicals_args = parser.parse_args(["--agent", "chemicals"])

        self.assertEqual(coding_args.agent, "coding")
        self.assertEqual(chemicals_args.agent, "chemicals")

    def test_cli_counts_verbose_flags(self) -> None:
        args = build_parser().parse_args(["-vv"])

        self.assertEqual(args.verbose, 2)

    def test_cli_has_no_manual_context_flags(self) -> None:
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }

        self.assertNotIn("--context-window", option_strings)
        self.assertNotIn("--reserve-output", option_strings)
        self.assertNotIn("--recent-turns", option_strings)

    def test_cli_creates_selected_agent(self) -> None:
        model = StaticModel()

        coding_agent = create_agent("coding", model)
        chemicals_agent = create_agent("chemicals", model)

        self.assertIsInstance(coding_agent, CodingAgent)
        self.assertIsInstance(chemicals_agent, ChemicalsAgent)

    def test_custom_system_prompt_overrides_profile_default(self) -> None:
        agent = create_agent(
            "chemicals",
            StaticModel(),
            system_prompt="Custom industry prompt",
        )

        self.assertEqual(agent.system_prompt, "Custom industry prompt")

    @patch("builtins.input", side_effect=["Write notes.txt", "/exit"])
    @patch("chemx.frontends.cli.app.create_coding_session")
    @patch("chemx.frontends.cli.app._create_local_workspace")
    @patch("chemx.frontends.cli.app.prepare_backend")
    @patch("chemx.frontends.cli.app.create_backend", return_value=StaticModel())
    def test_interactive_coding_turn_uses_workspace_workflow(
        self,
        create_backend: Mock,
        prepare_backend: Mock,
        create_workspace: Mock,
        create_session: Mock,
        input_mock: Mock,
    ) -> None:
        workspace = create_workspace.return_value
        session = create_session.return_value
        session.run.return_value = "Wrote notes.txt."

        result = main(
            [
                "--provider",
                "deepseek",
                "--workspace",
                "project",
                "--max-steps",
                "7",
            ]
        )

        self.assertEqual(result, 0)
        create_backend.assert_called_once()
        prepare_backend.assert_called_once()
        create_workspace.assert_called_once()
        coding_agent = create_session.call_args.args[0]
        self.assertIsNotNone(coding_agent.progress_output)
        create_session.assert_called_once_with(
            coding_agent,
            workspace,
            max_steps=7,
        )
        session.run.assert_called_once_with("Write notes.txt")
        self.assertEqual(input_mock.call_count, 2)

    @patch("builtins.input", return_value="yes")
    def test_command_approval_accepts_yes(self, input_mock: Mock) -> None:
        approved = _approve_command(("gcc", "-o", "hello", "hello.c"))

        self.assertTrue(approved)
        input_mock.assert_called_once_with(
            "Allow command `gcc -o hello hello.c`? [y/N] "
        )

    @patch("builtins.input", return_value="")
    def test_command_approval_defaults_to_denied(self, input_mock: Mock) -> None:
        self.assertFalse(_approve_command(("make",)))
        input_mock.assert_called_once()

    def test_coding_output_uses_one_prefix_per_turn(self) -> None:
        output = _CodingOutput()
        stream = StringIO()

        with redirect_stdout(stream):
            output.begin()
            output.progress("Plan:\nCreate hello.c")
            output.progress("Step 1:\nkind: create_file")
            output.progress("Result 1 (ok):\nCreated hello.c.")
            output.finish("Task complete.")

        rendered = stream.getvalue()
        self.assertEqual(rendered.count("chemx>"), 1)
        self.assertIn("chemx> Plan:", rendered)
        self.assertIn("Step 1:", rendered)
        self.assertIn("Task complete.", rendered)

    @patch("builtins.input", side_effect=["Explain the sample", "/exit"])
    @patch("chemx.frontends.cli.app._create_local_workspace")
    @patch("chemx.frontends.cli.app.prepare_backend")
    @patch("chemx.frontends.cli.app.create_backend", return_value=StaticModel())
    def test_interactive_chemicals_turn_remains_conversational(
        self,
        create_backend: Mock,
        prepare_backend: Mock,
        create_workspace: Mock,
        input_mock: Mock,
    ) -> None:
        result = main(["--agent", "chemicals"])

        self.assertEqual(result, 0)
        create_workspace.assert_not_called()
        self.assertEqual(input_mock.call_count, 2)

    @patch("chemx.frontends.cli.app.CodingAgent.run_actions", return_value="done")
    @patch(
        "chemx.frontends.cli.app.Path.read_text",
        side_effect=["My exact plan", '[{"kind":"list_files"}]'],
    )
    @patch("chemx.frontends.cli.app._create_local_workspace")
    @patch("chemx.frontends.cli.app.create_backend")
    def test_explicit_actions_mode_does_not_create_model_backend(
        self,
        create_backend: Mock,
        create_workspace: Mock,
        read_text: Mock,
        run_actions: Mock,
    ) -> None:
        result = main(
            [
                "--agent",
                "coding",
                "--task",
                "Apply reviewed change",
                "--plan-file",
                "plan.txt",
                "--actions-file",
                "actions.json",
            ]
        )

        self.assertEqual(result, 0)
        create_backend.assert_not_called()
        create_workspace.assert_called_once()
        self.assertEqual(read_text.call_count, 2)
        run_actions.assert_called_once()

    @patch("chemx.frontends.cli.app.create_backend")
    def test_plan_file_requires_task_before_backend_setup(
        self,
        create_backend: Mock,
    ) -> None:
        result = main(["--plan-file", "plan.txt"])

        self.assertEqual(result, 2)
        create_backend.assert_not_called()

    @patch("chemx.frontends.cli.app.create_backend")
    def test_actions_file_requires_plan_file(
        self,
        create_backend: Mock,
    ) -> None:
        result = main(
            ["--task", "Edit parser", "--actions-file", "actions.json"]
        )

        self.assertEqual(result, 2)
        create_backend.assert_not_called()

    @patch("chemx.frontends.cli.app.create_backend")
    def test_chemicals_workspace_task_is_rejected_before_backend_setup(
        self,
        create_backend: Mock,
    ) -> None:
        result = main(["--agent", "chemicals", "--task", "Analyze repository"])

        self.assertEqual(result, 2)
        create_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
