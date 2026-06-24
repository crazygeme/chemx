import unittest
from unittest.mock import Mock, patch

from chemx.backends import Message
from chemx.core import ChemicalsAgent, CodingAgent
from chemx.frontends.cli import build_parser, create_agent, main


class StaticModel:
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
