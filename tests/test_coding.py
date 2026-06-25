import json
import logging
import unittest

from chemx.backends import Message
from chemx.core.coding import (
    ActionKind,
    ActionResult,
    CodingAction,
    CodingAgent,
    CodingSession,
    CodingPhase,
    PlanSource,
)


class StaticModel:
    def __init__(self, *responses: str) -> None:
        self.responses = iter(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        return next(self.responses)


class RecordingWorkspace:
    def __init__(self) -> None:
        self.actions: list[CodingAction] = []

    def inspect(self, task: str) -> str:
        return "Files:\nsrc/parser.py\ntests/test_parser.py"

    def execute(self, action: CodingAction) -> ActionResult:
        self.actions.append(action)
        outputs = {
            ActionKind.READ_FILE: "def parse_config(value): return value",
            ActionKind.REPLACE_TEXT: "Replaced exact text in src/parser.py.",
            ActionKind.RUN_COMMAND: "All tests passed.",
            ActionKind.FINISH: action.message or "Finished.",
        }
        return ActionResult(action, True, outputs.get(action.kind, "ok"))

    def changes(self) -> str:
        return "diff --git a/src/parser.py b/src/parser.py"


def action_response(kind: str, **values: object) -> str:
    return json.dumps({"kind": kind, **values})


class CodingAgentTests(unittest.TestCase):
    def test_debug_logging_describes_plan_and_actions(self) -> None:
        model = StaticModel(
            "1. Read parser.py.\n2. Finish.",
            action_response("read_file", path="src/parser.py"),
            action_response("finish", message="Review complete."),
            "Reviewed parser.py.",
        )
        agent = CodingAgent(model=model)

        with self.assertLogs(
            "chemx.core.coding.agent",
            level=logging.DEBUG,
        ) as logs:
            agent.run_workflow("Review parser", RecordingWorkspace())

        output = "\n".join(logs.output)
        self.assertIn("model-generated coding plan", output)
        self.assertIn("1. Read parser.py.", output)
        self.assertIn("model-selected coding step=1", output)
        self.assertIn('"kind": "read_file"', output)
        self.assertIn("workspace action to execute step=1", output)
        self.assertIn("path: src/parser.py", output)

    def test_action_prompt_explains_direct_command_approval(self) -> None:
        model = StaticModel(
            "Inspect files.",
            action_response("finish", message="No changes needed."),
            "No changes were required.",
        )
        agent = CodingAgent(model=model)

        agent.run_workflow("Review parser", RecordingWorkspace())

        self.assertIn(
            "commands run directly without Bash and require user approval",
            model.calls[1][1].content,
        )

    def test_coding_session_runs_workspace_backed_interactive_turn(self) -> None:
        model = StaticModel(
            "Inspect files.",
            action_response("finish", message="No changes needed."),
            "No changes were required.",
        )
        workspace = RecordingWorkspace()
        session = CodingSession(
            agent=CodingAgent(model=model),
            workspace=workspace,
            max_steps=7,
        )

        response = session.run("Review parser")

        self.assertEqual(response, "No changes were required.")
        self.assertEqual(workspace.actions[0].kind, ActionKind.FINISH)

    def test_model_plan_is_natural_language_then_actions_are_structured(self) -> None:
        model = StaticModel(
            "1. Read parser.py.\n2. Add validation.\n3. Run parser tests.",
            action_response("read_file", path="src/parser.py"),
            action_response(
                "replace_text",
                path="src/parser.py",
                old_text="return value",
                new_text="return value or raise_error()",
            ),
            action_response(
                "run_command",
                command=["python3", "-m", "unittest"],
            ),
            action_response("finish", message="Implementation is verified."),
            "Added validation and verified the tests.",
        )
        workspace = RecordingWorkspace()
        agent = CodingAgent(model=model)

        response = agent.run_workflow(
            "Reject empty configuration values",
            workspace,
        )

        self.assertEqual(response, "Added validation and verified the tests.")
        self.assertEqual(agent.current_run.plan.source, PlanSource.MODEL)
        self.assertEqual(
            [action.kind for action in workspace.actions],
            [
                ActionKind.READ_FILE,
                ActionKind.REPLACE_TEXT,
                ActionKind.RUN_COMMAND,
                ActionKind.FINISH,
            ],
        )
        self.assertEqual(agent.current_run.phase, CodingPhase.COMPLETE)
        self.assertIn("Do not write code yet", model.calls[0][1].content)
        self.assertIn("Choose exactly one next coding action", model.calls[1][1].content)

    def test_user_plan_is_preserved_but_model_selects_actions(self) -> None:
        model = StaticModel(
            action_response("finish", message="No changes needed."),
            "No changes were required.",
        )
        workspace = RecordingWorkspace()
        agent = CodingAgent(model=model)
        plan = "Inspect parser behavior and make no change unless it is incorrect."

        agent.run_plan("Review parser", plan, workspace)

        self.assertEqual(agent.current_run.plan.text, plan)
        self.assertEqual(agent.current_run.plan.source, PlanSource.USER)
        self.assertEqual(len(model.calls), 2)

    def test_explicit_actions_run_without_model(self) -> None:
        model = StaticModel()
        workspace = RecordingWorkspace()
        agent = CodingAgent(model=model)
        actions = [
            CodingAction(ActionKind.READ_FILE, path="src/parser.py"),
            CodingAction(
                ActionKind.RUN_COMMAND,
                command=("python3", "-m", "unittest"),
            ),
        ]

        response = agent.run_actions(
            "Review parser",
            "Read the parser and run tests.",
            actions,
            workspace,
        )

        self.assertEqual(model.calls, [])
        self.assertEqual(workspace.actions, actions)
        self.assertIn("Executed 2 user-authored action(s)", response)

    def test_failed_explicit_action_stops_without_model(self) -> None:
        class FailingWorkspace(RecordingWorkspace):
            def execute(self, action: CodingAction) -> ActionResult:
                self.actions.append(action)
                return ActionResult(action, False, "Exact text was not found.")

        model = StaticModel()
        workspace = FailingWorkspace()
        agent = CodingAgent(model=model)

        response = agent.run_actions(
            "Edit parser",
            "Replace one exact expression.",
            [
                CodingAction(
                    ActionKind.REPLACE_TEXT,
                    path="src/parser.py",
                    old_text="old",
                    new_text="new",
                )
            ],
            workspace,
        )

        self.assertEqual(model.calls, [])
        self.assertEqual(agent.current_run.phase, CodingPhase.FAILED)
        self.assertIn("failed at step 1", response)

    def test_workflow_stops_at_step_limit(self) -> None:
        model = StaticModel(
            "Inspect files.",
            action_response("list_files"),
            action_response("list_files"),
        )
        agent = CodingAgent(model=model)

        response = agent.run_workflow(
            "Inspect repository",
            RecordingWorkspace(),
            max_steps=2,
        )

        self.assertEqual(response, "Coding workflow reached the 2-step limit.")
        self.assertEqual(agent.current_run.phase, CodingPhase.FAILED)


if __name__ == "__main__":
    unittest.main()
