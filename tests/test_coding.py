import json
import logging
import unittest

from chemx.backends import Message, ToolCall
from chemx.core.coding import (
    ActionKind,
    ActionResult,
    CodingAction,
    CodingAgent,
    CodingSession,
    CodingPhase,
    ContextPolicy,
    PlanSource,
)


class StaticModel:
    context_window_tokens = 32_768

    def __init__(self, *responses: str) -> None:
        self.responses = iter(responses)
        self.calls: list[list[Message]] = []
        self.response_schemas: list[object] = []

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: object = None,
    ) -> str:
        self.calls.append(list(messages))
        self.response_schemas.append(response_schema)
        return next(self.responses)


class NativeToolModel(StaticModel):
    def __init__(
        self,
        responses: tuple[str, ...],
        tool_calls: tuple[ToolCall, ...],
    ) -> None:
        super().__init__(*responses)
        self.tool_calls = iter(tool_calls)
        self.tool_requests: list[tuple[list[Message], object]] = []

    def complete_tool(self, messages: list[Message], tools: object) -> ToolCall:
        self.tool_requests.append((list(messages), tools))
        return next(self.tool_calls)


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
    def test_progress_output_reports_plan_steps_commands_and_results(self) -> None:
        messages: list[str] = []
        model = StaticModel(
            "1. Read parser.py.\n2. Run tests.",
            action_response("read_file", path="src/parser.py"),
            action_response(
                "run_command",
                command=["python3", "-m", "unittest"],
            ),
            action_response("finish", message="Complete."),
            "Reviewed and tested parser.py.",
        )
        agent = CodingAgent(model=model, progress_output=messages.append)

        agent.run_workflow("Review parser", RecordingWorkspace())

        output = "\n".join(messages)
        self.assertIn("Plan:\n1. Read parser.py.", output)
        self.assertIn("Step 1: kind: read_file", output)
        self.assertIn("path: src/parser.py", output)
        self.assertIn("Result 1 (ok):", output)
        self.assertIn("Step 2: kind: run_command", output)
        self.assertIn(
            "command: ['python3', '-m', 'unittest']",
            output,
        )
        self.assertIn("Result 2 (ok): run_command", output)
        self.assertNotIn("All tests passed.", output)

    def test_progress_output_redacts_action_and_result_contents(self) -> None:
        messages: list[str] = []
        model = StaticModel(
            "Create a private document.",
            action_response(
                "create_file",
                path="notes.txt",
                content="private document contents",
            ),
            action_response("finish", message="Private completion message."),
            "Created notes.txt.",
        )
        agent = CodingAgent(model=model, progress_output=messages.append)

        agent.run_workflow("Create notes", RecordingWorkspace())

        output = "\n".join(messages)
        self.assertIn("content: [redacted, 25 chars]", output)
        self.assertIn("message: [redacted, 27 chars]", output)
        self.assertNotIn("private document contents", output)
        self.assertNotIn("Private completion message.", output)

    def test_debug_logging_keeps_plan_and_actions_in_progress_output(self) -> None:
        progress: list[str] = []
        model = StaticModel(
            "1. Read parser.py.\n2. Finish.",
            action_response("read_file", path="src/parser.py"),
            action_response("finish", message="Review complete."),
            "Reviewed parser.py.",
        )
        agent = CodingAgent(model=model, progress_output=progress.append)

        with self.assertLogs(
            "chemx.core.coding.agent",
            level=logging.DEBUG,
        ) as logs:
            agent.run_workflow("Review parser", RecordingWorkspace())

        output = "\n".join(logs.output)
        self.assertIn("model-generated coding plan", output)
        self.assertNotIn("1. Read parser.py.", output)
        self.assertNotIn('"kind": "read_file"', output)
        self.assertIn("1. Read parser.py.", "\n".join(progress))
        self.assertIn("path: src/parser.py", "\n".join(progress))

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
        self.assertIn(
            "Scope list_files to a directory identified by the plan",
            model.calls[1][1].content,
        )
        self.assertIsNone(model.response_schemas[0])
        self.assertEqual(
            model.response_schemas[1]["properties"]["kind"]["type"],
            "string",
        )
        self.assertIsNone(model.response_schemas[2])

    def test_invalid_model_action_is_repaired_before_execution(self) -> None:
        progress: list[str] = []
        model = StaticModel(
            "Inspect parser.",
            action_response("scan_directory", path="src"),
            action_response("read_file", path="src/parser.py"),
            action_response("finish", message="Complete."),
            "Reviewed parser.py.",
        )
        workspace = RecordingWorkspace()
        agent = CodingAgent(model=model, progress_output=progress.append)

        response = agent.run_workflow("Review parser", workspace)

        self.assertEqual(response, "Reviewed parser.py.")
        self.assertEqual(
            [action.kind for action in workspace.actions],
            [ActionKind.READ_FILE, ActionKind.FINISH],
        )
        self.assertIn("previous response was invalid", model.calls[2][1].content)
        self.assertIn("requesting a corrected tool call", "\n".join(progress))

    def test_native_tool_calls_drive_workspace_actions(self) -> None:
        model = NativeToolModel(
            responses=(
                "Inspect parser and finish.",
                "Reviewed parser.py.",
            ),
            tool_calls=(
                ToolCall("read_file", {"path": "src/parser.py"}),
                ToolCall("finish", {"message": "Complete."}),
            ),
        )
        workspace = RecordingWorkspace()
        agent = CodingAgent(model=model)

        response = agent.run_workflow("Review parser", workspace)

        self.assertEqual(response, "Reviewed parser.py.")
        self.assertEqual(
            [action.kind for action in workspace.actions],
            [ActionKind.READ_FILE, ActionKind.FINISH],
        )
        self.assertEqual(len(model.tool_requests), 2)
        tool_names = [tool.name for tool in model.tool_requests[0][1]]
        self.assertIn("read_file", tool_names)
        self.assertIn("finish", tool_names)

    def test_invalid_native_tool_call_is_repaired(self) -> None:
        progress: list[str] = []
        model = NativeToolModel(
            responses=(
                "Inspect parser and finish.",
                "Reviewed parser.py.",
            ),
            tool_calls=(
                ToolCall("inspect_file", {"path": "src/parser.py"}),
                ToolCall("read_file", {"path": "src/parser.py"}),
                ToolCall("finish", {"message": "Complete."}),
            ),
        )
        workspace = RecordingWorkspace()
        agent = CodingAgent(model=model, progress_output=progress.append)

        agent.run_workflow("Review parser", workspace)

        self.assertEqual(
            [action.kind for action in workspace.actions],
            [ActionKind.READ_FILE, ActionKind.FINISH],
        )
        self.assertIn(
            "requesting a corrected tool call",
            "\n".join(progress),
        )

    def test_two_invalid_model_actions_fail_without_workspace_execution(self) -> None:
        model = StaticModel(
            "Inspect parser.",
            action_response("scan_directory", path="src"),
            action_response("inspect_directory", path="src"),
        )
        workspace = RecordingWorkspace()
        agent = CodingAgent(model=model)

        with self.assertRaisesRegex(ValueError, "invalid action twice"):
            agent.run_workflow("Review parser", workspace)

        self.assertEqual(workspace.actions, [])

    def test_coding_session_runs_workspace_backed_interactive_turn(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "coding",
                    "objective": "Review parser",
                    "confidence": 0.95,
                }
            ),
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

    def test_coding_session_treats_good_job_as_conversation(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "conversation",
                    "objective": "good job",
                    "confidence": 0.99,
                }
            ),
            "Thanks!",
        )
        workspace = RecordingWorkspace()
        session = CodingSession(
            agent=CodingAgent(model=model),
            workspace=workspace,
        )

        response = session.run("good job")

        self.assertEqual(response, "Thanks!")
        self.assertEqual(workspace.actions, [])
        self.assertEqual(len(model.calls), 2)
        self.assertIn("requested workflow", model.calls[0][-1].content)
        self.assertIn(
            "social replies are conversation",
            model.calls[0][-1].content,
        )
        self.assertEqual(model.calls[1][-1].content, "good job")

    def test_coding_session_does_not_misclassify_task_with_praise(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "coding",
                    "objective": "Good job, now update the parser",
                    "confidence": 0.95,
                }
            ),
            "Update the parser.",
            action_response("finish", message="Complete."),
            "Parser update complete.",
        )
        workspace = RecordingWorkspace()
        session = CodingSession(
            agent=CodingAgent(model=model),
            workspace=workspace,
        )

        response = session.run("Good job, now update the parser")

        self.assertEqual(response, "Parser update complete.")
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

    def test_action_prompt_compacts_older_and_oversized_observations(self) -> None:
        class LargeWorkspace(RecordingWorkspace):
            def execute(self, action: CodingAction) -> ActionResult:
                self.actions.append(action)
                return ActionResult(action, True, "begin-" + ("x" * 200) + "-end")

        model = StaticModel(
            "Inspect files.",
            action_response("list_files"),
            action_response("list_files"),
            "Earlier list_files succeeded and returned repository output.",
            action_response("list_files"),
        )
        agent = CodingAgent(
            model=model,
            context_policy=ContextPolicy(
                max_observations=1,
                max_observation_chars=60,
                max_observation_chars_total=100,
            ),
        )

        agent.run_workflow("Inspect repository", LargeWorkspace(), max_steps=3)

        compaction_prompt = model.calls[3][1].content
        self.assertIn("Compress coding tool observations", compaction_prompt)
        self.assertIn("list_files (ok)", compaction_prompt)

        third_action_prompt = model.calls[4][1].content
        self.assertIn(
            "Earlier list_files succeeded",
            third_action_prompt,
        )
        self.assertIn("begin-", third_action_prompt)
        self.assertIn("-end", third_action_prompt)
        self.assertIn("content truncated", third_action_prompt)


if __name__ == "__main__":
    unittest.main()
