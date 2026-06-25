import json
import unittest

from chemx.backends import Message
from chemx.core.coding import (
    ActionKind,
    ActionResult,
    CodingAction,
    CodingAgent,
    CodingSession,
    WorkflowKind,
    WorkflowRouter,
)


class StaticModel:
    context_window_tokens = 32_768

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
        return "Files:\nsrc/parser.py"

    def execute(self, action: CodingAction) -> ActionResult:
        self.actions.append(action)
        outputs = {
            ActionKind.READ_FILE: "def parse(value): return value",
            ActionKind.CREATE_FILE: "Created docs/parser.md.",
            ActionKind.FINISH: action.message or "Finished.",
        }
        return ActionResult(action, True, outputs.get(action.kind, "ok"))

    def changes(self) -> str:
        return "created: docs/parser.md"


def action_response(kind: str, **values: object) -> str:
    return json.dumps({"kind": kind, **values})


class WorkflowRouterTests(unittest.TestCase):
    def test_document_request_uses_classifier_model(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "document",
                    "objective": "Generate a README guide for installation",
                    "confidence": 0.95,
                }
            )
        )
        route = WorkflowRouter(model).classify(
            "Generate a README guide for installation"
        )

        self.assertEqual(route.kind, WorkflowKind.DOCUMENT)
        self.assertEqual(len(model.calls), 1)

    def test_coding_request_uses_classifier_model(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "coding",
                    "objective": "Fix the parser tests",
                    "confidence": 0.95,
                }
            )
        )
        route = WorkflowRouter(model).classify("Fix the parser tests")

        self.assertEqual(route.kind, WorkflowKind.CODING)
        self.assertEqual(len(model.calls), 1)

    def test_ambiguous_request_uses_structured_model_route(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "conversation",
                    "objective": "Explain the architecture",
                    "confidence": 0.8,
                }
            )
        )

        route = WorkflowRouter(model).classify("Explain the architecture")

        self.assertEqual(route.kind, WorkflowKind.CONVERSATION)
        self.assertEqual(route.objective, "Explain the architecture")
        self.assertIn("Return one JSON object only", model.calls[0][-1].content)

    def test_invalid_model_route_is_rejected(self) -> None:
        model = StaticModel('{"kind":"unknown"}')

        with self.assertRaisesRegex(ValueError, "invalid workflow route"):
            WorkflowRouter(model).classify("Please handle this")


class RoutedSessionTests(unittest.TestCase):
    def test_document_request_uses_document_prompts_and_workspace_loop(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "document",
                    "objective": "Generate a document explaining the parser",
                    "confidence": 0.96,
                }
            ),
            "1. Inspect parser source.\n2. Draft the guide.\n3. Review it.",
            action_response("read_file", path="src/parser.py"),
            action_response(
                "create_file",
                path="docs/parser.md",
                content="# Parser\n\nParser usage.",
            ),
            action_response("finish", message="Document reviewed."),
            "Created and reviewed docs/parser.md.",
        )
        workspace = RecordingWorkspace()
        session = CodingSession(
            agent=CodingAgent(model=model),
            workspace=workspace,
        )

        response = session.run("Generate a document explaining the parser")

        self.assertEqual(response, "Created and reviewed docs/parser.md.")
        self.assertEqual(
            [action.kind for action in workspace.actions],
            [
                ActionKind.READ_FILE,
                ActionKind.CREATE_FILE,
                ActionKind.FINISH,
            ],
        )
        self.assertIn("requested workflow", model.calls[0][-1].content)
        self.assertIn("document-production plan", model.calls[1][-1].content)
        self.assertIn(
            "next document-production action",
            model.calls[2][-1].content,
        )

    def test_model_routed_conversation_does_not_touch_workspace(self) -> None:
        model = StaticModel(
            json.dumps(
                {
                    "kind": "conversation",
                    "objective": "Explain the architecture",
                    "confidence": 0.9,
                }
            ),
            "The architecture has a model-independent core.",
        )
        workspace = RecordingWorkspace()
        session = CodingSession(
            agent=CodingAgent(model=model),
            workspace=workspace,
        )

        response = session.run("Explain the architecture")

        self.assertEqual(
            response,
            "The architecture has a model-independent core.",
        )
        self.assertEqual(workspace.actions, [])


if __name__ == "__main__":
    unittest.main()
