"""Intent routing for interactive workspace sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from ...backends import Message, ModelBackend
from ..context import ContextPolicy, fit_messages


class WorkflowKind(str, Enum):
    """Interactive workflows available to a routed session."""

    CONVERSATION = "conversation"
    CODING = "coding"
    DOCUMENT = "document"


@dataclass(frozen=True)
class WorkflowRoute:
    """Validated workflow selection for one user input."""

    kind: WorkflowKind
    objective: str
    confidence: float

    def __post_init__(self) -> None:
        objective = self.objective.strip()
        if not objective:
            raise ValueError("Workflow route objective cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Workflow route confidence must be between 0 and 1.")
        object.__setattr__(self, "objective", objective)


@dataclass
class WorkflowRouter:
    """Use the model backend to select a workflow for every request."""

    model: ModelBackend
    context_policy: ContextPolicy | None = None

    def __post_init__(self) -> None:
        if self.context_policy is None:
            self.context_policy = ContextPolicy.from_backend(self.model)

    def classify(self, user_input: str) -> WorkflowRoute:
        """Return a validated route without executing the selected workflow."""
        objective = user_input.strip()
        if not objective:
            raise ValueError("User input cannot be empty.")
        return self._classify_with_model(objective)

    def _classify_with_model(self, objective: str) -> WorkflowRoute:
        assert self.context_policy is not None
        prompt = (
            "Classify the user's requested workflow. Return one JSON object "
            "only, with kind, objective, and confidence.\n"
            "Kinds:\n"
            '- "coding": inspect, change, debug, or test software\n'
            '- "document": research, draft, or revise a document\n'
            '- "conversation": answer or discuss without workspace changes\n'
            "Choose the workflow based on the requested output and whether "
            "workspace actions are required. A request to explain code without "
            "changing files is conversation; a request to create documentation "
            "is document even when code is its source. Greetings, thanks, "
            "acknowledgements, praise, and other social replies are conversation.\n"
            f"User input:\n{objective}\n\n"
            'Example: {"kind":"coding","objective":"Fix the parser",'
            '"confidence":0.9}'
        )
        messages = fit_messages(
            system_prompt="You route requests to bounded agent workflows.",
            history=(),
            current=Message(role="user", content=prompt),
            policy=self.context_policy,
        )
        response = self.model.complete(messages).strip()
        try:
            value = json.loads(response)
            if not isinstance(value, dict):
                raise ValueError
            kind = WorkflowKind(value["kind"])
            routed_objective = value.get("objective", objective)
            confidence = float(value.get("confidence", 0.5))
            if not isinstance(routed_objective, str):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Model returned an invalid workflow route.") from error
        return WorkflowRoute(kind, routed_objective, confidence)
