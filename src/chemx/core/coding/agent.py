"""Coding-specialized orchestration over plans, actions, and observations.

The agent owns decision-making but does not directly access the filesystem or
process environment. All effects pass through ``CodingWorkspace``, which keeps
execution policy replaceable and makes action results available for auditing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from ...backends import Message, ModelBackend
from ..agent import Agent
from .action import ActionKind, ActionResult, CodingAction, parse_action
from .loop import CodingLoop, CodingRun
from .plan import (
    CodingPlan,
    build_action_prompt,
    build_plan_prompt,
    build_summary_prompt,
)
from .workspace import CodingWorkspace

logger = logging.getLogger(__name__)


CODING_SYSTEM_PROMPT = """You are a software engineering agent.
Inspect relevant code before editing. Keep plans separate from tool actions.
Make narrow, justified changes, preserve unrelated behavior, and verify work
with appropriate commands. Never claim a file edit or test result unless a
workspace observation confirms it."""


@dataclass
class CodingAgent(Agent):
    """Coordinate a coding workflow without conflating intent and execution.

    There are three supported modes:

    - ``run_workflow``: the model creates the plan and selects actions;
    - ``run_plan``: the caller owns the plan, while the model selects actions;
    - ``run_actions``: the caller owns both plan and actions; no model is used.

    Every workspace result is appended to ``current_run.results`` before the
    next decision, ensuring later decisions are grounded in observed state.
    """

    model: ModelBackend
    system_prompt: str = CODING_SYSTEM_PROMPT
    loop: CodingLoop = field(default_factory=CodingLoop)
    current_run: CodingRun | None = field(default=None, init=False)

    def run(self, user_input: str) -> str:
        """Run a conversational coding turn without workspace tools."""
        return super().run(user_input)

    def run_workflow(
        self,
        task: str,
        workspace: CodingWorkspace,
        *,
        max_steps: int = 20,
    ) -> str:
        """Generate a plan, execute bounded actions, and summarize."""
        logger.info("automatic coding workflow started max_steps=%d", max_steps)
        context = workspace.inspect(task)
        logger.debug("workspace inspection completed context_chars=%d", len(context))
        plan = CodingPlan.from_model(
            self._complete_step(build_plan_prompt(task, context))
        )
        logger.info("model-generated coding plan created chars=%d", len(plan.text))
        logger.debug("model-generated coding plan:\n%s", plan.text)
        return self._execute_model_loop(
            task,
            plan,
            workspace,
            max_steps=max_steps,
        )

    def run_plan(
        self,
        task: str,
        plan: str,
        workspace: CodingWorkspace,
        *,
        max_steps: int = 20,
    ) -> str:
        """Use an exact user-authored plan while the model selects actions."""
        logger.info("user-plan coding workflow started max_steps=%d", max_steps)
        return self._execute_model_loop(
            task,
            CodingPlan.from_user(plan),
            workspace,
            max_steps=max_steps,
        )

    def run_actions(
        self,
        task: str,
        plan: str,
        actions: Sequence[CodingAction],
        workspace: CodingWorkspace,
    ) -> str:
        """Execute explicit user actions without making any model calls.

        This mode is deterministic with respect to the supplied workspace. It
        stops at the first failed action and never asks a model to repair or
        reinterpret the caller's instructions.
        """
        logger.info("explicit-action workflow started actions=%d", len(actions))
        run = self._initialize_run(task, CodingPlan.from_user(plan), workspace)
        for action in actions:
            self.loop.begin_action(run)
            logger.info(
                "executing user action step=%d kind=%s",
                run.step,
                action.kind.value,
            )
            logger.debug(
                "user-authored action step=%d:\n%s",
                run.step,
                _describe_action(action),
            )
            result = workspace.execute(action)
            self.loop.record_result(run, result)
            if not result.success:
                reason = (
                    f"User action {action.kind.value} failed at step {run.step}: "
                    f"{result.output}"
                )
                self.loop.fail(run, reason)
                self._record_workflow_turn(run.task, reason)
                return reason

        response = self._deterministic_summary(run, workspace.changes())
        self.loop.complete(run, response)
        self._record_workflow_turn(run.task, response)
        return response

    def _execute_model_loop(
        self,
        task: str,
        plan: CodingPlan,
        workspace: CodingWorkspace,
        *,
        max_steps: int,
    ) -> str:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        run = self._initialize_run(task, plan, workspace)

        for _ in range(max_steps):
            logger.debug("requesting next model action step=%d", run.step + 1)
            action_response = self._complete_step(
                build_action_prompt(
                    task=run.task,
                    plan=plan,
                    observations=run.results,
                )
            )
            logger.debug(
                "model-selected coding step=%d:\n%s",
                run.step + 1,
                action_response,
            )
            action = parse_action(action_response)
            logger.info(
                "model selected action step=%d kind=%s",
                run.step + 1,
                action.kind.value,
            )
            logger.debug(
                "workspace action to execute step=%d:\n%s",
                run.step + 1,
                _describe_action(action),
            )

            self.loop.begin_action(run)
            result = workspace.execute(action)
            self.loop.record_result(run, result)
            logger.debug(
                "workspace observation step=%d output_chars=%d",
                run.step,
                len(result.output),
            )

            if action.kind is ActionKind.FINISH and result.success:
                response = self._complete_step(
                    build_summary_prompt(
                        task=run.task,
                        plan=plan,
                        observations=run.results,
                        diff=workspace.changes(),
                    )
                )
                self.loop.complete(run, response)
                self._record_workflow_turn(run.task, response)
                return response

        reason = f"Coding workflow reached the {max_steps}-step limit."
        self.loop.fail(run, reason)
        self._record_workflow_turn(run.task, reason)
        return reason

    def _initialize_run(
        self,
        task: str,
        plan: CodingPlan,
        workspace: CodingWorkspace,
    ) -> CodingRun:
        run = self.loop.start(task)
        self.current_run = run
        logger.debug("initializing workspace state for coding run")
        self.loop.begin_planning(run, workspace.inspect(run.task))
        self.loop.set_plan(run, plan)
        return run

    def _complete_step(self, prompt: str) -> str:
        """Execute one model decision without logging prompt contents."""
        logger.debug("requesting model completion prompt_chars=%d", len(prompt))
        response = self.model.complete(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt),
            ]
        ).strip()
        if not response:
            raise RuntimeError("The model returned an empty response.")
        logger.debug("model completion received response_chars=%d", len(response))
        return response

    def _record_workflow_turn(self, task: str, response: str) -> None:
        self.history.extend(
            [
                Message(role="user", content=task),
                Message(role="assistant", content=response),
            ]
        )

    @staticmethod
    def _deterministic_summary(run: CodingRun, diff: str) -> str:
        successful = sum(result.success for result in run.results)
        return (
            f"Executed {successful} user-authored action(s).\n"
            f"Final diff:\n{diff or '(no changes)'}"
        )


def _describe_action(action: CodingAction) -> str:
    """Format one action for detailed diagnostics without empty fields."""
    details = [f"kind: {action.kind.value}"]
    for field_name in (
        "path",
        "query",
        "old_text",
        "new_text",
        "content",
        "script",
        "message",
    ):
        value = getattr(action, field_name)
        if value is not None:
            details.append(f"{field_name}: {value}")
    if action.command:
        details.append(f"command: {list(action.command)!r}")
    return "\n".join(details)
