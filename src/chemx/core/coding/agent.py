"""Coding-specialized orchestration over plans, actions, and observations.

The agent owns decision-making but does not directly access the filesystem or
process environment. All effects pass through ``CodingWorkspace``, which keeps
execution policy replaceable and makes action results available for auditing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ...backends import Message, ModelBackend
from ..agent import Agent
from ..context import fit_messages, truncate_text
from .action import ActionKind, ActionResult, CodingAction, parse_action
from .context import build_observation_compaction_prompt
from .loop import CodingLoop, CodingRun
from .plan import CodingPlan
from .workflow import CODING_WORKFLOW, WorkflowProfile
from .workspace import CodingWorkspace

logger = logging.getLogger(__name__)

ProgressOutput = Callable[[str], None]


CODING_SYSTEM_PROMPT = CODING_WORKFLOW.system_prompt


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
    workflow: WorkflowProfile = CODING_WORKFLOW
    loop: CodingLoop = field(default_factory=CodingLoop)
    progress_output: ProgressOutput | None = field(default=None, repr=False)
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
            self._complete_step(self.workflow.build_plan_prompt(task, context))
        )
        logger.info("model-generated coding plan created chars=%d", len(plan.text))
        self._emit_progress(f"Plan:\n{plan.text}")
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
        coding_plan = CodingPlan.from_user(plan)
        self._emit_progress(f"Plan:\n{coding_plan.text}")
        return self._execute_model_loop(
            task,
            coding_plan,
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
        coding_plan = CodingPlan.from_user(plan)
        self._emit_progress(f"Plan:\n{coding_plan.text}")
        run = self._initialize_run(task, coding_plan, workspace)
        for action in actions:
            self.loop.begin_action(run)
            logger.info(
                "executing user action step=%d kind=%s",
                run.step,
                action.kind.value,
            )
            self._emit_progress(
                f"Step {run.step}:\n{_describe_action(action)}"
            )
            result = workspace.execute(action)
            self.loop.record_result(run, result)
            self._emit_result(run.step, result)
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
        assert self.context_policy is not None
        run = self._initialize_run(task, plan, workspace)
        lifecycle = (
            self.workflow.lifecycle_factory()
            if self.workflow.lifecycle_factory is not None
            else None
        )

        for _ in range(max_steps):
            self._compact_observations_if_needed(run)
            logger.debug("requesting next model action step=%d", run.step + 1)
            recent_results = run.results[run.compacted_result_count :]
            action_prompt = self.workflow.build_action_prompt(
                run.task,
                plan,
                recent_results,
                self.context_policy,
                run.observation_summary,
            )
            action = self._select_action(
                action_prompt,
                validator=(
                    lifecycle.validate
                    if lifecycle is not None
                    else None
                ),
            )
            logger.info(
                "model selected action step=%d kind=%s",
                run.step + 1,
                action.kind.value,
            )
            self._emit_progress(
                f"Step {run.step + 1}: {_describe_action(action)}"
            )

            self.loop.begin_action(run)
            result = workspace.execute(action)
            self.loop.record_result(run, result)
            if lifecycle is not None:
                lifecycle.record(result)
            self._emit_result(run.step, result)
            logger.debug(
                "workspace observation step=%d output_chars=%d",
                run.step,
                len(result.output),
            )

            if action.kind is ActionKind.FINISH and result.success:
                self._compact_observations_if_needed(run)
                recent_results = run.results[run.compacted_result_count :]
                response = self._complete_step(
                    self.workflow.build_summary_prompt(
                        run.task,
                        plan,
                        recent_results,
                        workspace.changes(),
                        self.context_policy,
                        run.observation_summary,
                    )
                )
                self.loop.complete(run, response)
                self._record_workflow_turn(run.task, response)
                return response

        reason = (
            f"{self.workflow.name} workflow reached the {max_steps}-step limit."
        )
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
        assert self.context_policy is not None
        messages = fit_messages(
            system_prompt=self.system_prompt,
            history=(),
            current=Message(role="user", content=prompt),
            policy=self.context_policy,
        )
        response = self.model.complete(messages).strip()
        if not response:
            raise RuntimeError("The model returned an empty response.")
        logger.debug("model completion received response_chars=%d", len(response))
        return response

    def _select_action(
        self,
        prompt: str,
        *,
        validator: Callable[[CodingAction], None] | None = None,
    ) -> CodingAction:
        """Parse one model action, allowing one structured correction attempt."""
        response = self._complete_step(prompt)
        try:
            action = parse_action(response)
            if validator is not None:
                validator(action)
            return action
        except ValueError as first_error:
            validation_error = str(first_error)
            logger.warning("invalid model action; requesting correction: %s", first_error)
            self._emit_progress(
                "The model returned an invalid action; requesting corrected JSON."
            )

        repair_prompt = (
            f"{prompt}\n\n"
            "The previous response was invalid and no workspace action ran.\n"
            f"Validation error: {validation_error}\n"
            "Invalid response:\n"
            f"{truncate_text(response, 2_000)}\n\n"
            "Return exactly one corrected JSON action using a supported kind. "
            "Return JSON only."
        )
        repaired_response = self._complete_step(repair_prompt)
        try:
            action = parse_action(repaired_response)
            if validator is not None:
                validator(action)
            return action
        except ValueError as second_error:
            raise ValueError(
                "Model returned an invalid action twice. "
                f"Final validation error: {second_error}"
            ) from second_error

    def _compact_observations_if_needed(self, run: CodingRun) -> None:
        """Summarize older tool results once the raw observation limit is reached."""
        assert self.context_policy is not None
        uncompacted = run.results[run.compacted_result_count :]
        compact_count = 0
        remaining_chars = sum(
            min(len(result.output), self.context_policy.max_observation_chars)
            for result in uncompacted
        )
        while (
            len(uncompacted) - compact_count
            > self.context_policy.max_observations
            or remaining_chars
            > self.context_policy.max_observation_chars_total
        ):
            result = uncompacted[compact_count]
            remaining_chars -= min(
                len(result.output),
                self.context_policy.max_observation_chars,
            )
            compact_count += 1

        if compact_count == 0:
            return

        to_compact = uncompacted[:compact_count]
        prompt = build_observation_compaction_prompt(
            run.observation_summary,
            to_compact,
        )
        run.observation_summary = self._complete_step(prompt)
        run.compacted_result_count += len(to_compact)
        logger.info(
            "coding observations compacted results=%d summary_chars=%d",
            len(to_compact),
            len(run.observation_summary),
        )

    def _record_workflow_turn(self, task: str, response: str) -> None:
        self.history.extend(
            [
                Message(role="user", content=task),
                Message(role="assistant", content=response),
            ]
        )

    def _emit_progress(self, message: str) -> None:
        if self.progress_output is not None:
            self.progress_output(message)

    def _emit_result(self, step: int, result: ActionResult) -> None:
        status = "ok" if result.success else "failed"
        self._emit_progress(
            f"Result {step} ({status}): {result.action.kind.value}"
        )

    @staticmethod
    def _deterministic_summary(run: CodingRun, diff: str) -> str:
        successful = sum(result.success for result in run.results)
        return (
            f"Executed {successful} user-authored action(s).\n"
            f"Final diff:\n{diff or '(no changes)'}"
        )


def _describe_action(action: CodingAction) -> str:
    """Format action metadata without exposing payload contents."""
    details = [f"kind: {action.kind.value}"]
    if action.path is not None:
        details.append(f"path: {action.path}")
    for field_name in ("query", "old_text", "new_text", "content", "script", "message"):
        value = getattr(action, field_name)
        if value is not None:
            details.append(f"{field_name}: [redacted, {len(value)} chars]")
    if action.command:
        details.append(f"command: {list(action.command)!r}")
    return "; ".join(details)
