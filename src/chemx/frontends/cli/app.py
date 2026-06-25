"""Command-line frontend for chemx."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Protocol

from ...backends import (
    ModelBackend,
    ModelError,
    available_backends,
    create_backend,
    get_backend_registration,
    prepare_backend,
)
from ...core import (
    CHEMICALS_SYSTEM_PROMPT,
    CODING_SYSTEM_PROMPT,
    Agent,
    CodingAction,
    ChemicalsAgent,
    CodingAgent,
    LocalWorkspace,
    create_coding_session,
)
from ...core.observability import configure_logging

AgentFactory = Callable[..., Agent]


class InteractiveSession(Protocol):
    """Common interface used by the terminal interaction loop."""

    def run(self, user_input: str) -> str:
        """Run one interactive turn."""

    def clear(self) -> None:
        """Clear retained session state."""


@dataclass
class _CodingOutput:
    """Render one coding turn beneath a single ``chemx>`` prefix."""

    started: bool = False

    def begin(self) -> None:
        self.started = False

    def progress(self, message: str) -> None:
        if self.started:
            print(message)
        else:
            print(f"chemx> {message}")
            self.started = True

    def finish(self, response: str) -> None:
        if self.started:
            print(response)
        else:
            print(f"chemx> {response}")
        self.started = False


class _NoModelBackend:
    """Sentinel backend for strict user-plan execution."""

    context_window_tokens = 32_768

    def complete(self, messages: Sequence[object]) -> str:
        raise RuntimeError("Strict user-plan mode does not permit model calls.")


@dataclass(frozen=True)
class AgentProfile:
    """CLI construction metadata for a specialized agent."""

    factory: AgentFactory
    default_system_prompt: str


AGENT_PROFILES = {
    "chemicals": AgentProfile(
        factory=ChemicalsAgent,
        default_system_prompt=CHEMICALS_SYSTEM_PROMPT,
    ),
    "coding": AgentProfile(
        factory=CodingAgent,
        default_system_prompt=CODING_SYSTEM_PROMPT,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chemx",
        description="Run an interactive specialized agent.",
    )
    parser.add_argument(
        "--agent",
        choices=tuple(sorted(AGENT_PROFILES)),
        default="coding",
        help="agent workflow (default: coding)",
    )
    parser.add_argument(
        "--provider",
        choices=available_backends(),
        default="ollama",
        help="model provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model name (defaults depend on the provider)",
    )
    parser.add_argument("--base-url", help="override the provider API base URL")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="override the selected agent's system prompt",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="model request timeout in seconds",
    )
    parser.add_argument("--task", help="run one coding workspace task and exit")
    parser.add_argument(
        "--workspace",
        default=".",
        help="local folder used by coding workflows (default: current directory)",
    )
    parser.add_argument(
        "--plan-file",
        help="use this natural-language coding plan instead of generating one",
    )
    parser.add_argument(
        "--actions-file",
        help="execute explicit JSON actions without using a model",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="maximum model-selected tool actions per coding task (default: 20)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="show runtime status; repeat for detailed diagnostics",
    )
    return parser


def create_agent(
    agent_name: str,
    model: ModelBackend,
    *,
    system_prompt: str | None = None,
) -> Agent:
    """Create the specialized agent selected by the CLI."""
    profile = AGENT_PROFILES[agent_name]
    return profile.factory(
        model=model,
        system_prompt=system_prompt or profile.default_system_prompt,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    model_name = args.model or get_backend_registration(args.provider).default_model
    coding_output = _CodingOutput()

    if (args.plan_file or args.actions_file) and args.task is None:
        print(
            "Configuration error: --plan-file and --actions-file require --task.",
            file=sys.stderr,
        )
        return 2
    if args.actions_file and not args.plan_file:
        print(
            "Configuration error: --actions-file requires --plan-file.",
            file=sys.stderr,
        )
        return 2
    if args.task is not None and args.agent != "coding":
        print(
            "Configuration error: --task is currently supported only by "
            "the coding agent.",
            file=sys.stderr,
        )
        return 2

    if args.task is not None and args.actions_file:
        try:
            workspace = _create_local_workspace(args)
            plan = Path(args.plan_file).read_text(encoding="utf-8")
            actions = _read_actions(Path(args.actions_file))
            profile = AGENT_PROFILES["coding"]
            agent = CodingAgent(
                model=_NoModelBackend(),
                system_prompt=args.system_prompt or profile.default_system_prompt,
                progress_output=coding_output.progress,
            )
            coding_output.begin()
            response = agent.run_actions(args.task, plan, actions, workspace)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Workspace error: {error}", file=sys.stderr)
            return 2
        print("chemx: coding using explicit user actions")
        coding_output.finish(response)
        return 0

    try:
        backend = create_backend(
            provider=args.provider,
            model=model_name,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        prepare_backend(backend)
    except (ModelError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    agent = create_agent(
        args.agent,
        backend,
        system_prompt=args.system_prompt,
    )
    if isinstance(agent, CodingAgent):
        agent.progress_output = coding_output.progress
    print(f"chemx: {args.agent} using {args.provider}/{model_name}")

    if args.task is not None:
        if not isinstance(agent, CodingAgent):
            raise RuntimeError("Coding profile did not create a CodingAgent.")
        try:
            workspace = _create_local_workspace(args)
            coding_output.begin()
            if args.plan_file:
                plan = Path(args.plan_file).read_text(encoding="utf-8")
                response = agent.run_plan(
                    args.task,
                    plan,
                    workspace,
                    max_steps=args.max_steps,
                )
            else:
                response = agent.run_workflow(
                    args.task,
                    workspace,
                    max_steps=args.max_steps,
                )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Workspace error: {error}", file=sys.stderr)
            return 2
        coding_output.finish(response)
        return 0

    session: InteractiveSession = agent
    if isinstance(agent, CodingAgent):
        try:
            workspace = _create_local_workspace(args)
            session = create_coding_session(
                agent,
                workspace,
                max_steps=args.max_steps,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Workspace error: {error}", file=sys.stderr)
            return 2

    print("Commands: /clear, /exit")

    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        command = user_input.strip().lower()
        if command in {"/exit", "/quit"}:
            return 0
        if command == "/clear":
            session.clear()
            print("chemx> Conversation cleared.")
            continue
        if not command:
            continue

        try:
            if isinstance(agent, CodingAgent):
                coding_output.begin()
            response = session.run(user_input)
        except (ModelError, RuntimeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            continue

        if isinstance(agent, CodingAgent):
            coding_output.finish(response)
        else:
            print(f"chemx> {response}")


def _create_local_workspace(args: argparse.Namespace) -> LocalWorkspace:
    root = Path(args.workspace).expanduser().resolve()
    return LocalWorkspace(
        root=root,
        command_approval=_approve_command,
    )


def _approve_command(command: tuple[str, ...]) -> bool:
    """Ask for explicit permission before running a workspace command."""
    try:
        answer = input(
            f"Allow command `{shlex.join(command)}`? [y/N] "
        )
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


def _read_actions(path: Path) -> list[CodingAction]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Actions file must contain a JSON list.")
    return [CodingAction.from_dict(item) for item in value]
