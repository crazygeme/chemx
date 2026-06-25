# ChemX Overall Architecture

## Overview

ChemX is a small, extensible AI agent framework specializing in software engineering and documentation tasks. It provides a modular architecture where **agents** orchestrate **tools** via **backend** model providers, and **frontends** handle user interaction. The core design emphasizes separation of concerns: conversation mechanics, domain-specific logic, tool execution, and backend abstraction are all independent.

## Directory Structure

```
src/chemx/
├── __init__.py          # Package root, exports public API symbols
├── __main__.py          # Entry point for `python -m chemx`
├── backends/            # Model provider abstraction & implementations
│   ├── __init__.py
│   ├── base.py           # Message, ModelBackend protocol, ModelError
│   ├── http.py           # Generic HTTP backend
│   ├── lifecycle.py      # Backend lifecycle helpers
│   ├── registry.py       # Backend registration & lookup
│   ├── openai/           # OpenAI backend
│   ├── deepseek/         # DeepSeek backend
│   └── ollama/           # Ollama backend
├── core/                 # Core agent logic
│   ├── __init__.py       # Re-exports from all submodules
│   ├── agent.py          # Base Agent class: conversation, compaction
│   ├── context.py        # ContextPolicy, token/character budgeting
│   ├── observability.py  # Logging configuration
│   ├── coding/           # Coding‑specific subsystem
│   │   ├── __init__.py   # Public API surface
│   │   ├── action.py     # ActionKind, CodingAction, ActionResult, parse_action
│   │   ├── agent.py      # CodingAgent: workflow orchestration
│   │   ├── context.py    # Observation compaction helper
│   │   ├── local_workspace.py  # LocalWorkspace implementation
│   │   ├── loop.py       # CodingLoop, CodingRun, CodingPhase
│   │   ├── plan.py       # CodingPlan, PlanSource, prompt builders
│   │   ├── router.py     # WorkflowRouter, WorkflowKind, WorkflowRoute
│   │   ├── session.py    # CodingSession factory
│   │   ├── workspace.py  # CodingWorkspace abstract base
│   │   └── workflow.py   # WorkflowProfile, CODING_WORKFLOW, DOCUMENT_WORKFLOW
│   └── chemicals/        # Chemical‑specific subsystem
│       ├── __init__.py
│       ├── agent.py      # ChemicalsAgent
│       └── loop.py       # ChemicalsLoop
├── frontends/            # User interface layers
│   ├── __init__.py
│   └── cli/              # CLI frontend
│       ├── __init__.py
│       ├── __main__.py   # `python -m chemx frontend cli` entry
│       └── app.py        # Application logic
└── tools/                # Tool definitions
    ├── __init__.py
    ├── command.py        # Shell command execution
    ├── filesystem.py     # File read/write/search
    ├── git.py            # Git operations
    └── search.py         # Text search utility
```

## Core Concepts

### Agent
The base `Agent` class (in `core/agent.py`) is a dataclass holding a `ModelBackend`, a system prompt, conversation history, and a `ContextPolicy`. Its `run()` method accepts user input, compacts history if needed, fits messages into the context window, calls the model, and appends the response to history. It does not have direct access to tools or workspace – those are added by subclasses.

### Context Policy
`ContextPolicy` (in `core/context.py`) defines token/character budgets, recent turn retention, and observation truncation limits. It is computed from the backend’s `context_window_tokens` and can be overridden. Helper functions (`fit_messages`, `estimate_message_tokens`, `truncate_text`) ensure messages fit within the policy.

### Tool
Tools are encapsulated as `CodingAction` objects (in `core/coding/action.py`). Each action has a `kind` (e.g., `read_file`, `run_command`) and required parameters. Actions are executed by a `CodingWorkspace` implementation, which returns `ActionResult` (success flag + output text).

### Backend
Backends implement the `ModelBackend` protocol (in `backends/base.py`): a `context_window_tokens` attribute and a `complete(messages)` method returning a string. Concrete backends (OpenAI, DeepSeek, Ollama, HTTP) are registered in `backends/registry.py` and can be selected at runtime.

### Frontend
Frontends provide user interaction. The CLI frontend (in `frontends/cli/`) parses commands and invokes the appropriate agent workflow. It configures logging via `configure_logging()` from `core/observability.py`.

### Workspace
A `CodingWorkspace` (abstract base in `core/coding/workspace.py`) abstracts filesystem and shell access. `LocalWorkspace` (in `core/coding/local_workspace.py`) implements it using the local filesystem and `subprocess`. The workspace isolates the agent from direct OS access, enabling auditing and policy enforcement.

## Data Flow

1. **User input** arrives through a frontend (e.g., `chemx run "task"`).
2. The frontend creates or retrieves a `CodingAgent` (or `ChemicalsAgent`) with a selected backend.
3. The agent calls `run_workflow()`, `run_plan()`, or `run_actions()` depending on the mode.
4. Inside the workflow:
   - The agent generates a plan (or uses a user‑supplied plan).
   - In a loop, the agent selects an action (via model call if not `run_actions` mode).
   - The action is executed by the workspace; the result is appended to the run’s `results`.
   - Observations may be compacted if they exceed policy limits.
   - When the model emits a `finish` action or the step limit is reached, a summary is produced.
5. The response is returned to the frontend, which displays it to the user and optionally records it in conversation history.

## Module Descriptions

### `core/agent.py` – Base Agent
- Maintains ordered history.
- Compacts history using the model itself when retention limits are exceeded.
- `run()` is a single conversational turn (no tools).
- Subclasses like `CodingAgent` compose this with workspace workflows.

### `core/context.py` – Context Policy & Budgeting
- `ContextPolicy` dataclass with validation.
- `fit_messages()` trims history to fit within token/character budgets.
- `estimate_message_tokens()` approximates token count.
- `truncate_text()` truncates from head/tail with a marker.

### `core/coding/` – Coding Subsystem
- **action.py**: Defines `ActionKind` enum, `CodingAction` dataclass, `ActionResult`, and `parse_action()` which parses JSON into action objects with tolerance for surrounding text or code fences. `_extract_actions()` finds valid JSON objects in malformed responses.
- **agent.py**: `CodingAgent` extends `Agent` with three workflow modes:
  - `run_workflow`: model generates plan and selects actions.
  - `run_plan`: user provides plan, model selects actions.
  - `run_actions`: user provides both plan and actions; no model calls.
  Includes `_compact_observations_if_needed()` to summarize old tool results.
- **context.py**: `build_observation_compaction_prompt()` produces a summary of older observations. `format_observations()` formats recent results for action prompts.
- **local_workspace.py**: `LocalWorkspace` executes actions on the real filesystem and shell. It sanitizes paths, limits output size, and handles errors gracefully.
- **loop.py**: `CodingLoop` manages run state (`CodingRun`, `CodingPhase`). Tracks step count, results, observation summary, and current phase (planning, execution, completed, failed).
- **plan.py**: `CodingPlan` with source (`PlanSource.MODEL` or `PlanSource.USER`). Prompt builders (`build_plan_prompt`, `build_action_prompt`, `build_summary_prompt`) construct detailed prompts for the model.
- **router.py**: `WorkflowRouter` analyzes a task and returns a `WorkflowRoute` containing the selected `WorkflowKind` and workspace files. Used by the CLI to decide which workflow to run.
- **session.py**: `CodingSession` holds an agent and its associated workspace. `create_coding_session()` factory sets up dependencies.
- **workspace.py**: Abstract `CodingWorkspace` with `inspect()`, `execute()`, and `changes()` methods.
- **workflow.py**: `WorkflowProfile` dataclass holding name, system prompt, and prompt builders. Contains two predefined profiles: `CODING_WORKFLOW` and `DOCUMENT_WORKFLOW`.

### `core/chemicals/` – Chemical Domain Subsystem
- **agent.py**: `ChemicalsAgent` – likely specialized for chemical data queries and computations (not deeply inspected).
- **loop.py**: `ChemicalsLoop` – run loop for chemical tasks.

### `backends/` – Model Backends
- **base.py**: `Message` dataclass (immutable, role + content). `ModelBackend` protocol. `ModelError` exception.
- **registry.py**: Registers backends by name; `get_backend()` returns an instance.
- **lifecycle.py**: Helper for backend lifecycle (e.g., startup/shutdown).
- **openai/backend.py**, **deepseek/backend.py**, **ollama/backend.py**: Provider‑specific implementations.
- **http.py**: General HTTP backend for REST‑compatible APIs.

### `frontends/cli/` – Command‑Line Interface
- **__main__.py**: Entry point for `python -m chemx frontend cli`.
- **app.py**: Main application loop, parses subcommands (`run`, `session`, etc.), instantiates backends and agents, handles progress output.

### `tools/` – Tool Implementations
- **filesystem.py**: Functions for reading, writing, searching files (used by `LocalWorkspace`).
- **command.py**: Run shell commands with timeout and output capture.
- **git.py**: Git operations (status, diff).
- **search.py**: Text search across files (grep‑like).

## Key Design Decisions

1. **Backend as Protocol**: Backends are not abstract base classes but structural typing via `ModelBackend` Protocol. This allows duck‑typing and easy addition of new providers without inheritance.
2. **Workspace Abstraction**: All filesystem and shell access goes through `CodingWorkspace`. This enables unit testing, sandboxing, and future remote workspace support.
3. **Observation Compaction**: Rather than discarding old tool outputs, the agent uses the model to summarize them. This preserves information while fitting context windows.
4. **Two‑Level Conversation History**: Base `Agent` keeps a chat history; `CodingAgent` additionally maintains a `CodingRun` with its own observations. The `run_workflow` method also records the final result in the base history for continuity.
5. **Deterministic Action Mode**: `run_actions` allows fully scripted execution without any model calls, useful for replaying logs or testing.
6. **Workflow Profiles**: `WorkflowProfile` separates prompt logic from agent logic. New workflows (e.g., documentation) can be added by defining a profile instead of subclassing.
7. **JSON‑Only Action Parsing**: The model is instructed to output only JSON, but the parser tolerates markdown fences and extraneous text, making it robust in practice.

## Dependencies

From `pyproject.toml` (inferred):
- **openai**: OpenAI API client.
- **httpx** (probable): Used by HTTP backend and others.
- **pydantic** (probable): Used for data validation? (Not observed in core, but common).
- **rich** or similar: CLI formatting (not observed).
- Standard library: `logging`, `json`, `subprocess`, `dataclasses`, `enum`, `os`, `pathlib`.

Exact dependency list should be verified from `pyproject.toml`.

## Glossary

| Term | Definition |
|------|------------|
| Agent | Orchestrator that manages conversation, chooses actions, and invokes models. |
| Tool | A capability the agent can use, represented as a `CodingAction` and executed by a workspace. |
| Backend | A model provider implementing the `ModelBackend` protocol. |
| Workspace | Abstraction layer that isolates the agent from direct OS access. |
| Context Policy | Rules for fitting messages into the model’s context window. |
| Router | Analyzes a task to select the appropriate workflow and workspace files. |
| Workflow Profile | A named collection of prompts (system, plan, action, summary) that define a task domain. |
| CodingAction | A structured tool invocation with a kind and required parameters. |
| ActionResult | The outcome of executing a CodingAction: success flag and output text. |
| CodingRun | State of a single coding workflow execution: task, plan, results, phase. |
| Compactation | Summarization of older messages or observations to fit within context windows. |

---

*This document reflects the state of the codebase as observed. For precise details, consult the inline docstrings and source files.*