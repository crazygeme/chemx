# chemx

chemx is a minimal Python foundation for specialized agents. It provides an
interactive coding frontend, explicit domain workflows, and interchangeable
local and remote model backends.

## Architecture

```text
Frontend
 │
 └── CLI                 terminal input and output
 │
 ├── selects and configures a model backend
 │
 ▼
CodingAgent
 │
 ├── owns the coding system prompt
 ├── stores conversation history
 └── tracks initialize, reason, act, verify, and complete phases
 │
 ▼
ModelBackend protocol
 │
 ▼
Backend registry
 ├── OllamaBackend       local models through Ollama
 ├── OpenAIBackend       remote models through the OpenAI API
 └── DeepSeekBackend     remote models through the DeepSeek API
```

The coding agent depends only on the `ModelBackend` protocol. Its initial loop
supports the typical coding-agent cycle:

1. Inspect folder structure and optional repository status.
2. Create a natural-language implementation plan.
3. Select one structured action at a time.
4. Read, search, edit, create, or run commands through the workspace.
5. Feed every result back into the next action decision.
6. Review Git or filesystem changes and summarize observed results.

The plan is never treated as executable code. `CodingAction` represents concrete
operations such as `read_file`, `search_text`, `replace_text`, `create_file`,
`write_file`, and `run_command`. Git is optional: repositories receive a final
diff, while ordinary folders receive a created/modified/deleted summary.

A user-authored plan can replace automatic planning while the model still
selects individual tool actions:

```python
result = agent.run_plan(
    task="Handle empty config values",
    plan="Inspect parse_config, add empty-value validation, and run tests.",
    workspace=workspace,
)
```

For a fully model-free run, provide explicit `CodingAction` objects to
`run_actions()`. This is analogous to a reviewed automation script rather than
a natural-language plan.

The chemical-industry core provides a separate, safety-gated lifecycle:
intake, safety review, data gathering, analysis, validation, reporting, and
completion. A workflow can be blocked when safety information or evidence is
insufficient, and failed validation returns the workflow to analysis.

## Installation

Python 3.10 or later is required.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

## Local model

Install Ollama and ensure that the selected model is available:

```sh
ollama pull llama3.2
chemx --provider ollama --model llama3.2
```

When the local Ollama API is not running, chemx starts `ollama serve`
automatically and waits for it to become ready. If Ollama is not installed, the
CLI displays installation guidance. It does not install Ollama or download a
model automatically.

The default endpoint is `http://localhost:11434`. It can be changed with
`--base-url`; remote endpoints are checked but never started locally.

## Remote model

Set an API key in the environment and select an OpenAI model:

```sh
export OPENAI_API_KEY="your-openai-api-key"
chemx --provider openai --model gpt-4.1-mini
```

An OpenAI-compatible service can be selected with `--base-url`. The default
endpoint is `https://api.openai.com/v1`.

## DeepSeek model

Set a DeepSeek API key and select the provider:

```sh
export DEEPSEEK_API_KEY="your-deepseek-api-key"
chemx --provider deepseek --model deepseek-v4-flash
```

The default DeepSeek endpoint is `https://api.deepseek.com`. The default model
is `deepseek-v4-flash`; `deepseek-v4-pro` can be selected with `--model`.

## CLI

```text
chemx [--agent {coding,chemicals}]
      [--provider {deepseek,ollama,openai}] [--model MODEL]
      [--base-url URL]
      [--system-prompt TEXT] [--timeout SECONDS]
      [--task TEXT] [--workspace PATH] [--plan-file PLAN]
      [--actions-file JSON] [--verify-command COMMAND] [--max-steps N]
      [-v | -vv]
```

The coding workflow is selected by default. Select the chemical-industry
workflow explicitly:

```sh
chemx --agent chemicals --provider ollama --model llama3.2
```

Runtime logging is disabled except for errors by default. Use `-v` for
high-level lifecycle status or `-vv` for detailed backend, planning, action,
workspace, and tool diagnostics:

```sh
chemx -vv --task "Fix the parser"
```

Logs are written to standard error; normal agent output remains on standard
output.

During an interactive session:

- Enter a message to run one agent turn.
- Enter `/clear` to discard conversation history.
- Enter `/exit` or press `Ctrl-D` to stop.

## Coding workspace workflow

Run a typical coding-agent cycle against a local folder:

```sh
chemx --agent coding \
  --provider ollama \
  --workspace . \
  --task "Add validation for empty configuration values"
```

The agent inspects the folder, automatically creates a prose plan, then
selects structured actions one at a time. File edits use exact-text replacement
or explicit file creation. Commands are passed as argument arrays rather than
through a shell. The local workspace rejects commands that were not explicitly
allowlisted; the CLI allowlists only `--verify-command`. The final Git diff is
shown to the model only for reporting when Git is available. Non-Git folders
are supported and receive a filesystem change summary.

The default verification command is:

```sh
python3 -m unittest discover -s tests -v
```

It can be replaced:

```sh
chemx --task "Fix the parser" --verify-command "python3 -m pytest -q"
```

To supply the plan while retaining model-selected actions:

```sh
chemx --task "Apply reviewed parser fix" --plan-file plan.txt
```

To avoid the model entirely, provide both a plan and an explicit JSON action
list:

```sh
chemx --task "Apply reviewed parser fix" \
  --plan-file plan.txt \
  --actions-file actions.json
```

Example `actions.json`:

```json
[
  {"kind": "read_file", "path": "src/parser.py"},
  {
    "kind": "replace_text",
    "path": "src/parser.py",
    "old_text": "return value",
    "new_text": "return validate(value)"
  },
  {
    "kind": "run_command",
    "command": ["python3", "-m", "unittest", "tests.test_parser"]
  }
]
```

The package can also run without installation:

```sh
PYTHONPATH=src python3 -m chemx --provider ollama --model llama3.2
```

## Project layout

```text
src/chemx/core/agent.py            conversation state and run loop
src/chemx/core/coding/             coding prompt and task lifecycle
src/chemx/core/coding/plan.py      plan types and prompt formatting
src/chemx/core/coding/action.py    structured coding tool operations
src/chemx/core/coding/workspace.py coding environment boundary
src/chemx/core/coding/local_workspace.py local folder implementation
src/chemx/tools/                    file, search, command, Bash, and Git tools
src/chemx/core/chemicals/          chemical-industry prompt and workflow
src/chemx/backends/base.py         shared backend protocol and message types
src/chemx/backends/registry.py     backend registration and construction
src/chemx/backends/ollama/         Ollama implementation
src/chemx/backends/openai/         OpenAI-compatible implementation
src/chemx/backends/deepseek/       DeepSeek implementation
src/chemx/frontends/cli/           command-line frontend
tests/                             unit tests
```

## Development

Run the test suite with:

```sh
python3 -m unittest discover -s tests -v
```
