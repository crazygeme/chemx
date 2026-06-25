# Running chemx

chemx is invoked through the `chemx` console script installed in your
environment. The CLI supports interactive sessions and one-shot coding tasks.

## Basic command

```sh
chemx
```

This starts an interactive coding session with the default Ollama backend
(`ollama` provider, default model). The prompt accepts natural-language coding
requests and displays the agent’s progress.

**Interactive controls**

- **Enter** – submits the current line.
- **Shift+Enter** or **Alt+Enter** – inserts a newline for multi-line input.
- **Arrow keys, Ctrl-A, Ctrl-E, Ctrl-W** – standard line editing.
- **`/clear`** – discards conversation history.
- **`/exit`** or **Ctrl-D** – ends the session.

## CLI arguments

```text
chemx [--agent {chemicals,coding}]
      [--provider {deepseek,ollama,openai}] [--model MODEL]
      [--base-url URL]
      [--system-prompt TEXT] [--timeout SECONDS]
      [--task TEXT] [--workspace PATH] [--plan-file PLAN]
      [--actions-file JSON] [--max-steps N]
      [-v | -vv]
```

All arguments are optional; defaults are shown below.

### Agent selection

- **`--agent chemicals`** – chemical-industry workflow (conversational mode).
- **`--agent coding`** – coding workspace workflow (default).

The `--task` argument is only supported with the coding agent.

### Model backend

- **`--provider`** – `ollama` (default), `openai`, or `deepseek`.
- **`--model`** – model name (provider-specific default if omitted).
- **`--base-url`** – overrides the API endpoint URL.

### Request tuning

- **`--system-prompt TEXT`** – replaces the agent’s default system prompt.
- **`--timeout SECONDS`** – request timeout, default `60.0`.

### One-shot coding tasks

- **`--task TEXT`** – coding workflow task; runs and exits.
- **`--workspace PATH`** – local folder for the task (default: current directory).
- **`--plan-file PLAN`** – use a pre-written plan instead of auto‑generation.
- **`--actions-file JSON`** – execute explicit JSON actions (requires `--plan-file`).
- **`--max-steps N`** – maximum tool actions per task (default: 50).

### Verbosity

- **`-v`** – high-level lifecycle status to stderr.
- **`-vv`** – detailed backend, planning, action, workspace, and tool diagnostics.

Default output goes to stdout; logs go to stderr.

## Selecting a backend

### Ollama (local)

```sh
# Ensure the model is pulled first
ollama pull llama3.2

chemx --provider ollama --model llama3.2
```

If Ollama isn’t running, chemx starts `ollama serve` automatically and waits
for readiness. The default endpoint is `http://localhost:11434`; override with
`--base-url`. Remote Ollama endpoints are checked but never started locally.

### OpenAI (remote)

```sh
export OPENAI_API_KEY="your-key"
chemx --provider openai --model gpt-4.1-mini
```

Default endpoint: `https://api.openai.com/v1`. Use `--base-url` for
OpenAI‑compatible services (e.g., Azure).

### DeepSeek (remote)

```sh
export DEEPSEEK_API_KEY="your-key"
chemx --provider deepseek --model deepseek-v4-flash
```

Default endpoint: `https://api.deepseek.com`. The default model is
`deepseek-v4-flash`; `deepseek-v4-pro` can be selected with `--model`.

## Coding workflow examples

**Auto‑plan coding task**

```sh
chemx --task "Add validation for empty configuration values" --workspace myproject/
```

The agent inspects the workspace, creates a plan, then executes actions one at
a time. File edits use exact-text replacement. Proposed shell commands are
shown and require explicit approval (`y/N`).

**User‑supplied plan, model‑selected actions**

```sh
chemx --task "Apply reviewed parser fix" --plan-file plan.txt --workspace .
```

**Fully scripted actions (no model calls)**

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

## Running without installation

If you prefer not to install the package, set the `PYTHONPATH`:

```sh
PYTHONPATH=src python3 -m chemx --provider ollama --model llama3.2
```

All CLI arguments are supported this way.

## Chemicals agent (interactive)

The chemicals workflow is conversational and does not use a workspace. Start it
with:

```sh
chemx --agent chemicals --provider ollama --model llama3.2
```

Enter messages one at a time. The agent responds after each turn.

## Logging

Use `-v` or `-vv` to see lifecycle and diagnostic information on stderr:

```sh
chemx -vv --task "Fix the parser"
```

Coding workflows always show the generated plan, the action selected, and
its workspace result status as normal output. Action payloads and workspace
results are redacted from the progress display. Backend request and response
bodies are not logged.
