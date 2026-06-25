# Installation

## Prerequisites

- **Python 3.10 or later**  
  Check your version:
  ```sh
  python3 --version
  ```
- **Git** (optional, for cloning the repository)
- **Ollama** (optional, only if you plan to use the Ollama backend)  
  [Install Ollama](https://ollama.com) and ensure the `ollama` command is available.

## Install chemx

Clone the repository and install the package in editable mode:

```sh
git clone https://github.com/example/chemx.git
cd chemx
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e .
```

The only direct Python dependency is [`prompt-toolkit`](https://python-prompt-toolkit.readthedocs.io/) (version ≥ 3.0.48). It is installed automatically.

All model backends (Ollama, OpenAI, DeepSeek) are included in the package. No extra Python packages are required – the necessary HTTP client logic is bundled.

## Verify the installation

```sh
chemx --help
```

If you see the command-line help, chemx is ready. The default provider is `ollama` with its default model. To use a different backend, see [Run](run.md).

## Environment setup (backends)

Set API keys **before** running for the OpenAI or DeepSeek backends:

```sh
export OPENAI_API_KEY="your-openai-api-key"     # for OpenAI
export DEEPSEEK_API_KEY="your-deepseek-api-key" # for DeepSeek
```

No environment variables are needed for the Ollama backend unless you override the default endpoint.
