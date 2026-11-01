# curex

Ask questions about a CSV/Excel/JSON dataset in plain English; an LLM writes the pandas code and curex runs it in a restricted executor.

## Why

Exploratory analysis is mostly the same dozen pandas idioms. curex turns "average sales by region" into code, runs it against your DataFrame, and returns the output and any chart, with a choice of OpenAI, Anthropic or a local Ollama model behind it.

## Features

- Natural-language queries, with intent and entity extraction before code generation
- Pluggable LLM providers (OpenAI, Anthropic, local Ollama), switchable at runtime
- AST-validated, restricted-builtins executor for the generated code; plots saved as PNG
- Three interfaces: interactive CLI (Rich), FastAPI REST API, and a WebSocket stream
- CSV, Excel and JSON loading with dtype optimization for larger files
- Per-request usage, cost and timing tracking

## Quickstart

```bash
git clone <repository-url> curex && cd curex
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py server            # http://localhost:8000, interactive docs at /docs
```

In another shell (the server keeps the dataset and model in memory):

```bash
curl -X POST localhost:8000/data/sample          # 1,000-row demo sales dataset
curl -X POST localhost:8000/models/configure -H 'content-type: application/json' \
  -d '{"provider":"anthropic","model_name":"claude-3-5-sonnet-latest","api_key":"<your key>"}'
curl -X POST localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"query":"average sales by region"}'
```

Use `"provider":"openai"` with an OpenAI key, or `"provider":"local","model_name":"codellama","base_url":"http://localhost:11434"` with a running [Ollama](https://ollama.ai) (`ollama pull codellama`). Upload your own file with `curl -F file=@data.csv localhost:8000/data/upload`.

The CLI (`python main.py cli --help`) exposes the same commands (`load-data`, `configure-model`, `analyze`, ...), but each subcommand is a separate process and state is not persisted between them yet, so the server is the working end-to-end path.

## Architecture

- `src/nlp/` classifies the query (intent, columns, filters) into an `AnalysisRequest`.
- `src/llm/` is a provider abstraction (`LLMProvider` base + OpenAI/Anthropic/local) with a manager that picks a model by task complexity.
- `src/generation/` builds prompts with dataset schema context and post-processes generated code.
- `src/execution/executor.py` validates the code's AST (no imports, no `open`/`eval`/`exec`, no `os`/`sys` access), runs it with a whitelisted builtins table and preloaded `df`, `pd`, `np`, `plt`, `sns`, `px`, and captures stdout and figures.
- `src/app/` orchestrates the pipeline; `src/cli/` and `src/web/` are thin front ends over it.

## Security note

The executor is a guardrail against careless LLM output, not a security boundary: it is an in-process `exec` with an AST filter, and the `timeout` argument is not enforced. Run it only on your own data and machine; for untrusted users, put execution in a separate process or container.

## Testing

```bash
python -m pytest -q tests
```

The tests exercise the executor only (no network or LLM calls).

## Adding a provider

Subclass `LLMProvider` in `src/llm/providers/`, implement `generate_code`, `validate_config` and `get_capabilities`, and register it in `LLMManager`.

## License

MIT, see [LICENSE](LICENSE). Author: Pritesh Patel.
