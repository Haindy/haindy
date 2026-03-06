# HAINDY

Desktop-first autonomous testing agent that turns requirements into executable test runs.

## Quick project map

- `src/main.py`: CLI entrypoint
- `src/agents/`: orchestrator/planner/runner/action agents
- `src/agents/computer_use/session.py`: computer-use action loop
- `src/config/settings.py`: configuration and env handling
- `tests/`: test suite
- `test_scenarios/`: sample inputs
- `docs/RUNBOOK.md`: operational setup notes
- `docs/plans/`: implementation plans

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[dev]"
```

3. Install Playwright runtime:
```bash
.venv/bin/playwright install chromium
```

4. Configure environment:
```bash
cp .env.example .env
```

Minimum required settings:
- `OPENAI_API_KEY` (used by orchestration)
- `CU_PROVIDER=openai`, `CU_PROVIDER=google`, or `CU_PROVIDER=anthropic`

Provider-specific:
- OpenAI computer-use: `HAINDY_COMPUTER_USE_MODEL` (default `gpt-5.4`)
- Google computer-use: `GOOGLE_CU_MODEL` and Vertex credentials/settings (see `.env.example`)
- Anthropic computer-use: `ANTHROPIC_API_KEY`, optional `ANTHROPIC_CU_MODEL` (default `claude-sonnet-4-6`), optional `ANTHROPIC_CU_MAX_TOKENS` (default `16384`)

## Run

Plan and context files are both required:
```bash
.venv/bin/python -m src.main --plan test_scenarios/wikipedia_search_simple.txt --context test_scenarios/wikipedia_search_simple.txt
```

Mobile ADB backend (hard override):
```bash
.venv/bin/python -m src.main --mobile --plan <plan_file> --context <mobile_context_file>
```

For `--mobile` runs, context should provide either:
- `adb_serial` + `app_package` (optional `app_activity`), or
- explicit `adb_commands` that discover/select device and open the target app.

Optional debug logging:
```bash
.venv/bin/python -m src.main --plan <plan_file> --context <context_file> --debug
```

## Developer checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/mypy src
.venv/bin/pytest
```
