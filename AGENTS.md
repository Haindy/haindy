# Repository Guidelines

This file is intentionally minimal.

## 1) Map: important files and docs

- `README.md`: setup + run quickstart.
- `src/main.py`: CLI entrypoint.
- `src/config/settings.py`: runtime configuration and env vars.
- `src/agents/`: orchestration and action agents.
- `src/agents/computer_use/session.py`: computer-use provider loop (OpenAI/Google).
- `src/core/`: shared types and interfaces.
- `src/journal/`: execution journaling and pattern matching.
- `src/monitoring/`: report generation and logs.
- `tests/`: automated tests.
- `test_scenarios/`: sample requirement/context inputs.
- `docs/RUNBOOK.md`: environment and operational notes.
- `docs/plans/`: implementation/refactor plans.

## 2) Rules: keep it clean

- Always use the local virtual environment:
  - `source .venv/bin/activate`
  - If missing: `python3 -m venv .venv`
- Install dependencies before running tools:
  - `.venv/bin/pip install -r requirements.lock`
  - `.venv/bin/pip install -e ".[dev]"`
- Install Playwright browser runtime:
  - `.venv/bin/playwright install chromium`
- Before finishing a change, run:
  - `.venv/bin/ruff check .`
  - `.venv/bin/ruff format .`
  - `.venv/bin/mypy src`
  - `.venv/bin/pytest`
- Prefer small, targeted changes. Avoid compatibility fallbacks unless explicitly requested.
