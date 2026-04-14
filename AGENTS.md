# Repository Guidelines

This file is intentionally minimal.

## 1) Map: important files and docs

- `README.md`: setup + run quickstart.
- `.env.example`: supported env vars and default runtime knobs.
- `src/main.py`: CLI entrypoint.
- `src/config/settings.py`: runtime configuration and env vars.
- `src/runtime/environment.py`: canonical backend/environment normalization.
- `src/agents/`: orchestration and action agents.
- `src/agents/situational_agent.py`: entrypoint/setup assessment for desktop, web, and mobile contexts.
- `src/agents/computer_use/session.py`: computer-use provider loop (OpenAI/Google).
- `src/desktop/`: Linux/X11 desktop automation controller, driver, capture, replay, and input.
- `src/mobile/`: Android ADB automation controller and driver.
- `src/runtime/`: execution context building, caches, replay, and runtime helpers.
- `src/core/`: shared types and interfaces.
- `src/journal/`: execution journaling and pattern matching.
- `src/monitoring/`: report generation and logs.
- `tests/`: automated tests.
- `test_scenarios/`: sample requirement/context inputs.
- `docs/RUNBOOK.md`: environment and operational notes.
- `docs/design/`: current architecture docs.
- `docs/plans/`: implementation/refactor plans.

## 2) Rules: keep it clean

- Always use the local virtual environment:
  - `source .venv/bin/activate`
  - If missing: `python3 -m venv .venv`
- Install dependencies before running tools:
  - `.venv/bin/pip install -r requirements.lock`
  - `.venv/bin/pip install -e ".[dev]"`
- Desktop automation is Linux/X11-only today:
  - install the OS dependencies from `docs/RUNBOOK.md` when working on `src/desktop/` or running desktop flows
- Mobile automation uses ADB:
  - ensure `adb` is available when working on `src/mobile/` or `--mobile` flows
- Treat backend semantics as shared contract:
  - if you change backend names, aliases, defaults, or target-type behavior, update `src/runtime/environment.py`, `src/config/settings.py`, `.env.example`, `README.md`, and relevant tests together
- Treat runtime/config surface as shared contract:
  - if you add or rename env vars, defaults, cache paths, or provider settings, update `src/config/settings.py`, `.env.example`, `README.md`, `docs/RUNBOOK.md`, and tests together
- Before finishing a change, run:
  - `.venv/bin/ruff check .`
  - `.venv/bin/ruff format .`
  - `.venv/bin/mypy haindy`
  - `.venv/bin/pytest`
- Prefer small, targeted changes. Avoid compatibility fallbacks unless explicitly requested.
