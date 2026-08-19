# Repository Guidelines

This file is intentionally minimal.

## 1) Map: important files and docs

- `README.md`: setup + run quickstart.
- `.env.example`: supported env vars and default runtime knobs.
- `haindy/main.py`: CLI entrypoint.
- `haindy/config/settings.py`: runtime configuration and env vars.
- `haindy/runtime/environment.py`: canonical backend/environment normalization.
- `haindy/agents/`: orchestration and action agents.
- `haindy/agents/situational_agent.py`: entrypoint/setup assessment for desktop, web, and mobile contexts.
- `haindy/agents/computer_use/session.py`: computer-use provider loop (OpenAI/Google/Anthropic).
- `haindy/linux/`, `haindy/macos/`, `haindy/windows/`: desktop automation controllers, drivers, capture, replay, and input.
- `haindy/mobile/`: Android ADB and iOS idb automation controllers and drivers.
- `haindy/runtime/`: execution context building, caches, replay, and runtime helpers.
- `haindy/core/`: shared types and interfaces.
- `haindy/journal/`: execution journaling and pattern matching.
- `haindy/monitoring/`: report generation and logs.
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
- Desktop automation is platform-specific:
  - install the OS dependencies from `docs/RUNBOOK.md` when working on `haindy/linux/`, `haindy/macos/`, `haindy/windows/`, or running desktop flows
- Mobile automation supports Android through ADB and iOS through idb:
  - ensure the relevant tooling is available when working on `haindy/mobile/` or running mobile flows
- Treat backend semantics as shared contract:
  - if you change backend names, aliases, defaults, or target-type behavior, update `haindy/runtime/environment.py`, `haindy/config/settings.py`, `.env.example`, `README.md`, and relevant tests together
- Treat runtime/config surface as shared contract:
  - if you add or rename env vars, defaults, cache paths, or provider settings, update `haindy/config/settings.py`, `.env.example`, `README.md`, `docs/RUNBOOK.md`, and tests together
- Before release-facing, provider, runtime, or surface changes, manually run the repo-local `.agents/skills/haindy-self-regression` skill after installing the branch build. This is not required for every small edit or docs-only commit.
- Before finishing a change, run:
  - `.venv/bin/ruff check .`
  - `.venv/bin/ruff format .`
  - `.venv/bin/mypy haindy`
  - `.venv/bin/pytest`
- Prefer small, targeted changes. Avoid compatibility fallbacks unless explicitly requested.
