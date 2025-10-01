# Repository Guidelines

## Project Structure & Module Organization
The core agent logic lives in `src/`, with subpackages for `agents/`, `orchestration/`, `browser/`, and `grid/`. Runtime configuration and environment handling sit under `src/config/` and `src/security/`. Automated test suites are in `tests/`, while reusable JSON scenarios and prompts live in `test_scenarios/` and `demo_journals/`. Visual assets and generated evidence land in `debug_screenshots/`, `reports/`, and `htmlcov/`. Keep experimental notebooks or analysis in `docs/` or `examples/` so the runtime modules stay focused.

## Build, Test, and Development Commands
Install dependencies locally with `pip install -e ".[dev]"`, then provision Playwright using `playwright install chromium`. Run the CLI entry point via `python -m src.main -j test_scenarios/wikipedia_search.json` or switch to interactive planning with `python -m src.main --requirements`. Execute the full test suite using `pytest`; deselect long scenarios with `pytest -m "not slow"`. Pre-flight checks should cover `ruff check`, `black .`, `isort .`, and `mypy src` (or run `pre-commit run --all-files`).

## Coding Style & Naming Conventions
Python code targets 3.10+ with 4-space indentation. Formatting follows Black (88-character lines) and isort’s Black profile; Ruff enforces linting and import rules. Favor descriptive `snake_case` for functions and variables, `PascalCase` for classes, and `SCREAMING_SNAKE_CASE` for constants. Type hints are required on public functions, matching the `mypy` configuration. Place new agents under `src/agents/` and shared utilities in `src/core/` to keep responsibilities clear.

## Testing Guidelines
Primary tests live in `tests/` with files named `test_*.py`; asynchronous cases use `pytest-asyncio`. Apply the `unit`, `integration`, and `slow` markers defined in `pytest.ini`. The default command `pytest --cov=src` must exceed the 60% coverage floor and produces HTML output in `htmlcov/index.html`. Scenario-driven regressions go in `test_scenarios/`, and long-running exploratory suites should surface evidence via `reports/` or `demo_output/` for review.

## Commit & Pull Request Guidelines
Commits should be imperative, concise, and scoped (e.g., “Integrate Gemini 2.5 Flash for enhanced grid accuracy”). Reference related issues when available and include artifacts—logs, screenshots, coverage deltas—in the commit or PR description. Pull requests should describe the agent-level impact, list verification commands, and call out any new environment variables or Playwright requirements. When altering orchestration or agent behavior, attach a brief note in `docs/` or `GRID_TEST_PROGRESS.md` summarizing the change for downstream contributors.
