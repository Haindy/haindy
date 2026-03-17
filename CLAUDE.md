# HAINDY - Autonomous AI Testing Agent

## Important Rules

1. **No emojis**: Do not use emojis in comments, PR descriptions, commit messages, or any other content.
2. **No backward compatibility**: This is a new tool and not in prod. There is no reason to require backwards compatibility or to mark things as deprecated. Just replace old code with new code.
3. **No timeouts on test runs**: NEVER use timeouts when running tests or any long-running operations. This is not a time-sensitive operation and tests are designed to run for extended periods of time.

## Project Overview

HAINDY is an autonomous AI testing agent that uses a multi-agent architecture to accept high-level requirements and autonomously execute testing workflows. The system coordinates specialized AI agents to plan, execute, and report on tests against desktop (Linux/X11), web, and mobile (Android ADB) targets using computer-use AI APIs.

## Development Runbook

Before running tests or demos, refer to `docs/RUNBOOK.md` for environment setup, OS dependencies, and troubleshooting.

## Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[dev]"
cp .env.example .env  # then fill in API keys
```

### Running
```bash
# Standard run (both flags required)
python -m src.main --plan <requirements_file> --context <context_file>

# Mobile ADB backend
python -m src.main --mobile --plan <plan> --context <context>

# Debug mode
python -m src.main --plan <plan> --context <context> --debug

# OAuth auth management
python -m src.main --codex-auth login|logout|status

# API connectivity test
python -m src.main --test-api
```

### Testing
```bash
.venv/bin/pytest
.venv/bin/pytest -m "not slow"
.venv/bin/pytest --cov=src
```

### Code Quality
```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/mypy src
```

## Architecture

### Agent Pipeline

```
User Input (requirements file + context file)
    |
    v
ScopeTriageAgent          normalize requirements, extract scope/exclusions/ambiguities
    |
    v
TestPlannerAgent          requirements -> structured TestPlan (cases + steps)
    |
    v
SituationalAgent          validate context adequacy; determine setup instructions
    |
    v
WorkflowCoordinator
    |
    v
TestRunner                iterate TestPlan -> TestCase -> TestStep
    |
    v
ActionAgent               execute each step via computer-use session
    |
    v
ComputerUseSession        OpenAI / Google / Anthropic provider loop
    |
    v
Automation Driver         desktop (X11/uinput) | mobile (ADB) | browser (legacy)
    |
    v
Report Generation         HTML report + JSONL execution log
```

### Key Source Locations

| Path | Purpose |
|------|---------|
| `src/main.py` | CLI entrypoint |
| `src/agents/` | All agent implementations |
| `src/agents/computer_use/session.py` | Multi-provider computer-use orchestrator |
| `src/orchestration/coordinator.py` | Multi-agent workflow coordinator |
| `src/desktop/` | Linux/X11 automation (uinput, ffmpeg, xrandr) |
| `src/mobile/` | Android ADB automation |
| `src/runtime/` | Execution context, caches, replay |
| `src/config/settings.py` | Pydantic settings, all env vars |
| `src/core/types.py` | Core types: TestPlan, TestCase, TestStep, ActionType |
| `src/monitoring/` | JSONL logging, HTML report generation |

### Computer-Use Providers

The system supports three AI providers for computer-use (configured via `HAINDY_CU_PROVIDER`):
- `openai` - OpenAI computer-use (default)
- `google` - Google Gemini via Vertex AI
- `anthropic` - Anthropic Claude computer-use

### Automation Backends

Configured via `HAINDY_AUTOMATION_BACKEND`:
- `desktop` - Linux/X11 via uinput; requires OS dependencies from `docs/RUNBOOK.md`
- `mobile_adb` - Android via ADB; requires `adb` in PATH
- unset - browser/planning mode only

### Technology Stack

- **Python 3.10+**, asyncio throughout
- **AI**: `openai`, `google-genai`, `anthropic` (computer-use APIs)
- **Automation**: `evdev`/uinput (desktop), ADB (mobile)
- **Data**: `pydantic` v2, `pillow`, `numpy`, `jinja2`, `jsonlines`
- **Dev**: `ruff`, `mypy`, `pytest`, `pytest-asyncio`

## Shared Contracts

When changing these areas, update all affected files together:

- **Backend names/aliases/defaults**: `src/runtime/environment.py`, `src/config/settings.py`, `.env.example`, `README.md`, tests
- **Env vars/cache paths/provider settings**: `src/config/settings.py`, `.env.example`, `README.md`, `docs/RUNBOOK.md`, tests
