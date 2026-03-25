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

# First-run setup wizard
haindy setup

# Check system dependencies and configuration
haindy doctor

# Non-interactive setup (for CI)
haindy setup --non-interactive

# Configure credentials (stored in system keychain)
haindy --auth login openai    # or google / anthropic / openai-codex
haindy --auth status          # verify

# Optional: create ~/.haindy/settings.json for persistent non-secret settings
# See .agents/skills/haindy-setup/SKILL.md for full setup guidance
```

For CI/CD or if you prefer a flat file, copy `.env.example` to `.env` and fill
in the values. Environment variables take priority over all other sources.

### Running
```bash
# Standard run (both flags required)
haindy --plan <requirements_file> --context <context_file>

# Mobile ADB backend
haindy --mobile --plan <plan> --context <context>

# Debug mode
haindy --plan <plan> --context <context> --debug

# OAuth auth management
haindy --auth login openai-codex
haindy --auth status
haindy --auth clear openai-codex

# API connectivity test
haindy --test-api
```

Keep `python -m haindy.main ...` as a local/dev fallback when you intentionally
need the module entrypoint.

### Testing
```bash
.venv/bin/pytest
.venv/bin/pytest -m "not slow"
.venv/bin/pytest --cov=haindy
```

### Code Quality
```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/mypy haindy
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
| `haindy/main.py` | CLI entrypoint |
| `haindy/agents/` | All agent implementations |
| `haindy/agents/computer_use/session.py` | Multi-provider computer-use orchestrator |
| `haindy/orchestration/coordinator.py` | Multi-agent workflow coordinator |
| `haindy/desktop/` | Linux/X11 automation (uinput, ffmpeg, xrandr) |
| `haindy/mobile/` | Android ADB automation |
| `haindy/runtime/` | Execution context, caches, replay |
| `haindy/config/settings.py` | Pydantic settings, all env vars |
| `haindy/core/types.py` | Core types: TestPlan, TestCase, TestStep, ActionType |
| `haindy/monitoring/` | JSONL logging, HTML report generation |

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

- **Backend names/aliases/defaults**: `haindy/runtime/environment.py`, `haindy/config/settings.py`, `.env.example`, `README.md`, tests
- **Env vars/cache paths/provider settings**: `haindy/config/settings.py`, `haindy/config/settings_file.py` (`_JSON_TO_FIELD`), `.env.example`, `README.md`, `docs/RUNBOOK.md`, tests
- **Adding a new settings field**: update `Settings` in `haindy/config/settings.py`, add to `SETTINGS_ENV_VARS`, add to `_JSON_TO_FIELD` in `haindy/config/settings_file.py`, add to `.env.example`
- **Adding a new secret/API key**: update `_SECRET_FIELD_TO_PROVIDER` in `haindy/config/settings.py`, update `_PROVIDER_TO_ACCOUNT` in `haindy/auth/credentials.py`, update `haindy/cli/auth_commands.py`
