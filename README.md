# HAINDY

Desktop-first autonomous testing agent that turns requirements into
executable test runs.

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

### Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[dev]"
```

### macOS

- Base dependency installation is supported.
- Developer checks are supported.
- The Linux desktop automation backend is not supported on macOS yet.
- Use macOS for development, tests, planning, and mobile ADB flows.

### Linux

- Base dependency installation is supported.
- Desktop automation is supported on Linux/X11 after installing the extra
  runtime tools from [docs/RUNBOOK.md](/Users/fkeegan/src/haindy/haindy/docs/RUNBOOK.md).
- Use Linux when you need the `desktop` automation backend.

### Install Backend Prerequisites

- No Playwright browser runtime installation is required for the current codebase.
- Linux desktop runs: install the Linux/X11 desktop automation dependencies from
  [docs/RUNBOOK.md](/Users/fkeegan/src/haindy/haindy/docs/RUNBOOK.md).
- Mobile ADB runs: ensure `adb` is installed and your Android device or emulator
  is reachable.

### Configure Environment

```bash
cp .env.example .env
```

Minimum required settings:

- `HAINDY_OPENAI_API_KEY` for OpenAI API-key auth and for OpenAI computer-use
- `HAINDY_CU_PROVIDER=openai`, `HAINDY_CU_PROVIDER=google`, or
  `HAINDY_CU_PROVIDER=anthropic`

OpenAI auth modes:

- Default non-CU OpenAI auth uses `HAINDY_OPENAI_API_KEY`.
- `--codex-auth login` stores a local encrypted Codex OAuth session and makes
  non-CU OpenAI requests use OAuth instead of the API key.
- `--codex-auth logout` removes the stored OAuth session and reverts non-CU
  OpenAI requests to API-key auth.
- `--codex-auth status` shows the active non-CU OpenAI auth mode.
- Stored Codex OAuth credentials live outside the repo in the user state
  directory.

Provider-specific:

- OpenAI computer-use: `HAINDY_COMPUTER_USE_MODEL` (default `gpt-5.4`) and
  `HAINDY_OPENAI_API_KEY`. Codex OAuth does not apply to CU runs.
- Google computer-use: `HAINDY_GOOGLE_CU_MODEL` and Vertex credentials/settings
  (see `.env.example`)
- Anthropic computer-use: `HAINDY_ANTHROPIC_API_KEY`, optional
  `HAINDY_ANTHROPIC_CU_MODEL` (default `claude-sonnet-4-6`), optional
  `HAINDY_ANTHROPIC_CU_MAX_TOKENS` (default `16384`)

Platform notes:

- macOS: do not rely on `HAINDY_AUTOMATION_BACKEND=desktop`; that backend is
  Linux/X11-only today.
- Linux: set `HAINDY_AUTOMATION_BACKEND=desktop` when running desktop automation.

## Run

Plan and context files are both required:

```bash
.venv/bin/python -m src.main \
  --plan test_scenarios/wikipedia_search_simple.txt \
  --context test_scenarios/wikipedia_search_simple.txt
```

Mobile ADB backend (hard override):

```bash
.venv/bin/python -m src.main \
  --mobile \
  --plan <plan_file> \
  --context <mobile_context_file>
```

For `--mobile` runs, context should provide either:

- `adb_serial` + `app_package` (optional `app_activity`), or
- explicit `adb_commands` that discover/select device and open the target app.

Desktop backend note:

- Linux/X11 only at the moment. On macOS, use developer/test flows or `--mobile`.

Optional debug logging:

```bash
.venv/bin/python -m src.main \
  --plan <plan_file> \
  --context <context_file> \
  --debug
```

Codex OAuth login:

```bash
.venv/bin/python -m src.main --codex-auth login
.venv/bin/python -m src.main --codex-auth status
.venv/bin/python -m src.main --codex-auth logout
```

## Developer checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/mypy src
.venv/bin/pytest
```
