# HAINDY

Desktop-first autonomous testing agent with two operating modes:

- Standard mode: batch planning and execution from `--plan` plus `--context`
- Tool-call mode: session-based JSON CLI for coding agents

## Quick project map

- `src/main.py`: shared CLI entrypoint
- `src/tool_call_mode/`: tool-call CLI, daemon, IPC, runtime, and session state helpers
- `src/agents/`: planner, runner, action, and situational agents
- `src/desktop/`: Linux/X11 desktop automation
- `src/mobile/`: Android ADB automation
- `src/config/settings.py`: env-backed runtime configuration
- `docs/design/tool-call-mode/`: tool-call mode design docs
- `docs/RUNBOOK.md`: setup and operational notes
- `.agents/skills/haindy/SKILL.md`: bundled skill for agent workflows

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[dev]"
```

### 3. Install backend prerequisites

- Linux/X11 desktop automation: install the runtime tools in [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Android automation: ensure `adb` is installed and the target device or emulator is reachable
- macOS is fine for development and tests, but the `desktop` backend is Linux/X11-only today

### 4. Configure environment

```bash
cp .env.example .env
```

Important settings:

- `HAINDY_OPENAI_API_KEY` for OpenAI API-key auth and OpenAI computer use
- `HAINDY_CU_PROVIDER=openai|google|anthropic`
- `HAINDY_AUTOMATION_BACKEND=desktop|mobile_adb` to set the default backend
- `HAINDY_HOME` to override the tool-call session home (default: `~/.haindy`)

OpenAI auth modes:

- Default non-CU OpenAI auth uses `HAINDY_OPENAI_API_KEY`
- `--codex-auth login` stores a local encrypted Codex OAuth session for non-CU OpenAI requests
- `--codex-auth status` shows the active non-CU OpenAI auth mode
- `--codex-auth logout` clears the stored OAuth session

## Standard mode

Plan and context files are both required:

```bash
.venv/bin/python -m src.main \
  --plan test_scenarios/wikipedia_search_simple.txt \
  --context test_scenarios/wikipedia_search_simple.txt
```

Force the mobile backend:

```bash
.venv/bin/python -m src.main \
  --mobile \
  --plan <plan_file> \
  --context <context_file>
```

Optional debug logging:

```bash
.venv/bin/python -m src.main \
  --plan <plan_file> \
  --context <context_file> \
  --debug
```

## Tool-call mode

Tool-call mode is a separate runtime for coding agents. Every command prints exactly one JSON object to stdout. Session state, screenshots, and daemon logs live under `~/.haindy/sessions/<id>/` unless `HAINDY_HOME` overrides the root.

Start a session:

```bash
.venv/bin/python -m src.main session new --desktop
.venv/bin/python -m src.main session new --android --android-serial emulator-5554
```

Use the returned `session_id` explicitly:

```bash
.venv/bin/python -m src.main session status --session <SESSION_ID>
.venv/bin/python -m src.main act "tap the Login button" --session <SESSION_ID>
.venv/bin/python -m src.main test "sign in and verify the dashboard appears" --session <SESSION_ID>
.venv/bin/python -m src.main session set USERNAME alice@example.com --session <SESSION_ID>
.venv/bin/python -m src.main session close --session <SESSION_ID>
```

Tool-call mode guidance:

- Prefer `test` over `act` whenever outcome validation matters
- Use `session set --value-file ... --secret` for sensitive values
- Desktop sessions assume the target site or app is already running
- Desktop `session new --url ...` is intentionally deferred from V1
- `explore` is V2 work and is not part of the current CLI surface

## Codex OAuth

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
