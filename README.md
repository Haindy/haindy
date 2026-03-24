# HAINDY

Desktop-first autonomous testing agent with two operating modes:

- Standard mode: batch planning and execution from `--plan` plus `--context`
- Tool-call mode: session-based JSON CLI for coding agents

## Quick project map

- `haindy/main.py`: shared CLI entrypoint
- `haindy/tool_call_mode/`: tool-call CLI, daemon, IPC, runtime, and session state helpers
- `haindy/agents/`: planner, runner, action, and situational agents
- `haindy/desktop/`: Linux/X11 desktop automation
- `haindy/macos/`: macOS desktop automation (pynput + mss)
- `haindy/mobile/`: Android ADB and iOS idb automation
- `haindy/config/settings.py`: env-backed runtime configuration
- `docs/design/tool-call-mode/`: tool-call mode design docs
- `docs/RUNBOOK.md`: setup and operational notes
- `.agents/skills/haindy/SKILL.md`: bundled skill for agent workflows

## Installation

```bash
pip install haindy
haindy setup
```

`haindy setup` runs the interactive first-run wizard: it checks dependencies,
configures credentials, and installs skills for any AI CLIs it detects.

Run `haindy doctor` at any time to verify your environment.

## Development Setup

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

After the editable install, activating `.venv` exposes `haindy` on `PATH`.
If you prefer not to activate the virtual environment, use `.venv/bin/haindy`.
`python -m haindy.main ...` remains available as an internal/dev fallback.

### 3. Install backend prerequisites

- Linux/X11 desktop automation: install the runtime tools in [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Android automation: ensure `adb` is installed and the target device or emulator is reachable
- iOS automation (macOS only): `brew install idb-companion` — see [docs/RUNBOOK.md](docs/RUNBOOK.md) for full prerequisites
- macOS desktop automation: grant Accessibility and Screen Recording permissions to your terminal app (System Settings > Privacy & Security) — see [docs/RUNBOOK.md](docs/RUNBOOK.md) for details

### 4. Configure credentials and settings

**Recommended: interactive setup**

```bash
haindy --auth-set openai      # prompts for API key; stored in system keychain
haindy --auth-set google      # prompts for Vertex project, location, and API key
haindy --auth-set anthropic   # prompts for API key; stored in system keychain
haindy --auth-status          # shows which providers have credentials configured
```

**Settings file** (`~/.haindy/settings.json`): create this for persistent non-secret configuration:

```json
{
  "computer_use": { "provider": "google" },
  "execution": { "automation_backend": "desktop" },
  "logging": { "level": "INFO" }
}
```




**Migrating from .env**: if you have an existing `.env` file:

```bash
haindy --config-migrate          # reads .env, splits settings to settings.json and keys to keychain
haindy --config-show             # verify the effective configuration
```

**CI/CD**: environment variables still work and take the highest priority. Copy `.env.example` to `.env` and fill in values, or export them directly.

OpenAI auth modes:

- Default non-CU OpenAI auth uses the stored/env `HAINDY_OPENAI_API_KEY`
- `--codex-auth login` stores a local encrypted Codex OAuth session for non-CU OpenAI requests
- `--codex-auth status` shows the active non-CU OpenAI auth mode
- `--codex-auth logout` clears the stored OAuth session

## Standard mode

Plan and context files are both required:

```bash
haindy \
  --plan test_scenarios/wikipedia_search_simple.txt \
  --context test_scenarios/wikipedia_search_simple.txt
```

Force the Android mobile backend:

```bash
haindy \
  --mobile \
  --plan <plan_file> \
  --context <context_file>
```

Force the iOS backend:

```bash
haindy \
  --ios \
  --plan <plan_file> \
  --context <context_file>
```

Optional debug logging:

```bash
haindy \
  --plan <plan_file> \
  --context <context_file> \
  --debug
```

## Tool-call mode

Tool-call mode is a separate runtime for coding agents. Every command prints exactly one JSON object to stdout. Session state, screenshots, and daemon logs live under `~/.haindy/sessions/<id>/` unless `HAINDY_HOME` overrides the root.

Start a session:

```bash
haindy session new --desktop
haindy session new --android --android-serial emulator-5554
haindy session new --ios [--ios-udid <UDID>] [--ios-app <BUNDLE_ID>]
```

Use the returned `session_id` explicitly:

```bash
haindy session status --session <SESSION_ID>
haindy act "tap the Login button" --session <SESSION_ID>
haindy test "sign in and verify the dashboard appears" --session <SESSION_ID>
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session close --session <SESSION_ID>
```

Tool-call mode guidance:

- Prefer `test` over `act` whenever outcome validation matters
- Use `session set --value-file ... --secret` for sensitive values
- Desktop sessions assume the target site or app is already running
- Desktop `session new --url ...` is intentionally deferred from V1
- `explore` is V2 work and is not part of the current CLI surface
- `session new` now launches the daemon independently, so later commands should keep working after the original CLI process exits
- If a wrapper still kills the entire process container or cgroup, fall back to a long-lived shell and run `python -m haindy.main __tool_call_daemon ...` for debugging

## Codex OAuth

```bash
haindy --codex-auth login
haindy --codex-auth status
haindy --codex-auth logout
```

## Developer checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/mypy haindy
.venv/bin/pytest
```
