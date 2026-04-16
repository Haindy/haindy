# HAINDY

[![CI](https://github.com/Haindy/haindy/actions/workflows/ci.yml/badge.svg)](https://github.com/Haindy/haindy/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/haindy.svg)](https://pypi.org/project/haindy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Give coding agents computer use.

HAINDY lets coding tools like Claude Code, Codex CLI, and OpenCode interact with real desktop and mobile apps by seeing the screen, clicking, typing, scrolling, and validating flows. Use it when your agent needs to work with a real UI instead of a DOM or selector tree.

```bash
pip install haindy
haindy setup
```

After setup, open your coding agent and use the `haindy` skill against a live app.

## Agent integration

HAINDY ships with bundled skills that `haindy setup` installs automatically for detected AI CLIs.

Supported CLIs:

- Claude Code
- Codex CLI
- OpenCode

If one of those CLIs is installed, `haindy setup` copies the HAINDY skill into the agent's skill directory and points the agent at the setup flow. For other coding agents, you can still use HAINDY by prompting them directly with the examples below.

## Try it now

Open your coding agent and use the `haindy` skill against a running desktop app, Android emulator or device, or iOS simulator or device.

If you are running an app locally, try prompts like:

- "Use HAINDY to do exploratory testing on my app."
- "Use HAINDY to test creating a new account."
- "Use HAINDY to check whether the login flow works end to end."
- "Use HAINDY to explore the settings screen and report whether notifications can be toggled."

Your agent will start the session, interact with the UI, and return screenshots and structured results.

## CLI usage

HAINDY can also be driven directly from the command line when you want explicit command-by-command control.

Start a session, read the returned `session_id`, and pass it explicitly to later commands:

```bash
haindy session new --desktop
haindy screenshot --session <SESSION_ID>
haindy act "click the Login button" --session <SESSION_ID>
haindy session close --session <SESSION_ID>
```

For mobile:

```bash
haindy session new --android --android-serial emulator-5554
haindy session new --ios --ios-udid <UDID>
```

For session hygiene:

```bash
haindy session prune --older-than 7
```

Every command returns structured JSON. A typical response looks like this:

```json
{
  "session_id": "8f4d2c1e-7c2d-4d92-a0bc-3d0a9c6c1b5e",
  "run_id": null,
  "command": "screenshot",
  "status": "success",
  "response": "Screenshot captured.",
  "screenshot_path": "/absolute/path/to/screenshot.png",
  "meta": {
    "exit_reason": "completed",
    "duration_ms": 0,
    "actions_taken": 0
  }
}
```

Under the hood, each action goes through a computer-use AI provider (OpenAI, Google Gemini, or Anthropic Claude) that takes a screenshot, reasons about the UI, and performs real OS-level input -- mouse, keyboard, scroll -- against the actual application. No DOM hooks, no selectors, no browser automation.

### Supported platforms

| Platform | Automation method |
|----------|------------------|
| Linux/X11 | uinput + xdotool + ffmpeg |
| macOS | pynput + mss |
| Android | ADB |
| iOS | idb |

## act vs test vs explore

- **`act`** -- execute a single action ("click the submit button", "type hello into the search field")
- **`test`** -- dispatch a multi-step scenario with outcome validation, then poll `test-status`
- **`explore`** -- dispatch an open-ended goal, then poll `explore-status`

Use `test` when the scenario is backed by written requirements, a test plan, wireframes, or other explicit documentation and you want structured execution plus validation. Use `explore` when the goal is clear but the path is not, or when you are working from product knowledge rather than supporting docs. Use `act` when you want tight step-by-step control and to inspect the screen after each command.

### Session variables

Store values your agent can reference across commands:

```bash
haindy session set USERNAME alice@example.com --session <ID>
haindy session set PASSWORD --value-file credentials.txt --secret --session <ID>
haindy session vars --session <ID>
```

## Run tests from requirements

HAINDY also includes a pipeline of specialized AI agents that can plan and execute tests autonomously from a requirements file.

```text
Requirements -> Scope Triage -> Test Planner -> Situational Agent -> Test Runner -> Report
```

```bash
haindy run --plan requirements.txt --context context.txt
haindy run --mobile --plan requirements.txt --context context.txt    # Android
haindy run --ios --plan requirements.txt --context context.txt       # iOS
```

This produces an HTML report with screenshots, pass/fail results, and a JSONL execution log.

## Configuration

### Credentials

```bash
haindy auth login openai        # stored in system keychain
haindy auth login openai-codex  # OAuth-based login
haindy auth login google
haindy auth login anthropic
haindy auth status              # verify
```

### Providers

HAINDY uses two providers independently: one for planning/analysis, one for computer-use actions.

```bash
haindy provider set openai                   # planning/analysis
haindy provider set-computer-use google      # computer-use
```

### Settings file

Create `~/.haindy/settings.json` for persistent non-secret configuration:

```json
{
  "agent": { "provider": "openai" },
  "computer_use": { "provider": "google" },
  "openai": { "model": "gpt-5.4", "computer_use_model": "gpt-5.4" },
  "google": { "model": "gemini-3-flash-preview", "computer_use_model": "gemini-3-flash-preview" },
  "anthropic": { "model": "claude-sonnet-4-6", "computer_use_model": "claude-sonnet-4-6" },
  "execution": {
    "automation_backend": "desktop",
    "actions_action_timeout_seconds": 600
  },
  "logging": { "level": "INFO" }
}
```

Environment variables override all other sources. Timeout settings use seconds. In `settings.json`, use `execution.actions_action_timeout_seconds`; the older `execution.actions_action_timeout_ms` key is only accepted as a legacy read-time alias. See [`.env.example`](.env.example) for the full list.

## Platform prerequisites

| Platform | Requirements |
|----------|-------------|
| Linux/X11 | `ffmpeg`, `xdotool`, `xclip`, `/dev/uinput` access |
| macOS | Grant Accessibility + Screen Recording to your terminal (System Settings > Privacy & Security) |
| Android | `adb` installed, device/emulator reachable |
| iOS (macOS) | `brew install idb-companion`, device paired |

`haindy doctor` checks all of these for you. See [docs/RUNBOOK.md](docs/RUNBOOK.md) for detailed setup.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e ".[dev]"
```

```bash
ruff check .          # lint
ruff format --check . # format check
mypy haindy           # type check
pytest                # tests
```

## Architecture

| Directory | Purpose |
|-----------|---------|
| `haindy/tool_call_mode/` | Tool-call CLI, daemon, IPC, session state |
| `haindy/agents/computer_use/` | Multi-provider computer-use session orchestrator |
| `haindy/agents/` | Scope triage, test planner, situational, action, and test runner agents |
| `haindy/desktop/` | Linux/X11 automation (uinput, xdotool, ffmpeg) |
| `haindy/macos/` | macOS automation (pynput, mss) |
| `haindy/mobile/` | Android (ADB) and iOS (idb) automation |
| `haindy/config/` | Settings, env vars, settings file loader |
| `haindy/orchestration/` | Multi-agent workflow coordination |
| `haindy/monitoring/` | JSONL logging, HTML report generation |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and how to submit changes.

## License

[MIT](LICENSE)
