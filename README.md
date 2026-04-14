# HAINDY

[![CI](https://github.com/Haindy/haindy/actions/workflows/ci.yml/badge.svg)](https://github.com/Haindy/haindy/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/haindy.svg)](https://pypi.org/project/haindy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Computer-use for coding agents. HAINDY gives AI coding tools (Claude Code, Codex CLI, OpenCode, and others) the ability to see the screen, click, type, and test real applications across desktop, Android, and iOS.

```bash
pip install haindy
haindy setup
```

## Quickstart

Start a session, read the returned `session_id`, and pass it explicitly to the next commands:

```bash
haindy session new --desktop
haindy screenshot --session <SESSION_ID>
haindy act "click the Login button" --session <SESSION_ID>
haindy session close --session <SESSION_ID>
```

## What it does

Your coding agent calls HAINDY to interact with a live UI. Every command returns structured JSON.

```bash
haindy session new --desktop                                       # start a session
haindy act "click the Login button" --session <ID>                 # execute an action
haindy test "sign in and verify the dashboard loads" --session <ID> # dispatch a multi-step test
haindy test-status --session <ID>                                  # poll test progress / result
haindy explore "find the notification settings screen" --session <ID> # dispatch exploration
haindy explore-status --session <ID>                               # poll explore progress / result
haindy screenshot --session <ID>                                   # capture current state
haindy session close --session <ID>                                # clean up
haindy session prune --older-than 7                                # remove old dead sessions
```

A typical response looks like this:

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

```bash
haindy session new --desktop
haindy session new --android --android-serial emulator-5554
haindy session new --ios --ios-udid <UDID>
```

## Agent integration

HAINDY ships with bundled skills that `haindy setup` installs automatically for detected AI CLIs. Once installed, your coding agent can discover and use HAINDY directly.

Supported CLIs: Claude Code, Codex CLI, OpenCode.

### act vs test vs explore

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

## Batch mode

HAINDY also includes a pipeline of specialized AI agents that can plan and execute tests autonomously from a requirements file -- no coding agent required:

```
Requirements ──> Scope Triage ──> Test Planner ──> Situational Agent ──> Test Runner ──> Report
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
haindy auth login openai      # stored in system keychain
haindy auth login google
haindy auth login anthropic
haindy auth status             # verify
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
