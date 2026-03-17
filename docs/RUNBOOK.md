# HAINDY Runbook

## Purpose

This runbook covers the host prerequisites and operational notes for HAINDY's two execution surfaces:

- Standard batch mode via `--plan` and `--context`
- Tool-call mode via `session`, `act`, and `test`

## Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[dev]"
```

## Linux/X11 desktop prerequisites

Desktop automation is Linux/X11-only today.

Required runtime tools:

- `ffmpeg` for primary screen capture
- `xdotool` for the fallback input backend
- `xclip` for clipboard reads and writes

Optional but useful:

- ImageMagick `import` as a fallback screenshot backend
- `/dev/uinput` access for smoother virtual input when available

Host expectations:

- Run inside an X11 session with `DISPLAY` available, or set `HAINDY_DESKTOP_DISPLAY`
- Start the target desktop app or web app before opening a tool-call desktop session
- Maximize the target window when possible so screenshots stay focused on the app

## Android / ADB prerequisites

Required runtime tool:

- `adb`

Host expectations:

- `adb devices` must show the target device or emulator in `device` state
- Authorize the host on the device if prompted
- Pass `--android-serial` when multiple devices are connected
- Pass `--android-app` when the session should launch a package on startup

## Environment contract

Important env vars:

- `HAINDY_AUTOMATION_BACKEND=desktop|mobile_adb`
- `HAINDY_HOME` for tool-call session state root
- `HAINDY_CU_PROVIDER=openai|google|anthropic`
- `HAINDY_OPENAI_API_KEY` for OpenAI API-key auth and OpenAI computer use

Tool-call mode does not introduce a separate backend env var. Use `HAINDY_AUTOMATION_BACKEND` or explicit `--desktop` / `--android`.

## Tool-call mode operational notes

Session layout:

```text
~/.haindy/
  sessions/<session-id>/
    daemon.sock
    daemon.pid
    session.json
    screenshots/
    logs/daemon.log
```

Operational rules:

- Every tool-call command emits one JSON object to stdout
- Daemon logs must go to `logs/daemon.log` or stderr when `--debug` is set
- Session variables are memory-only in V1; secret values should be passed with `--value-file` when possible
- `session status` captures a fresh screenshot and counts as one action
- Desktop `session new --url ...` is not part of V1
- `explore` is V2 and intentionally absent

Useful commands:

```bash
.venv/bin/python -m src.main session new --desktop
.venv/bin/python -m src.main session list
.venv/bin/python -m src.main session status --session <SESSION_ID>
.venv/bin/python -m src.main act "tap the Login button" --session <SESSION_ID>
.venv/bin/python -m src.main test "complete checkout and verify the order summary" --session <SESSION_ID>
.venv/bin/python -m src.main session close --session <SESSION_ID>
```

## Troubleshooting

- If a tool-call command loses the daemon mid-run, inspect `~/.haindy/sessions/<id>/logs/daemon.log`
- If `session list` does not show a session, the daemon is dead or its socket was cleaned up as stale
- If desktop capture fails, verify `DISPLAY`, `ffmpeg`, and X11 permissions
- If desktop input fails, verify `xdotool` or `/dev/uinput` access
- If Android startup fails, run `adb devices` and confirm the package name or serial
