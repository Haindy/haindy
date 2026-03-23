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

After the editable install, `haindy` should be available on `PATH` inside the
activated virtual environment. If you are not activating `.venv`, use
`.venv/bin/haindy`. Keep `python -m src.main ...` as a debugging or
development fallback.

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

**Credentials** (recommended: store in system keychain):

```bash
haindy --auth-set openai      # OpenAI API key
haindy --auth-set google      # Vertex project, location, API key
haindy --auth-set anthropic   # Anthropic API key
```

Alternatively, set `HAINDY_OPENAI_API_KEY`, `HAINDY_ANTHROPIC_API_KEY`, `HAINDY_VERTEX_API_KEY` as environment variables (highest priority, good for CI/CD).

**Settings file** (`~/.haindy/settings.json`): non-secret configuration. A `.haindy.json` in the working directory can override project-specific settings. See `haindy --config-show` for the full effective configuration.

Important env vars (still supported, override all other sources):

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
- `session new` launches the daemon independently and returns only after the socket is ready
- Desktop `session new --url ...` is not part of V1
- `explore` is V2 and intentionally absent

Useful commands:

```bash
haindy session new --desktop
haindy session list
haindy session status --session <SESSION_ID>
haindy act "tap the Login button" --session <SESSION_ID>
haindy test "complete checkout and verify the order summary" --session <SESSION_ID>
haindy session close --session <SESSION_ID>
```

## Troubleshooting

- If a tool-call command loses the daemon mid-run, inspect `~/.haindy/sessions/<id>/logs/daemon.log`
- If `session list` does not show a session, the daemon is dead or its socket was cleaned up as stale
- If a wrapper kills the entire process container or cgroup after `session new`, detached daemon survival is not possible there; rerun from a normal shell or keep the hidden `python -m src.main __tool_call_daemon ...` fallback alive in a long-lived PTY for debugging
- If desktop capture fails, verify `DISPLAY`, `ffmpeg`, and X11 permissions
- If desktop input fails, verify `xdotool` or `/dev/uinput` access
- If Android startup fails, run `adb devices` and confirm the package name or serial
