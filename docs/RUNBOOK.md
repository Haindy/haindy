# HAINDY Runbook

## Purpose

This runbook covers the host prerequisites and operational notes for HAINDY's two execution surfaces:

- Standard batch mode via `--plan` and `--context`
- Tool-call mode via `session`, `act`, `test`, and `explore`

## Quick Install

```bash
pip install haindy
haindy setup
```

`haindy setup` runs the interactive first-run wizard. It checks OS dependencies,
configures credentials, and installs skills for any AI CLIs it detects on the
host. Pass `--non-interactive` for CI environments.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[dev]"
haindy setup
```

After the editable install, `haindy` should be available on `PATH` inside the
activated virtual environment. If you are not activating `.venv`, use
`.venv/bin/haindy`. Keep `python -m haindy.main ...` as a debugging or
development fallback.

## macOS desktop prerequisites

Desktop automation on macOS uses `pynput` (input injection) and `mss`
(screenshot capture). Both are installed automatically via `requirements.lock`
on macOS. No additional system tools are required, but two macOS privacy
permissions must be granted before the first run.

### Required permissions

**Accessibility** — needed for keyboard injection via pynput.

1. Open System Settings -> Privacy & Security -> Accessibility
2. Add and enable the terminal emulator you use (e.g. Terminal, iTerm2) or the
   Python executable if running headlessly

**Screen Recording** — needed for `mss` to capture the screen.

1. Open System Settings -> Privacy & Security -> Screen Recording
2. Add and enable the same terminal or Python process

The agent prints a clear error and exits if either permission is missing.

### Retina displays

Screenshots are captured at native pixel resolution (e.g. 2560x1600 on a
13-inch MacBook Pro). The computer-use model sees this full-resolution image.
Mouse click coordinates returned by the model are automatically scaled from
pixel space to logical points before injection. No manual configuration is
needed; the scale factor is detected at session startup.

Resolution switching is not available on macOS. The system uses whatever
resolution is currently set in Display Preferences.

### Host expectations

- Start the target application before opening a tool-call session
- Maximize or position the target window so screenshots capture the app content
- Set `HAINDY_AUTOMATION_BACKEND=desktop`; on macOS the right driver is
  selected automatically

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `This process is not trusted` or keyboard input is ignored | Accessibility not granted | Add terminal to Accessibility in System Settings |
| Screenshots are black or `CGWindowListCreateImage` fails | Screen Recording not granted | Add terminal to Screen Recording in System Settings |
| Mouse clicks land in the wrong position | Scale factor mismatch | Run `haindy session status` and compare the screenshot size to the reported viewport |

---

## Linux/X11 desktop prerequisites

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

## iOS / idb prerequisites

macOS only. Required tools:

```bash
brew install idb-companion
pip install fb-idb
```

Host expectations:

- Real devices (iOS 16+): enable Developer Mode (Settings > Privacy & Security > Developer Mode)
- Real devices: connect via USB and accept the trust prompt on the device
- Simulators: boot via Xcode or `xcrun simctl boot <UDID>`
- Run `idb list-targets` to verify the device or simulator appears as `Booted` or `Connected`
- Set `HAINDY_IOS_DEFAULT_DEVICE_UDID` when multiple targets are connected

Run with `--ios` flag:

```bash
haindy --ios --plan <plan> --context <context>
```

Or set `HAINDY_AUTOMATION_BACKEND=mobile_ios` in the environment.

## Environment contract

**Credentials** (recommended: store in system keychain):

```bash
haindy auth login openai      # OpenAI API key
haindy auth login google      # Vertex project, location, API key
haindy auth login anthropic   # Anthropic API key
```

Alternatively, set `HAINDY_OPENAI_API_KEY`, `HAINDY_ANTHROPIC_API_KEY`, `HAINDY_VERTEX_API_KEY` as environment variables (highest priority, good for CI/CD).

**Settings file** (`~/.haindy/settings.json`): non-secret configuration. Provider models are stored per provider, with separate `computer_use_model` values for providers that support CU. Example:

```json
{
  "agent": { "provider": "openai" },
  "computer_use": { "provider": "google" },
  "openai": { "model": "gpt-5.4", "computer_use_model": "gpt-5.4" },
  "openai-codex": { "model": "gpt-5.4" },
  "google": {
    "model": "gemini-3-flash-preview",
    "computer_use_model": "gemini-3-flash-preview"
  },
  "anthropic": {
    "model": "claude-sonnet-4-6",
    "computer_use_model": "claude-sonnet-4-6"
  },
  "execution": {
    "actions_action_timeout_seconds": 600
  }
}
```

Useful commands:

```bash
haindy provider set openai
haindy provider set-computer-use google
haindy provider set-model google gemini-3-flash-preview
haindy provider set-model google gemini-3-flash-preview --computer-use
```

`openai-codex` is non-CU only and cannot be selected for computer-use or assigned a CU model.

Important env vars (still supported, override all other sources):

- `HAINDY_AUTOMATION_BACKEND=desktop|mobile_adb|mobile_ios`
- `HAINDY_HOME` for tool-call session state root
- `HAINDY_CU_PROVIDER=openai|google|anthropic`
- `HAINDY_ACTIONS_COMPUTER_TOOL_ACTION_TIMEOUT_SECONDS` for the per-action computer-use timeout budget
- `HAINDY_OPENAI_API_KEY` for OpenAI API-key auth and OpenAI computer use

Timeout settings use seconds across the runtime and configuration surface. Use `execution.actions_action_timeout_seconds` in `settings.json`; the older `execution.actions_action_timeout_ms` key is only accepted as a legacy read-time alias for older configs.

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
- Session variables are memory-only for the life of the daemon; secret values should be passed with `--value-file` when possible
- `session status` captures a fresh screenshot and counts as one action
- `session new` launches the daemon independently and returns only after the socket is ready
- Desktop `session new --url ...` is still deferred
- `test` and `explore` are async dispatch commands; poll `test-status` or `explore-status` for terminal results
- While a background task is active, `act`, `session status`, `test`, and `explore` return `session_busy`
- While a background task is active, `test-status`, `explore-status`, `screenshot`, `session set`, `session unset`, `session vars`, and `session close` remain available
- `session close` cancels an active background task before shutting the daemon down
- `session prune --older-than <days>` removes old dead session directories without touching live sessions

Useful commands:

```bash
haindy session new --desktop
haindy session list
haindy session status --session <SESSION_ID>
haindy act "tap the Login button" --session <SESSION_ID>
haindy test "complete checkout and verify the order summary" --session <SESSION_ID>
haindy test-status --session <SESSION_ID>
haindy explore "find the notification settings screen" --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>
haindy session close --session <SESSION_ID>
haindy session prune --older-than 7
```

When a background `test` stalls or times out, inspect the session-local forensic trail first:

- `~/.haindy/sessions/<SESSION_ID>/session.json` for `latest_background_run_id`, `latest_test_phase`, and `latest_test_action_artifact_path`
- `~/.haindy/sessions/<SESSION_ID>/action_artifacts/*.json` for per-step action, verification, and status-transition details
- `data/traces/<RUN_ID>.json` for the run trace tied to the stable background `run_id`
- `data/model_logs/model_calls.jsonl` for model-call history correlated by that same `run_id`

## Model-call artifacts

Durable model-call logs are written to `data/model_logs/model_calls.jsonl`.
Entries include both successful calls and non-rate-limit failed attempts across
standard mode and computer-use flows.

By policy, rate-limit-style retry noise is not written as durable failed-call
entries. This includes HTTP `429`, `resource_exhausted`, and equivalent provider
signals. Related runtime logger output and retry handling still occur normally.

When a logged call includes attached screenshots, the image files are stored
under `data/model_logs/screenshots/`.

## Troubleshooting

Run `haindy doctor` first. It checks all required OS dependencies, credentials,
and permissions, and prints a clear pass/fail summary for each item.

- If a tool-call command loses the daemon mid-run, inspect `~/.haindy/sessions/<id>/logs/daemon.log`
- If `session list` does not show a session, the daemon is dead or its socket was cleaned up as stale
- If a wrapper kills the entire process container or cgroup after `session new`, detached daemon survival is not possible there; rerun from a normal shell or keep the hidden `python -m haindy.main __tool_call_daemon ...` fallback alive in a long-lived PTY for debugging
- If desktop capture fails, verify `DISPLAY`, `ffmpeg`, and X11 permissions
- If desktop input fails, verify `xdotool` or `/dev/uinput` access
- If Android startup fails, run `adb devices` and confirm the package name or serial
- If iOS startup fails, run `idb list-targets` and confirm the UDID and state is `Booted`

## Windows setup

### Prerequisites

- Python 3.11+ (install from python.org or `winget install Python.Python.3.11`)
- ADB (optional, for Android targets): `winget install Google.PlatformTools`
- Windows Terminal recommended for Rich/ANSI rendering; legacy `cmd.exe` also works

### Long paths

HAINDY stores session data under paths like `~/.haindy/sessions/<uuid>/screenshots/`.
Enable long path support to avoid `ERROR_FILENAME_EXCED_RANGE`:

```powershell
# requires an elevated PowerShell prompt
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

Or open `gpedit.msc` > Computer Configuration > Administrative Templates >
System > Filesystem > Enable Win32 long paths.

### Virtual environment activation

Windows uses `.venv\Scripts\` instead of `.venv/bin/`:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.lock
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\haindy setup
```

### Credentials (Windows Credential Manager)

`haindy auth login openai` stores the API key in Windows Credential Manager via
the `keyring` library. Verify with `haindy auth status` and `haindy doctor`.

### UAC elevation caveat

pynput uses `SendInput` to inject keyboard and mouse events. On Windows, `SendInput`
cannot deliver events to windows running under a higher integrity level (elevated
UAC). If HAINDY is run from a normal shell, input injection into UAC-elevated
target windows will silently fail. Work around this by running HAINDY itself
elevated, or by keeping the target application at the same integrity level.

### Tool-call mode

Tool-call mode is available on Windows. HAINDY uses the same public commands
(`session`, `act`, `test`, `explore`, and `screenshot`) as on Linux and macOS,
but the daemon transport is different under the hood: Windows uses a localhost
TCP port recorded in `daemon.port` instead of a Unix domain socket.

If a Windows tool-call session fails to start, inspect
`~/.haindy/sessions/<id>/logs/daemon.log` first and confirm the sibling
`daemon.port` file was created.
