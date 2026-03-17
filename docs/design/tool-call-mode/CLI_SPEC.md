# Haindy Agentic Mode - CLI Specification

## Invocation

```
haindy <subcommand> [options]
```

All tool call mode subcommands are grouped under two top-level subcommands:
- `session` - manage sessions and session variables
- Direct action subcommands: `act`, `test`

The active session is resolved in order:
1. Explicit `--session <id>` flag on the command
2. `HAINDY_SESSION` environment variable
3. Error if neither is set

---

## Global Flags

| Flag | Default | Description |
|---|---|---|
| `--session <id>` | `$HAINDY_SESSION` | Session ID override. Ignored for `session new`. |
| `--json` | Enabled by default in tool call mode | Emit JSON on stdout. Always on for tool call mode; flag exists for scripting clarity. |
| `--debug` | off | Emit verbose daemon logs to stderr (does not affect stdout JSON). |

---

## `haindy session` Subcommands

### `haindy session new`

Start a new session. Spawns the daemon process, initializes the device connection, and returns the session ID.

```
haindy session new [--android | --desktop] [options]
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--android` | | Use Android ADB backend. Mutually exclusive with `--desktop`. |
| `--android-serial <serial>` | | Target a specific Android device or emulator by ADB serial (e.g. `emulator-5554`). Optional; uses the only connected device if omitted. |
| `--android-app <package>` | | Android package name to launch on session start (e.g. `com.example.app`). Optional. |
| `--desktop` | | Use the desktop backend (computer use: OS screen capture + AI-driven input). Mutually exclusive with `--android`. Default if neither specified and no env config. |
| `--url <url>` | | URL to open on session start (desktop only). Optional. |
| `--idle-timeout <seconds>` | 1800 | Kill daemon after this many seconds without a command. |

**Stdout on success:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "Session started with Android ADB backend. Device found: Pixel 7 (emulator-5554).",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_001.png",
  "meta": {"exit_reason": "completed", "duration_ms": 1823, "actions_taken": 0}
}
```

**Recommended usage in a skill:**

```bash
export HAINDY_SESSION=$(haindy session new --android | jq -r .session_id)
```

---

### `haindy session close`

Terminate the session daemon and release the device connection.

```
haindy session close [--session <id>]
```

**Stdout:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "Session closed. 14 steps were executed during this session.",
  "screenshot_path": null,
  "meta": {"exit_reason": "completed", "duration_ms": 43, "actions_taken": 0}
}
```

---

### `haindy session list`

List all active sessions on this machine.

```
haindy session list
```

**Stdout:**

```json
{
  "session_id": null,
  "command": "session",
  "status": "success",
  "response": "2 active sessions found.",
  "screenshot_path": null,
  "meta": {"exit_reason": "completed", "duration_ms": 12, "actions_taken": 0},
  "sessions": [
    {
      "session_id": "a3f9c2d1-...",
      "backend": "android",
      "created_at": "2026-03-13T14:22:01Z",
      "steps_executed": 7,
      "idle_seconds": 42
    }
  ]
}
```

---

### `haindy session status`

Return the current state of an active session, including the latest screenshot.

```
haindy session status [--session <id>]
```

**Stdout:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "Session is active. Last command was 'act' 23 seconds ago. Device is on the home screen.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_007.png",
  "meta": {"exit_reason": "completed", "duration_ms": 891, "actions_taken": 1}
}
```

---

### `haindy session set`

Store a named variable in the session. The variable can be referenced as `$NAME` in any subsequent command string. The daemon interpolates all `$NAME` tokens before passing the instruction to agents.

```
haindy session set <NAME> <VALUE> [--secret] [--session <id>]
```

**Flags:**

| Flag | Description |
|---|---|
| `--secret` | Mark the variable as secret. Secret values are stored in daemon memory only (not written to session.json or logs). Any `response` field that would echo the value back replaces it with `[redacted]`. |

**Examples:**

```bash
haindy session set USERNAME alice@example.com
haindy session set PASSWORD hunter2 --secret
haindy session set BASE_URL https://staging.example.com

# Then use in commands:
haindy act "type '$USERNAME' into the email field"
haindy test "sign in with $USERNAME and $PASSWORD and verify the dashboard appears"
haindy act "navigate to $BASE_URL/login"
```

**Stdout:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "Variable PASSWORD set (secret).",
  "screenshot_path": null,
  "meta": {"exit_reason": "completed", "duration_ms": 3, "actions_taken": 0}
}
```

---

### `haindy session unset`

Remove a named variable from the session.

```
haindy session unset <NAME> [--session <id>]
```

---

### `haindy session vars`

List all variable names defined in the session. Secret variable values are not shown.

```
haindy session vars [--session <id>]
```

**Stdout:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "3 variables defined: USERNAME, PASSWORD (secret), BASE_URL.",
  "screenshot_path": null,
  "meta": {"exit_reason": "completed", "duration_ms": 2, "actions_taken": 0},
  "vars": {
    "USERNAME": "alice@example.com",
    "PASSWORD": "[secret]",
    "BASE_URL": "https://staging.example.com"
  }
}
```

---

## Action Subcommands

All action subcommands share this behavior:
- Require an active session (via `HAINDY_SESSION` or `--session`).
- Return a single JSON object on stdout (the standard envelope, see OVERVIEW.md).
- Exit 0 on `success`, exit 1 on `failure` or `error`.
- Always capture a screenshot after the command completes and include `screenshot_path`.

---

### `haindy act`

Execute a single direct interaction on the device. No outcome validation is performed. Maps directly to the Action Agent.

```
haindy act "<instruction>" [--session <id>]
```

**Argument:**

`<instruction>` - A natural language description of the single action to perform.

**Examples:**

```bash
haindy act "tap the Login button"
haindy act "type 'hunter2' into the password field"
haindy act "scroll down until the Terms section is visible"
haindy act "press the back button"
```

**Stdout on success:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "act",
  "status": "success",
  "response": "Tapped the Login button. The button had the text 'Log In' and was located in the center of the screen.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_003.png",
  "meta": {"exit_reason": "completed", "duration_ms": 1243, "actions_taken": 1}
}
```

**Stdout on failure:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "act",
  "status": "failure",
  "response": "Could not find a Login button on the current screen. The screen shows an empty loading state with a spinner. The app may not have finished loading.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_003.png",
  "meta": {"exit_reason": "element_not_found", "duration_ms": 987, "actions_taken": 0}
}
```

**When to use:** When the coding agent knows the exact interaction and does not need Haindy to verify an outcome. For example, tapping a clearly visible element or typing into a focused field. Use `test` for anything that requires outcome validation.

---

### `haindy test`

Run a full test scenario. The Test Planner generates a structured plan from the description, then the Test Runner executes each step and validates outcomes. Maps to Test Planner + Test Runner.

```
haindy test "<scenario>" [--session <id>] [--max-steps <n>]
```

**Argument:**

`<scenario>` - A natural language description of a complete test scenario. May be multi-sentence. Should describe the goal, the actions, and the expected end state.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--max-steps <n>` | 20 | Maximum number of steps the Test Runner may execute. |

**Examples:**

```bash
haindy test "sign in with user@example.com and password hunter2, navigate to Settings, change the display name to 'Bob', save, and verify the new name appears in the profile header"

haindy test "attempt to sign in with an incorrect password three times and verify that the account lockout screen appears after the third attempt"
```

**Stdout on success:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "test",
  "status": "success",
  "response": "Test passed in 6 steps. Successfully signed in, navigated to Settings, updated the display name to 'Bob', saved changes. The profile header confirmed 'Bob' as the display name.",
  "steps_total": 6,
  "steps_passed": 6,
  "steps_failed": 0,
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_012.png",
  "meta": {"exit_reason": "completed", "duration_ms": 18420, "actions_taken": 14}
}
```

**Stdout on failure:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "test",
  "status": "failure",
  "response": "Test failed at step 4 of 6. Steps 1-3 (sign in, navigate to Settings, enter new name) passed. Step 4 failed: tapping 'Save' showed a validation error 'Display name must be at least 3 characters'. The name 'Bob' was rejected.",
  "steps_total": 6,
  "steps_passed": 3,
  "steps_failed": 1,
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_009.png",
  "meta": {"exit_reason": "assertion_failed", "duration_ms": 11230, "actions_taken": 9}
}
```

Note: `test` adds `steps_total`, `steps_passed`, and `steps_failed` fields to the standard envelope.

**When to use:** For anything that requires outcome validation - single interactions with expected results, multi-step journeys, regression checks, and open-ended scenarios. This is the primary command in tool call mode.

---

### `haindy explore` (v2 - not available in v1)

> **Not implemented in v1.** Use `test` for open-ended goals in the meantime - it accepts natural language scenarios and plans them automatically.

Blocked on: extending the Situational Agent from text-context gating to live-screen assessment (screenshot in, device state description out).

**Planned interface:**

```
haindy explore "<goal>" [--session <id>] [--max-steps <n>]
```

**Intended use case:** When the coding agent does not know the current device state and cannot write a `test` scenario without first knowing what screen it is on. `explore` will take a screenshot, assess the situation, build a mini-plan, and execute it autonomously.

---

## Environment Variables

| Variable | Description |
|---|---|
| `HAINDY_SESSION` | Active session ID. Set this after `session new` to avoid passing `--session` on every call. |
| `HAINDY_HOME` | Override the base directory (default: `~/.haindy`). Useful in CI. |
| `HAINDY_BACKEND` | Default backend for new sessions: `desktop` or `android`. Overridden by `--android`/`--desktop`. |

---

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Command succeeded (`status: success`). |
| 1 | Command failed or Haindy encountered an error (`status: failure` or `status: error`). |
| 2 | CLI usage error (bad arguments, unknown subcommand). |
| 3 | No active session found. |

---

## Full JSON Contract Reference

### Standard Envelope (all action subcommands)

```json
{
  "session_id": "string (UUID) | null",
  "command": "act | test | session",
  "status": "success | failure | error",
  "response": "string (natural language, always present)",
  "screenshot_path": "string (absolute path) | null",
  "meta": {
    "exit_reason": "completed | assertion_failed | max_steps_reached | max_actions_reached | agent_error | device_error | element_not_found",
    "duration_ms": "integer",
    "actions_taken": "integer"
  }
}
```

### Extended fields for `test`

```json
{
  "steps_total": "integer",
  "steps_passed": "integer",
  "steps_failed": "integer"
}
```

### Extended fields for `session list`

```json
{
  "sessions": [
    {
      "session_id": "string",
      "backend": "desktop | android",
      "created_at": "ISO 8601 datetime",
      "steps_executed": "integer",
      "idle_seconds": "integer"
    }
  ]
}
```

### Extended fields for `session vars`

```json
{
  "vars": {
    "<NAME>": "string value | \"[secret]\""
  }
}
```

### Error envelope (when Haindy itself fails)

```json
{
  "session_id": "string | null",
  "command": "string",
  "status": "error",
  "response": "Haindy encountered an internal error. The daemon process may have crashed. Details: <exception message>.",
  "screenshot_path": null,
  "meta": {"exit_reason": "agent_error", "duration_ms": 0, "actions_taken": 0}
}
```

---

## Contract Stability Guarantee

The top-level fields (`session_id`, `command`, `status`, `response`, `screenshot_path`, `meta`) are stable. New fields may be added in minor versions. Fields will not be removed or renamed without a major version bump. Skills and agents should only depend on these core fields unless they explicitly opt into extended fields.
