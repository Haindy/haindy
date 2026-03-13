# Haindy Agentic Mode - CLI Specification

## Invocation

```
haindy <subcommand> [options]
```

All tool call mode subcommands are grouped under two top-level subcommands:
- `session` - manage sessions
- Direct action subcommands: `act`, `step`, `test`, `explore`

The active session is resolved in order:
1. `HAINDY_SESSION` environment variable
2. Explicit `--session <id>` flag on the command
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

Start a new session. Spawns the daemon process, initializes the device/browser connection, and returns the session ID.

```
haindy session new [--android | --desktop] [options]
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--android` | | Use Android ADB backend. Mutually exclusive with `--desktop`. |
| `--android-serial <serial>` | | Target a specific Android device or emulator by ADB serial (e.g. `emulator-5554`). Optional; uses the only connected device if omitted. |
| `--android-app <package>` | | Android package name to launch on session start (e.g. `com.example.app`). Optional. |
| `--desktop` | | Use Playwright/Chromium backend. Mutually exclusive with `--android`. Default if neither specified and no env config. |
| `--url <url>` | | URL to open on session start (desktop only). Optional. |
| `--idle-timeout <seconds>` | 1800 | Kill daemon after this many seconds without a command. |

**Stdout on success:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "Session started with Android ADB backend. Device found: Pixel 7 (emulator-5554).",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_001.png"
}
```

**Recommended usage in a skill:**

```bash
export HAINDY_SESSION=$(haindy session new --android | jq -r .session_id)
```

---

### `haindy session close`

Terminate the session daemon and release the device/browser connection.

```
haindy session close [--session <id>]
```

**Stdout:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "Session closed. 14 steps were executed during this session."
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
  "command": "session",
  "status": "success",
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
  "response": "Session is active. Last command was 'step' 23 seconds ago. Device is on the home screen.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_007.png"
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

Execute a single direct interaction on the device or browser. No outcome validation is performed. Maps directly to the Action Agent.

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
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_003.png"
}
```

**Stdout on failure:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "act",
  "status": "failure",
  "response": "Could not find a Login button on the current screen. The screen shows an empty loading state with a spinner. The app may not have finished loading.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_003.png"
}
```

**When to use:** When the coding agent knows the exact interaction and does not need Haindy to verify an outcome. For example, navigating to a known URL, tapping a clearly visible element, or typing into a focused field.

---

### `haindy step`

Execute a natural language step: interpret the instruction into one or more actions, execute them, and validate the expected outcome. Maps to the Test Runner operating on a single step.

```
haindy step "<instruction>" [--session <id>] [--max-actions <n>]
```

**Argument:**

`<instruction>` - A natural language step that includes both the action and the expected result. The Test Runner will interpret this, drive the Action Agent, and verify the outcome.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--max-actions <n>` | 10 | Maximum number of Action Agent calls the Test Runner may make to complete this step. |

**Examples:**

```bash
haindy step "tap the Login button and verify the user is taken to the dashboard"
haindy step "enter 'user@example.com' in the email field and 'hunter2' in the password field, then submit the form"
haindy step "verify that the error message 'Invalid credentials' is visible on screen"
```

**Stdout on success:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "step",
  "status": "success",
  "response": "Step completed. Tapped 'Log In', entered credentials, submitted the form. The dashboard screen appeared with the header 'Welcome back, Alice'.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_006.png"
}
```

**Stdout on failure:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "step",
  "status": "failure",
  "response": "Step failed. Submitted the login form but the dashboard did not appear. Instead, an error banner shows 'Your account has been suspended'. Expected the dashboard screen but observed an account suspension message.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_006.png"
}
```

**When to use:** When the coding agent wants to describe an interaction and a pass/fail outcome without writing out every individual action. The Test Runner handles multi-action sequences and retries automatically.

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
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_012.png"
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
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_009.png"
}
```

Note: `test` adds `steps_total`, `steps_passed`, and `steps_failed` fields to the standard envelope.

**When to use:** When the coding agent wants to validate a complete user journey end-to-end with structured pass/fail semantics. Appropriate after implementing a feature to verify it works holistically.

---

### `haindy explore`

Open-ended goal execution. The Situational Agent assesses the current device state, the Test Planner builds a short exploratory plan, and the Test Runner executes it. The coding agent provides a goal without needing to know the exact starting state or the specific steps.

```
haindy explore "<goal>" [--session <id>] [--max-steps <n>]
```

**Argument:**

`<goal>` - A high-level goal described in natural language. May include credentials, target features, or constraints.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--max-steps <n>` | 15 | Maximum number of steps the agent may take to achieve the goal. |

**Examples:**

```bash
haindy explore "sign in as user@example.com with password hunter2"
haindy explore "find out if the checkout flow accepts a coupon code and what happens when an expired code is entered"
haindy explore "navigate to the account deletion flow and describe what confirmation steps are required"
```

**Stdout on success:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "explore",
  "status": "success",
  "response": "Goal achieved in 4 steps. The app was on the home screen. Navigated to the sign-in page, entered credentials, and successfully signed in. The dashboard is now visible showing 'Welcome, Alice'.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_004.png"
}
```

**Stdout on failure:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "explore",
  "status": "failure",
  "response": "Goal not fully achieved. Found the coupon code field at checkout and entered 'EXPIRED2023'. The app showed the message 'This coupon has expired' and did not apply the discount. The checkout flow continued normally after dismissing the error. Partial result: the UI handles expired codes gracefully with a clear error message.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_008.png"
}
```

**When to use:** When the coding agent does not know the exact state of the device or exactly how many steps are needed. Also appropriate for investigative tasks where the outcome is a description rather than a pass/fail.

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
  "session_id": "string (UUID)",
  "command": "act | step | test | explore | session",
  "status": "success | failure | error",
  "response": "string (natural language, always present)",
  "screenshot_path": "string (absolute path) | null"
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

### Error envelope (when Haindy itself fails)

```json
{
  "session_id": "string | null",
  "command": "string",
  "status": "error",
  "response": "Haindy encountered an internal error. The daemon process may have crashed. Details: <exception message>.",
  "screenshot_path": null
}
```

---

## Contract Stability Guarantee

The top-level fields (`session_id`, `command`, `status`, `response`, `screenshot_path`) are stable. New fields may be added in minor versions. Fields will not be removed or renamed without a major version bump. Skills and agents should only depend on these five core fields unless they explicitly opt into extended fields.
