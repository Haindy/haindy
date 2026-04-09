# Haindy Tool Call Mode - CLI Specification

## Invocation

```
haindy <subcommand> [options]
```

All tool call mode subcommands are grouped under:
- `session` - manage sessions and session variables
- Synchronous action subcommands: `act`, `screenshot`
- Async dispatch subcommands: `test`, `explore`
- Status poll subcommands: `test-status`, `explore-status`

Commands that operate on an existing session require an explicit `--session <id>` flag.
`haindy session new` does not take `--session` because it creates the session.

---

## Global Flags

| Flag | Default | Description |
|---|---|---|
| `--session <id>` | none | Required for commands that operate on an existing session. Ignored for `session new`. |
| `--json` | Enabled by default in tool call mode | Emit JSON on stdout. Always on for tool call mode; flag exists for scripting clarity. |
| `--debug` | off | Emit verbose daemon logs to stderr (does not affect stdout JSON). |

---

## `haindy session` Subcommands

### `haindy session new`

Start a new session. Spawns the daemon process, initializes the device connection, and returns the session ID.

For desktop sessions, tool call mode does not own project startup. The coding agent is responsible for making sure the target site or native desktop app is already running before or around session start; Haindy owns the UI interaction after that point.

```
haindy session new [--android | --ios | --desktop] [options]
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--android` | | Use Android ADB backend. Mutually exclusive with `--ios` and `--desktop`. |
| `--android-serial <serial>` | | Target a specific Android device or emulator by ADB serial (e.g. `emulator-5554`). Optional; uses the only connected device if omitted. |
| `--android-app <package>` | | Android package name to launch on session start (e.g. `com.example.app`). Optional. |
| `--ios` | | Use iOS idb backend. Mutually exclusive with `--android` and `--desktop`. |
| `--ios-udid <udid>` | | Target a specific iOS device or simulator by UDID. Optional; uses the only connected device if omitted. |
| `--ios-app <bundle_id>` | | iOS bundle identifier to launch on session start (e.g. `com.example.app`). Optional. |
| `--desktop` | | Use the desktop backend (computer use: OS screen capture + AI-driven input). Mutually exclusive with `--android` and `--ios`. Default if neither specified and no env config. |
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

**Recommended usage in a skill or tool runner:**

```bash
haindy session new --android
# Read `session_id` from the JSON response, then pass it explicitly:
haindy session status --session <SESSION_ID>
haindy test "open the app and verify the dashboard appears" --session <SESSION_ID>
```

**Device startup guidance:**

- Web project: make sure the site or dev server is already running before `haindy session new --desktop`. If a browser is not open yet, instruct Haindy to open one and navigate to the URL like a human would. Prefer a maximized browser window.
- Native desktop app: make sure the app is already running before `haindy session new --desktop`. If needed, instruct Haindy to bring it to the foreground using normal desktop UI actions. Prefer a maximized app window when possible.
- Android: start the session against a device or emulator that ADB can reach, and pass `--android-serial` / `--android-app` when needed.
- iOS: start the session against a device or simulator that idb can reach, and pass `--ios-udid` / `--ios-app` when needed.
- Desktop sessions may downshift resolution for speed and token savings, so maximizing the target browser or app window helps keep screenshots focused on the app instead of surrounding desktop noise.

---

### `haindy session close`

Terminate the session daemon and release the device connection.

```
haindy session close --session <id> [--force]
```

**Flags:**

| Flag | Description |
|---|---|
| `--force` | Force-close the session immediately instead of waiting for an in-progress command to finish. Intended for stuck or timed-out sessions. |

**Stdout:**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "session",
  "status": "success",
  "response": "Session closed. 14 device actions were executed during this session.",
  "screenshot_path": null,
  "meta": {"exit_reason": "completed", "duration_ms": 43, "actions_taken": 0}
}
```

---

### `haindy session list`

List all live sessions on this machine. Stale daemon artifacts are ignored.

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

Return the current state of an active session, including the latest screenshot. Handled by the Action Agent in observe-only mode.

```
haindy session status --session <id> [--timeout <seconds>]
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--timeout <seconds>` | 300 | Maximum wall-clock time for describing the current screen before Haindy returns `meta.exit_reason: command_timeout`. |

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

`actions_taken` is `1` here because `session status` actively captures a fresh screenshot to describe the current screen.

---

### `haindy session set`

Store a named variable in the session. The variable can be referenced as `{{NAME}}` in any subsequent command string. The daemon interpolates matching `{{NAME}}` tokens before passing the final instruction string to agents.

```
haindy session set <NAME> <VALUE> [--secret] --session <id>
haindy session set <NAME> --value-file <path> [--secret] --session <id>
```

**Flags:**

| Flag | Description |
|---|---|
| `--secret` | Mark the variable as secret. Secret values are stored in daemon memory only (not written to session.json or logs). Any `response` field that would echo the value back replaces it with `[redacted]`. |
| `--value-file <path>` | Read the variable value from a file instead of the command line. Useful for secrets, long JSON payloads, and values that should not appear in shell history. |

Note: `--secret` protects the value after Haindy receives it. Shells expand environment variables before `haindy` starts, so `haindy session set PASSWORD "$TEST_PASSWORD" --secret` is still passing the resolved value on the command line. If the value is sensitive, prefer `--value-file` over command-line input.

**Interpolation rules:**

- Substitution is exact text replacement of `{{NAME}}` with the stored value.
- Only variables that exist in the session are substituted. Unknown `{{NAME}}` tokens are left unchanged in the instruction string.
- `{{NAME}}` has no special meaning to the shell, so instruction strings can be double-quoted or single-quoted without affecting interpolation.
- No shell-style parsing is performed after substitution. Quotes, whitespace, and newlines remain part of the final instruction string and are passed to the model as-is.

**Examples:**

```bash
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>
haindy session set BASE_URL https://staging.example.com --session <SESSION_ID>

# Then use in commands (double or single quotes both work — {{NAME}} is not a shell token):
haindy act "type {{USERNAME}} into the email field" --session <SESSION_ID>
haindy test "sign in with {{USERNAME}} and {{PASSWORD}} and verify the dashboard appears" --session <SESSION_ID>
haindy act "navigate to {{BASE_URL}}/login" --session <SESSION_ID>
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
haindy session unset <NAME> --session <id>
```

---

### `haindy session vars`

List all variable names defined in the session. Secret variable values are not shown.

```
haindy session vars --session <id>
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

## Synchronous Action Subcommands

These commands block until the operation completes and return the result directly.

Shared behavior:
- Require an explicit `--session <id>`.
- Support `--timeout <seconds>`. If the timeout is reached, the command ends with `meta.exit_reason: command_timeout`.
- Return a single JSON object on stdout (the standard envelope, see OVERVIEW.md).
- Exit 0 on `success`, exit 1 on `failure` or `error`.
- Always capture a screenshot after the command completes and include `screenshot_path`.

---

### `haindy act`

Execute a single direct interaction on the device. No outcome validation is performed. Maps directly to the Action Agent.

```
haindy act "<instruction>" --session <id> [--timeout <seconds>]
```

**Argument:**

`<instruction>` - A natural language description of the single action to perform.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--timeout <seconds>` | 300 | Maximum wall-clock time for the command before Haindy stops execution and returns `meta.exit_reason: command_timeout`. |

**Examples:**

```bash
haindy act "tap the Login button" --session <SESSION_ID>
haindy act "type 'hunter2' into the password field" --session <SESSION_ID>
haindy act "scroll down until the Terms section is visible" --session <SESSION_ID>
haindy act "press the back button" --session <SESSION_ID>
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

## Async Dispatch Subcommands

These commands dispatch work to the daemon's background task runner and return immediately. The coding agent polls for progress using the corresponding status command (`test-status` or `explore-status`).

Shared behavior:
- Require an explicit `--session <id>`.
- Return immediately with `meta.exit_reason: dispatched` and exit 0.
- The daemon runs one background task at a time per session. If a background task is already running, the dispatch returns `status: error` with `meta.exit_reason: session_busy`.
- The `--timeout` flag sets the wall-clock budget for the background task, not for the dispatch itself.

---

### `haindy test`

Dispatch a test scenario for background execution. The Test Planner generates a structured plan from the description, then the Test Runner executes each step and validates outcomes. The coding agent polls `test-status` for progress and results.

The scenario description must be detailed and unambiguous. Vague or high-level descriptions produce unreliable plans. The coding agent should provide explicit steps, specific UI elements to interact with, concrete values to enter, and clear expected outcomes. If the coding agent does not have enough detail to write an unambiguous scenario, it should use `explore` or `session status` first to understand the current state.

```
haindy test "<scenario>" --session <id> [--max-steps <n>] [--timeout <seconds>]
```

**Argument:**

`<scenario>` - A detailed, unambiguous description of the test scenario. Must include: what actions to perform, in what order, what values to use, and what the expected outcome is. May be multi-sentence.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--max-steps <n>` | 20 | Maximum number of steps the Test Runner may execute before returning `max_steps_reached`. |
| `--timeout <seconds>` | 300 | Maximum wall-clock time for the background task. When reached, the test stops with `timeout`. |

**Examples:**

```bash
# Good: detailed, unambiguous, explicit steps and expected outcomes
haindy test "Step 1: Tap the email field and type 'alice@example.com'. Step 2: Tap the password field and type '{{PASSWORD}}'. Step 3: Tap the 'Sign In' button. Step 4: Verify the dashboard screen appears with the text 'Welcome, Alice' in the header." --session <SESSION_ID>

# Good: clear precondition, specific action, explicit assertion
haindy test "Starting from the Settings screen, tap 'Change Password', enter 'oldpass123' in the current password field and 'newpass456' in the new password field, tap 'Save', and verify a success toast appears saying 'Password updated'." --session <SESSION_ID>

# Bad: vague, no specific steps or expected outcomes
# haindy test "test the login flow" --session <SESSION_ID>
```

**Stdout (dispatch acknowledgement):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "test",
  "status": "success",
  "response": "Test dispatched. Poll test-status for progress.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_005.png",
  "meta": {"exit_reason": "dispatched", "duration_ms": 52, "actions_taken": 1}
}
```

The dispatch takes an initial screenshot before starting the background task. This captures the device state at the moment the test was accepted and provides visual context in the dispatch response.

**When to use:** When the coding agent has a well-defined scenario with explicit steps and expected outcomes. The scenario should be detailed enough that a human tester could follow it without asking questions. For open-ended goals or unknown screen states, use `explore` instead.

---

### `haindy test-status`

Poll the progress of a running or completed `test` background task.

```
haindy test-status --session <id>
```

If no test has been dispatched in this session, returns `status: error` with `meta.exit_reason: completed` and a response explaining no test is active.

**Stdout (test in progress):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "test-status",
  "status": "success",
  "response": "Test in progress. Completed step 3 of 6: typed 'alice@example.com' into the email field. Currently executing step 4: tap the 'Sign In' button.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_008.png",
  "test_status": "in_progress",
  "current_step": "Step 4: Tap the 'Sign In' button and wait for the dashboard to load.",
  "steps_total": 6,
  "steps_completed": 3,
  "steps_failed": 0,
  "issues_found": {},
  "elapsed_time_seconds": 14,
  "meta": {"exit_reason": "completed", "duration_ms": 5, "actions_taken": 7}
}
```

**Stdout (test passed):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "test-status",
  "status": "success",
  "response": "Test passed. All 6 steps completed successfully. The dashboard shows 'Welcome, Alice' as expected.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_012.png",
  "test_status": "passed",
  "current_step": null,
  "steps_total": 6,
  "steps_completed": 6,
  "steps_failed": 0,
  "issues_found": {},
  "elapsed_time_seconds": 42,
  "meta": {"exit_reason": "completed", "duration_ms": 4, "actions_taken": 14}
}
```

**Stdout (test failed):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "test-status",
  "status": "success",
  "response": "Test failed at step 4 of 6. Steps 1-3 passed. Step 4 failed: after tapping 'Sign In', the screen showed 'Invalid credentials' instead of the dashboard.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_009.png",
  "test_status": "failed",
  "current_step": null,
  "steps_total": 6,
  "steps_completed": 3,
  "steps_failed": 1,
  "issues_found": {
    "step_4": "Expected dashboard with 'Welcome, Alice'. Observed: error message 'Invalid credentials' on the sign-in screen."
  },
  "elapsed_time_seconds": 31,
  "meta": {"exit_reason": "assertion_failed", "duration_ms": 3, "actions_taken": 9}
}
```

**Extended fields for `test-status`:**

| Field | Type | Description |
|---|---|---|
| `test_status` | string | `in_progress`, `passed`, `failed`, `error`, `timeout`, `max_steps_reached`. |
| `current_step` | string or null | Description of the step currently being executed. `null` when test is finished. |
| `steps_total` | integer | Total steps in the planned test. |
| `steps_completed` | integer | Steps that have passed so far. |
| `steps_failed` | integer | Steps that have failed. |
| `issues_found` | object | Map of step identifiers to failure descriptions. Empty when no failures. |
| `elapsed_time_seconds` | integer | Wall-clock time since the test was dispatched. |

Note: `meta.exit_reason` on a `test-status` poll reflects the background task state, not the poll itself. While in progress, it is `completed` (the poll succeeded). When the test finishes, it reflects the test outcome (`completed`, `assertion_failed`, `max_steps_reached`, `timeout`, `agent_error`, `device_error`).

Note: `screenshot_path` reflects the latest screenshot taken by the background task. The coding agent should take its own screenshots via `screenshot` at its own timing if it needs to verify device state independently, because Haindy's screenshots may not capture the exact moment the agent needs.

---

### `haindy explore`

Dispatch an open-ended exploration goal for background execution. The Awareness Agent owns a tight loop: it examines the latest screenshot, maintains a living TODO list of concrete next actions, calls the Action Agent directly to execute the next item, and reassesses after every action. The loop continues until the goal is reached, the agent gives up as stuck, human intervention is detected, or a limit is hit (`max_steps`, optional `timeout`). The coding agent polls `explore-status` for progress.

Unlike `test`, `explore` does not build a fixed plan up front and does not use the Test Planner or Test Runner. The Awareness Agent is free to edit, reorder, skip, and add TODO items on every iteration as it learns more about the app. This makes exploration resilient to wrong assumptions: if a screen does not look like the agent expected, it simply updates the TODO instead of recording an assertion failure.

On every iteration the Awareness Agent also watches for signs of human intervention or device loss -- the device was moved to a different app, the emulator shut down, a notification or consent dialog appeared, or the screen is otherwise in a state the agent did not cause. Notifications and dialogs are handled inline by adding TODO items to dismiss or respond to them. Unrecoverable interventions (device gone, foreign app in focus) end the loop with `aborted` so the coding agent can diagnose and decide whether to retry.

Unlike `test`, `explore` does not require detailed step-by-step instructions. The coding agent provides a goal, and Haindy autonomously figures out how to navigate the device to achieve it. Any additional context the coding agent provides (current app state, relevant features, expected layout) improves the quality of exploration.

The goal should be achievable by looking at the screens and interacting with the app. It should not require knowledge that is not discoverable from the UI itself.

```
haindy explore "<goal>" --session <id> [--max-steps <n>] [--timeout <seconds>]
```

**Argument:**

`<goal>` - A natural language description of what to explore or achieve. Should be focused and realistic -- achievable by navigating and interacting with visible UI elements.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--max-steps <n>` | 50 | Maximum number of steps before returning `max_steps_reached`. |
| `--timeout <seconds>` | none | Optional wall-clock budget for the background task. If omitted, explore runs until the goal is reached, the agent gets stuck, or max-steps is hit. |

**Examples:**

```bash
# Focused, achievable goal
haindy explore "find and open the notification settings screen" --session <SESSION_ID>

# Goal with helpful context
haindy explore "navigate to the order history and find the most recent order. The app should have a bottom navigation bar with an 'Orders' tab." --session <SESSION_ID>

# Open-ended discovery
haindy explore "explore the settings menu and report what configuration options are available" --session <SESSION_ID>
```

**Stdout (dispatch acknowledgement):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "explore",
  "status": "success",
  "response": "Explore dispatched. Poll explore-status for progress.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_003.png",
  "meta": {"exit_reason": "dispatched", "duration_ms": 48, "actions_taken": 1}
}
```

The dispatch takes an initial screenshot before starting the background task. This is the same screenshot the Awareness Agent uses to build its initial TODO list.

**When to use:** When the coding agent does not know the current device state, cannot write an unambiguous `test` scenario, or wants to discover what the app looks like and what options are available. Use `explore` for reconnaissance and `test` for verification.

---

### `haindy explore-status`

Poll the progress of a running or completed `explore` background task.

```
haindy explore-status --session <id>
```

If no explore has been dispatched in this session, returns `status: error` with a response explaining no explore is active.

**Stdout (explore in progress):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "explore-status",
  "status": "success",
  "response": "Exploring. Currently looking at the Settings main screen after navigating from the home screen via the gear icon.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_005.png",
  "explore_status": "in_progress",
  "current_focus": "Examining the Settings main screen to find notification-related options.",
  "todo": [
    {"action": "Open the Profile tab from the bottom navigation bar", "status": "done"},
    {"action": "Tap the gear icon on the Profile screen", "status": "done"},
    {"action": "Tap the 'Notifications' row on the Settings main screen", "status": "in_progress"},
    {"action": "Read the notification toggles and summarize the options available", "status": "pending"}
  ],
  "observations": [
    "Home screen has a bottom navigation bar with Home, Search, Orders, Profile tabs.",
    "Profile tab contains a gear icon that leads to Settings.",
    "Settings main screen shows: Account, Notifications, Privacy, About."
  ],
  "elapsed_time_seconds": 18,
  "meta": {"exit_reason": "completed", "duration_ms": 4, "actions_taken": 5}
}
```

**Stdout (goal reached):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "explore-status",
  "status": "success",
  "response": "Goal reached. Found and opened the notification settings screen. It contains toggles for push notifications, email notifications, and SMS alerts.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_008.png",
  "explore_status": "goal_reached",
  "current_focus": null,
  "todo": [
    {"action": "Open the Profile tab from the bottom navigation bar", "status": "done"},
    {"action": "Tap the gear icon on the Profile screen", "status": "done"},
    {"action": "Tap the 'Notifications' row on the Settings main screen", "status": "done"},
    {"action": "Read the notification toggles and summarize the options available", "status": "done"}
  ],
  "observations": [
    "Home screen has a bottom navigation bar with Home, Search, Orders, Profile tabs.",
    "Profile tab contains a gear icon that leads to Settings.",
    "Settings main screen shows: Account, Notifications, Privacy, About.",
    "Notification settings screen has toggles: Push Notifications (on), Email Notifications (on), SMS Alerts (off)."
  ],
  "elapsed_time_seconds": 27,
  "meta": {"exit_reason": "goal_reached", "duration_ms": 3, "actions_taken": 8}
}
```

**Stdout (stuck):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "explore-status",
  "status": "success",
  "response": "Exploration ended. Could not find a way to reach the notification settings. After navigating to Settings, all sub-menus were explored but none contained notification options. The app may not have a dedicated notification settings screen.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_015.png",
  "explore_status": "stuck",
  "current_focus": null,
  "todo": [
    {"action": "Open the Profile tab", "status": "done"},
    {"action": "Tap the gear icon on the Profile screen", "status": "done"},
    {"action": "Look for a 'Notifications' row on the Settings main screen", "status": "done"},
    {"action": "Check every Settings sub-menu for notification-related options", "status": "done"},
    {"action": "Locate the notification settings screen", "status": "skipped"}
  ],
  "observations": [
    "Settings contains: Account, Display, Language, About.",
    "Account sub-menu has: Email, Password, Delete Account.",
    "Display sub-menu has: Theme, Font Size.",
    "No notification-related options found in any settings sub-menu."
  ],
  "elapsed_time_seconds": 45,
  "meta": {"exit_reason": "stuck", "duration_ms": 4, "actions_taken": 15}
}
```

**Stdout (aborted -- human intervention or device loss):**

```json
{
  "session_id": "a3f9c2d1-...",
  "command": "explore-status",
  "status": "success",
  "response": "Exploration aborted. The device is no longer showing the target app. The screen is now on the Android home launcher, which Haindy did not cause. Something external (a user interaction, another app taking focus, or the emulator restarting) moved the device out of the exploration context.",
  "screenshot_path": "/home/user/.haindy/sessions/a3f9c2d1-.../screenshots/step_011.png",
  "explore_status": "aborted",
  "current_focus": null,
  "todo": [
    {"action": "Open the Profile tab", "status": "done"},
    {"action": "Tap the gear icon on the Profile screen", "status": "done"},
    {"action": "Tap the 'Notifications' row on the Settings main screen", "status": "in_progress"},
    {"action": "Read the notification toggles and summarize the options available", "status": "pending"}
  ],
  "observations": [
    "Home screen has a bottom navigation bar with Home, Search, Orders, Profile tabs.",
    "Profile tab contains a gear icon that leads to Settings.",
    "Settings main screen shows: Account, Notifications, Privacy, About.",
    "Device unexpectedly returned to Android launcher after the last action."
  ],
  "elapsed_time_seconds": 22,
  "meta": {"exit_reason": "aborted", "duration_ms": 4, "actions_taken": 11}
}
```

**Extended fields for `explore-status`:**

| Field | Type | Description |
|---|---|---|
| `explore_status` | string | `in_progress`, `goal_reached`, `stuck`, `aborted`, `timeout`, `max_steps_reached`, `error`. |
| `current_focus` | string or null | What the Awareness Agent is currently trying to do. `null` when explore is finished. |
| `todo` | list of objects | The Awareness Agent's living TODO list. Each entry is `{"action": string, "status": "pending" \| "in_progress" \| "done" \| "skipped"}`. The list is mutable across iterations -- items may be added, reordered, or skipped as the agent learns more. Completed, skipped, and pending items are all shown so the coding agent can reconstruct the trajectory. |
| `observations` | list of strings | Accumulating list of things the Awareness Agent has observed during exploration. Each entry is a factual observation about the app state or navigation structure. |
| `elapsed_time_seconds` | integer | Wall-clock time since explore was dispatched. |

Note: `explore_status` avoids the terms "passed" and "failed" because exploration is not a pass/fail assertion. It either reaches the goal (`goal_reached`), determines it cannot proceed on its own (`stuck`), is interrupted by something outside Haindy's control (`aborted`), or is bounded by limits (`timeout`, `max_steps_reached`).

`aborted` specifically means the Awareness Agent detected that the device is no longer in a state Haindy produced -- e.g. the target app lost focus, the emulator restarted, or a user touched the device. It is distinct from `stuck`, which means Haindy tried and could not find a way forward on its own.

Note: `screenshot_path` reflects the latest screenshot taken by the background task. The coding agent should take its own screenshots via `screenshot` at its own timing if it needs to verify device state independently.

---

### `haindy session prune`

Delete session directories older than a given age. Does not affect live sessions with running daemons.

```
haindy session prune --older-than <days>
```

**Flags:**

| Flag | Description |
|---|---|
| `--older-than <days>` | Required. Delete session directories whose `created_at` is older than this many days. |

**Stdout:**

```json
{
  "session_id": null,
  "command": "session",
  "status": "success",
  "response": "Pruned 3 session directories older than 7 day(s).",
  "screenshot_path": null,
  "meta": {"exit_reason": "completed", "duration_ms": 45, "actions_taken": 0}
}
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `HAINDY_HOME` | Override the base directory (default: `~/.haindy`). Useful in CI. |
| `HAINDY_AUTOMATION_BACKEND` | Default backend for new sessions: `desktop`, `mobile_adb`, or `mobile_ios`. Overridden by `--android`/`--ios`/`--desktop`. |

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

### Standard Envelope (all commands)

```json
{
  "session_id": "string (UUID) | null",
  "command": "act | screenshot | test | test-status | explore | explore-status | session",
  "status": "success | failure | error",
  "response": "string (natural language, always present)",
  "screenshot_path": "string (absolute path) | null",
  "meta": {
    "exit_reason": "string (see exit_reason values below)",
    "duration_ms": "integer",
    "actions_taken": "integer"
  }
}
```

`actions_taken` counts device operations performed. For sync commands, this is the count from the command itself. For async dispatch, this is 1 (the initial screenshot). For status polls, this is the background task's running total. A fresh screenshot taken for `session status` counts as 1. Session startup and shutdown bookkeeping does not.

### exit_reason values

| Context | Values |
|---|---|
| Sync commands (`act`, `screenshot`, `session *`) | `completed`, `element_not_found`, `command_timeout`, `agent_error`, `device_error`, `session_busy` |
| Async dispatch (`test`, `explore`) | `dispatched`, `session_busy` |
| `test-status` (terminal) | `completed`, `assertion_failed`, `max_steps_reached`, `timeout`, `agent_error`, `device_error` |
| `explore-status` (terminal) | `goal_reached`, `stuck`, `aborted`, `max_steps_reached`, `timeout`, `agent_error`, `device_error` |

### Extended fields for `test-status`

```json
{
  "test_status": "in_progress | passed | failed | error | timeout | max_steps_reached",
  "current_step": "string | null",
  "steps_total": "integer",
  "steps_completed": "integer",
  "steps_failed": "integer",
  "issues_found": {"step_N": "string"},
  "elapsed_time_seconds": "integer"
}
```

### Extended fields for `explore-status`

```json
{
  "explore_status": "in_progress | goal_reached | stuck | aborted | timeout | max_steps_reached | error",
  "current_focus": "string | null",
  "todo": [
    {"action": "string", "status": "pending | in_progress | done | skipped"}
  ],
  "observations": ["string"],
  "elapsed_time_seconds": "integer"
}
```

### Extended fields for `session list`

```json
{
  "sessions": [
    {
      "session_id": "string",
      "backend": "desktop | android | ios",
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
