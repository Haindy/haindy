---
name: haindy
description: Use when you need to drive HAINDY's tool-call mode to inspect a live desktop, Android, or iOS session, execute direct UI actions, run scenario-based tests, manage session variables, or interpret HAINDY's JSON command results.
metadata:
  short-description: Drive HAINDY tool-call mode
---

# Using HAINDY Tool-Call Mode

HAINDY is an autonomous testing agent that controls a real Android device (via ADB), iOS device (via idb), or desktop environment (via computer use). You issue natural language commands through its CLI and receive structured JSON results. HAINDY runs as a background session daemon that keeps the device connection alive between your commands.

## Session setup

```bash
haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]
haindy session new --ios [--ios-udid <UDID>] [--ios-app <BUNDLE_ID>]
haindy session new --desktop
```

Read `session_id` from the JSON response and pass it explicitly on every later command.

Rules:
- Web: make sure the site or dev server is already running before starting a desktop session. If no browser is open yet, instruct HAINDY to open one and navigate like a human would.
- Native desktop app: make sure the app is already running before starting a desktop session. If needed, instruct HAINDY to bring it to the foreground.
- Prefer maximized windows. Desktop sessions may downshift resolution for speed and token savings.
- Android: start against a device or emulator ADB can reach, and pass `--android-serial` / `--android-app` when needed.
- iOS: start against a device or simulator idb can reach, and pass `--ios-udid` / `--ios-app` when needed.
- Use `haindy session set --secret` for credentials and tokens. Prefer `--value-file` when the value is sensitive.

Store variables before using them:

```bash
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>
```

Close and clean up when done:

```bash
haindy session close --session <SESSION_ID>
haindy session prune --older-than 7
```

## Commands

```bash
haindy act "<instruction>" --session <SESSION_ID>
haindy screenshot --session <SESSION_ID>
haindy session status --session <SESSION_ID>
haindy session set <NAME> <VALUE> [--secret] --session <SESSION_ID>
haindy session set <NAME> --value-file <PATH> [--secret] --session <SESSION_ID>
haindy session unset <NAME> --session <SESSION_ID>
haindy session vars --session <SESSION_ID>

haindy test "<scenario>" --session <SESSION_ID>
haindy test-status --session <SESSION_ID>
haindy explore "<goal>" --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>
```

Choose the right command:
- `act`: exact interaction, no validation.
- `test`: detailed, unambiguous scenario with explicit steps and expected outcomes.
- `explore`: open-ended goal when you do not yet know a reliable step-by-step path.
- `session status`: AI description of the current screen.
- `screenshot`: raw screenshot without AI processing.

Rules:
- Use `test` when you can write precise steps. Use `explore` when you cannot.
- `act` does not validate anything. If the tap succeeds but the expected result does not appear, `act` can still return `success`.

## `test` requires detailed requirements

Do not pass vague one-liners. Provide explicit steps, specific UI elements, concrete values, and clear expected outcomes.

Good:

```bash
haindy test "Step 1: Tap the email field and type '{{USERNAME}}'. Step 2: Tap the password field and type '{{PASSWORD}}'. Step 3: Tap 'Sign In'. Step 4: Verify the dashboard appears with text 'Welcome, Alice' in the header." --session <SESSION_ID>
```

Bad:

```bash
haindy test "test the login flow" --session <SESSION_ID>
```

If you do not have enough detail for `test`, use `explore` or `session status` first.

## `explore` accepts a goal

Pass a goal achievable by navigating visible UI. It does not need step-by-step instructions.

```bash
haindy explore "find the notification settings screen and report what options are available" --session <SESSION_ID>
```

`explore` runs until the goal is reached, the agent gets stuck, external interference aborts the run, or a limit is hit. Pass `--timeout <seconds>` if you want to cap execution time.

## Async pattern

`test` and `explore` dispatch background work and return immediately. They do not wait for the full run to finish.

```bash
haindy test "..." --session <SESSION_ID>
haindy test-status --session <SESSION_ID>
haindy test-status --session <SESSION_ID>

haindy explore "..." --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>
```

While a background task is active:
- `act`, `session status`, `test`, and `explore` may return `session_busy`
- `test-status`, `explore-status`, `screenshot`, `session set`, `session unset`, `session vars`, and `session close` are still allowed
- `session close` cancels the active background task before the daemon exits

Read these fields on status polls:
- `run_id`: stable async identifier for a `test` run
- `test_status`: `in_progress`, `passed`, `failed`, `error`, `timeout`, `max_steps_reached`
- `current_step`, `phase`, `last_model_agent`, `latest_action_artifact_path`
- `steps_total`, `steps_completed`, `steps_failed`, `issues_found`, `elapsed_time_seconds`
- `explore_status`: `in_progress`, `goal_reached`, `stuck`, `aborted`, `timeout`, `max_steps_reached`, `error`
- `current_focus`, `todo`, `observations`, `elapsed_time_seconds`

`explore` is driven by an Awareness Agent that maintains a living TODO list and can backtrack freely. `aborted` means something outside HAINDY moved the device or changed focus. That is not automatically a bug in the app under test.

## Session variables

Store a variable once, then reference it by the exact same name you chose:

```bash
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# Reference with {{NAME}} in any instruction string
haindy act "type {{USERNAME}} into the email field" --session <SESSION_ID>
haindy test "Step 1: Type {{USERNAME}} into the email field. Step 2: Type {{PASSWORD}} into the password field. Step 3: Tap 'Sign In'. Step 4: Verify the dashboard loads." --session <SESSION_ID>
```

Rules:
- Use `{{NAME}}` — double curly braces around the variable name. This syntax is shell-safe and works in both single- and double-quoted strings.
- Use the exact name you passed to `session set`. If you stored it as `USERNAME`, reference it as `{{USERNAME}}`, not `{{LOGIN_EMAIL}}`, `{{USER}}`, or anything else.
- Unknown `{{NAME}}` tokens are left unchanged in the instruction string.
- Prefer `--value-file` for secrets so sensitive data does not travel in shell history.
- Secret values are masked in responses and are not persisted to `session.json`.

## Screenshots

The dispatch response for `test` and `explore` includes a screenshot of the device at accept time. Status poll responses include the latest screenshot from the background task.

Use `haindy screenshot --session <SESSION_ID>` when you need a screenshot at your own chosen moment. HAINDY's screenshot timing may not match yours.

## JSON response

Always inspect:

- `status`: `success`, `failure`, or `error`
- `response`: natural-language explanation of what happened
- `meta.exit_reason`: machine-readable reason the command ended
- `run_id`: stable background identifier for async `test` runs when present
- `screenshot_path`: absolute path to the latest screenshot when a session is active

Important `exit_reason` values:

- `completed`: command finished normally
- `dispatched`: async work was accepted; poll the matching `*-status` command
- `assertion_failed`: the expected result did not occur
- `element_not_found`: the target was not visible
- `max_steps_reached` or `max_actions_reached`: the command ran out of budget
- `command_timeout`: the command hit its wall-clock timeout
- `goal_reached`, `stuck`, `aborted`, `timeout`: terminal outcomes for `explore`
- `session_busy`: another command is already running in that session; poll the matching `*-status` command or close the session instead of retrying blindly
- `agent_error` or `device_error`: HAINDY itself failed

## On failure

- For `act`: `element_not_found` means the target is not visible. Use `session status` or `screenshot` to check screen state. Do not retry the same `act` more than twice.
- For `test`: read `test-status`, not just the dispatch response. Check `assertion_failed`, `max_steps_reached`, `timeout`, `phase`, `current_step`, and `latest_action_artifact_path`.
- For `explore`: `stuck` means HAINDY tried and could not find a way forward. `aborted` means the device left HAINDY's control. Read `observations`, `todo`, and the latest screenshot before deciding what to do next.
- For any command: `agent_error` or `device_error` means HAINDY itself failed. Read `response` for diagnostics.
