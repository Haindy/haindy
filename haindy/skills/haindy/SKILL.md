---
name: haindy
description: Use when you need to drive HAINDY's tool-call mode to inspect a live desktop, Android, or iOS session, execute direct UI actions, run scenario-based tests, manage session variables, or interpret HAINDY's JSON command results.
metadata:
  short-description: Drive HAINDY tool-call mode
---

# Using HAINDY Tool-Call Mode

HAINDY is an autonomous computer use agent that controls a real Android device (via ADB), iOS device (via idb), or desktop environment (via computer use). You issue natural language commands through its CLI and receive structured JSON results. HAINDY runs as a background session daemon that keeps the device connection alive between your commands.

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
- Android: check whether the target app is already open before trying to launch it. Prefer opening it through HAINDY computer use commands. Only fall back to launching it programmatically (e.g. via `--android-app`) as a last resort.
- iOS: check whether the target app is already open before trying to launch it. Prefer opening it through HAINDY computer use commands. Only fall back to launching it programmatically (e.g. via `--ios-app`) as a last resort.
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

## Choosing the right command

**Default: use `explore` or `act`. Use `test` only when explicitly asked to run a test suite.**

| Command | Use when |
|---|---|
| `explore` | The goal spans multiple steps or requires navigating the app to find something. This is the default for interactive flows, demos, comparisons, and open-ended goals. |
| `act` | You need a single, precise interaction and want to inspect the screen yourself before deciding the next step. |
| `session status` | You want an AI description of the current screen without interacting. |
| `screenshot` | You want a raw screenshot without AI processing. |
| `test` | The user has explicitly asked for regression or system testing and has provided written specs, a test plan, Figma files, or other documentation to validate against. |

Rules:
- **`explore` is the default for any multi-step goal.** Navigating an app, opening a feature, comparing two platforms, running a demo flow — all of these are `explore`.
- **`act` is for one atomic interaction per call.** One tap, one swipe, one text entry, one key press. Never chain multiple interactions in a single `act` call.
- **`act` does not validate anything.** If the tap succeeds but the expected result does not appear, `act` still returns `success`. Use `session status` or `screenshot` after `act` to verify state.
- **`test` is not a general-purpose multi-step driver.** Do not use it because a task has steps or an expected outcome. Only use `test` when the user explicitly asks for system or regression testing and provides documentation to validate against. If you are unsure whether `test` applies, it does not — use `explore`.

## `explore`

The default command for anything that requires navigating the app. Pass a goal; HAINDY figures out the path.

```bash
haindy explore "open the Maps app, search for Madrid, and report what the search results screen shows" --session <SESSION_ID>
```

`explore` runs until the goal is reached, the agent gets stuck, external interference aborts the run, or a limit is hit. Pass `--timeout <seconds>` to cap execution time.

`explore` is driven by an Awareness Agent that maintains a living TODO list and can backtrack freely. `aborted` means something outside HAINDY moved the device or changed focus — that is not a bug in the app under test.

## `act`

For a single atomic interaction when you want direct control.

```bash
haindy act "tap the Maps icon" --session <SESSION_ID>
haindy act "tap the search bar" --session <SESSION_ID>
haindy act "type Madrid" --session <SESSION_ID>
haindy act "tap the Search key on the keyboard" --session <SESSION_ID>
```

Each call is one interaction. Take a screenshot or `session status` between calls if you need to verify state before continuing.

## `test` — restricted use only

Only use `test` when the user has explicitly asked for regression or system testing and has provided documentation (specs, a test plan, wireframes, Figma) to validate against. Do not use `test` because the task is multi-step or has checkable outcomes — use `explore` for that.

```bash
haindy test "Step 1: Tap the email field and type '{{USERNAME}}'. Step 2: Tap the password field and type '{{PASSWORD}}'. Step 3: Tap 'Sign In'. Step 4: Verify the dashboard appears with text 'Welcome, Alice' in the header." --session <SESSION_ID>
```

Do not pass vague descriptions. Every step must name a specific UI element, a concrete value, and a clear expected outcome drawn from provided documentation.

## Async pattern

`explore` and `test` dispatch background work and return immediately. Poll for progress.

```bash
haindy explore "..." --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>

haindy test "..." --session <SESSION_ID>
haindy test-status --session <SESSION_ID>
haindy test-status --session <SESSION_ID>
```

While a background task is active:
- `act`, `session status`, `explore`, and `test` may return `session_busy`
- `explore-status`, `test-status`, `screenshot`, `session set`, `session unset`, `session vars`, and `session close` are still allowed
- `session close` cancels the active background task before the daemon exits

Read these fields on status polls:
- `run_id`: stable async identifier for a `test` run
- `explore_status`: `in_progress`, `goal_reached`, `stuck`, `aborted`, `timeout`, `max_steps_reached`, `error`
- `current_focus`, `todo`, `observations`, `elapsed_time_seconds`
- `test_status`: `in_progress`, `passed`, `failed`, `error`, `timeout`, `max_steps_reached`
- `current_step`, `phase`, `last_model_agent`, `latest_action_artifact_path`
- `steps_total`, `steps_completed`, `steps_failed`, `issues_found`, `elapsed_time_seconds`

## Session variables

Store a variable once, then reference it by the exact same name you chose:

```bash
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# Reference with {{NAME}} in any instruction string
haindy act "type {{USERNAME}} into the email field" --session <SESSION_ID>
haindy explore "sign in using {{USERNAME}} and {{PASSWORD}} and navigate to the account settings screen" --session <SESSION_ID>
```

Rules:
- Use `{{NAME}}` — double curly braces around the variable name. This syntax is shell-safe and works in both single- and double-quoted strings.
- Use the exact name you passed to `session set`. If you stored it as `USERNAME`, reference it as `{{USERNAME}}`, not `{{LOGIN_EMAIL}}`, `{{USER}}`, or anything else.
- Unknown `{{NAME}}` tokens are left unchanged in the instruction string.
- Prefer `--value-file` for secrets so sensitive data does not travel in shell history.
- Secret values are masked in responses and are not persisted to `session.json`.

## Screenshots

The dispatch response for `explore` and `test` includes a screenshot of the device at accept time. Status poll responses include the latest screenshot from the background task.

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

- For `explore`: `stuck` means HAINDY tried and could not find a way forward. `aborted` means the device left HAINDY's control. Read `observations`, `todo`, and the latest screenshot before deciding what to do next.
- For `act`: `element_not_found` means the target is not visible. Use `session status` or `screenshot` to check screen state. Do not retry the same `act` more than twice.
- For `test`: read `test-status`, not just the dispatch response. Check `assertion_failed`, `max_steps_reached`, `timeout`, `phase`, `current_step`, and `latest_action_artifact_path`.
- For any command: `agent_error` or `device_error` means HAINDY itself failed. Read `response` for diagnostics.
