---
name: haindy
description: Use when you need to drive HAINDY's tool-call mode to inspect a live desktop, Android, or iOS session, execute direct UI actions, run scenario-based tests, manage session variables, or interpret HAINDY's JSON command results.
metadata:
  short-description: Drive HAINDY tool-call mode
---

# Using HAINDY Tool-Call Mode

HAINDY controls a real desktop (Linux/X11 or macOS), Android device, or iOS device/simulator. In tool-call mode it runs as a background session daemon and every command returns exactly one JSON object on stdout.

## Invocation

`haindy` is pre-installed system-wide at `/usr/local/bin/haindy`. `idb` and `idb_companion` are also pre-installed at `/usr/local/bin/idb` and `/usr/local/bin/idb_companion`. Do NOT create a virtualenv, install packages, or run `pip install` or `brew install`. Just call `haindy` directly — if it is not on your PATH, use `/usr/local/bin/haindy`.

## Start a session

```bash
haindy session new --desktop
haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]
haindy session new --ios [--ios-udid <UDID>] [--ios-app <BUNDLE_ID>]
```

`--android-app` tells HAINDY to launch the app at session start via `adb shell monkey`. Omit it if you have already launched the app yourself (e.g. after installing it with `adb install`). Passing `--android-app` when the app is already in the foreground is harmless — HAINDY will detect that and skip the launch — but omitting it is clearer when you know the app is already running.

`--ios-udid` selects a specific device or simulator UDID. Omit it when only one simulator is booted. `--ios-app` launches the given bundle ID at session start; omit it if the app or home screen is already where you want to begin.

Read `session_id` from the JSON response and pass it explicitly on later commands.

Troubleshooting:

- `session new` should normally survive after the original CLI process exits because HAINDY launches the session daemon independently.
- If `session new` returns success but the very next command still says `No active session found`, treat it as a harness/process-lifetime issue rather than an app failure.
- Wrappers that kill the entire process container or cgroup can still defeat detached daemons. In that case, rerun from a normal shell or keep the hidden `python -m haindy.main __tool_call_daemon ...` fallback alive in a long-lived PTY for debugging.

Desktop rules:

- Start the target site or desktop app before opening the session
- Prefer a maximized target window
- Desktop `session new --url ...` is still deferred
- On macOS, grant **Accessibility** and **Screen Recording** permissions to the terminal app before starting a session (System Settings > Privacy & Security). A restart of the terminal may be required after granting permissions.

iOS rules:

- Requires macOS with `idb-companion` installed (`brew install idb-companion`) and `fb-idb` Python package
- Boot the target simulator in Xcode or via `xcrun simctl boot <UDID>` before starting the session
- For real devices: Developer Mode must be enabled (Settings > Privacy & Security > Developer Mode) and the device trusted on this Mac
- Run `idb list-targets` to confirm the device/simulator is visible before opening a session
- Modern iPhones have no hardware Home button; use `haindy act "go to home screen"` which maps to `idb ui button HOME`

## Core commands

```bash
haindy screenshot --session <SESSION_ID>
haindy session list
haindy session status --session <SESSION_ID>
haindy act "<single action>" --session <SESSION_ID>
haindy test "<scenario with an expected outcome>" --session <SESSION_ID>
haindy test-status --session <SESSION_ID>
haindy explore "<goal>" --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>
haindy session set <NAME> <VALUE> --session <SESSION_ID>
haindy session set <NAME> --value-file <PATH> --secret --session <SESSION_ID>
haindy session unset <NAME> --session <SESSION_ID>
haindy session vars --session <SESSION_ID>
haindy session close --session <SESSION_ID>
haindy session prune --older-than <DAYS>
```

Command choice:

- Use `screenshot` to capture the current screen cheaply — no AI model is invoked, returns immediately with `screenshot_path`
- Use `session status` when you also want an AI description of what is on screen
- Use `act` for one direct interaction when you do not need outcome validation
- Use `test` when you care whether the result actually happened and you can describe the scenario precisely
- Use `explore` when you have a goal but not a reliable step-by-step path yet

Prefer `screenshot` over `session status` whenever you just need to see the screen. Prefer `test` over `act` when the outcome matters.

## Async commands

`test` and `explore` return immediately after dispatch. They do not wait for the full run to finish.

- After `haindy test ...`, poll `haindy test-status --session <SESSION_ID>`
- After `haindy explore ...`, poll `haindy explore-status --session <SESSION_ID>`
- Keep polling until the task reaches a terminal state
- While a background task is active, `act`, `session status`, `test`, and `explore` will return `session_busy`
- During a background task, `test-status`, `explore-status`, `screenshot`, `session set`, `session unset`, `session vars`, and `session close` are still allowed
- `session close` cancels any active background task before the daemon exits

## Session variables

Store a variable once, then reference it by the exact same name you chose:

```bash
haindy session set EMAIL user@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/pass --secret --session <SESSION_ID>

# Reference with {{NAME}} in any instruction string
haindy act "type {{EMAIL}} into the email field" --session <SESSION_ID>
haindy test "sign in with {{EMAIL}} and {{PASSWORD}} and verify the dashboard appears" --session <SESSION_ID>
```

Rules:
- Use `{{NAME}}` — double curly braces around the variable name. This syntax is shell-safe and works in both single- and double-quoted strings.
- Use the exact name you passed to `session set`. If you stored it as `EMAIL`, reference it as `{{EMAIL}}`, not `{{LOGIN_EMAIL}}` or `{{USER}}`.
- Unknown `{{NAME}}` tokens are left unchanged in the instruction string.
- Prefer `--value-file` for secrets so sensitive data does not travel in shell history.
- Secret values are masked in responses and are not persisted to `session.json`.

## Read the JSON response

Always inspect:

- `status`: `success`, `failure`, or `error`
- `response`: natural-language explanation of what happened
- `meta.exit_reason`: machine-readable reason the command ended
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

## Current boundaries

- Tool-call mode does not own desktop app or web-server startup
- Session variables are memory-only for the life of the daemon
- Desktop `session new --url ...` is still deferred
