---
name: haindy
description: Use when you need to drive HAINDY's tool-call mode to inspect a live desktop or Android session, execute direct UI actions, run scenario-based tests, manage session variables, or interpret HAINDY's JSON command results.
metadata:
  short-description: Drive HAINDY tool-call mode
---

# Using HAINDY Tool-Call Mode

HAINDY controls a real Linux/X11 desktop or Android device. In tool-call mode it runs as a background session daemon and every command returns exactly one JSON object on stdout.

## Start a session

```bash
haindy session new --desktop
haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]
```

Read `session_id` from the JSON response and pass it explicitly on later commands.

Troubleshooting:

- `session new` should normally survive after the original CLI process exits because HAINDY launches the session daemon independently.
- If `session new` returns success but the very next command still says `No active session found`, treat it as a harness/process-lifetime issue rather than an app failure.
- Wrappers that kill the entire process container or cgroup can still defeat detached daemons. In that case, rerun from a normal shell or keep the hidden `python -m src.main __tool_call_daemon ...` fallback alive in a long-lived PTY for debugging.

Desktop rules:

- Start the target site or desktop app before opening the session
- Prefer a maximized target window
- Desktop `session new --url ...` is not part of V1

## Core commands

```bash
haindy session status --session <SESSION_ID>
haindy act "<single action>" --session <SESSION_ID>
haindy test "<scenario with an expected outcome>" --session <SESSION_ID>
haindy session set <NAME> <VALUE> --session <SESSION_ID>
haindy session set <NAME> --value-file <PATH> --secret --session <SESSION_ID>
haindy session vars --session <SESSION_ID>
haindy session close --session <SESSION_ID>
```

Command choice:

- Use `session status` to see the current screen
- Use `act` for one direct interaction when you do not need outcome validation
- Use `test` when you care whether the result actually happened

Prefer `test` over `act` when the outcome matters.

## Session variables

- Reference stored values as `$NAME`
- `$$` means a literal dollar sign
- Unknown variable names stay unchanged
- Prefer `--value-file` for secrets so sensitive data does not travel in shell history
- Secret values are masked in responses and are not persisted to `session.json`

## Read the JSON response

Always inspect:

- `status`: `success`, `failure`, or `error`
- `response`: natural-language explanation of what happened
- `meta.exit_reason`: machine-readable reason the command ended
- `screenshot_path`: absolute path to the latest screenshot when a session is active

Important `exit_reason` values:

- `completed`: command finished normally
- `assertion_failed`: the expected result did not occur
- `element_not_found`: the target was not visible
- `max_steps_reached` or `max_actions_reached`: the command ran out of budget
- `command_timeout`: the command hit its wall-clock timeout
- `session_busy`: another command is already running in that session
- `agent_error` or `device_error`: HAINDY itself failed

## V1 boundaries

- `explore` is V2 and is not available
- Tool-call mode does not own desktop app or web-server startup
- Session variables are memory-only for the life of the daemon
