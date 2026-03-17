# Haindy Tool Call Mode - Skill Specification

## Purpose

A skill (or context injection file) is a prompt fragment loaded into a coding agent's context on demand. This document specifies the design, content, and teaching goals of the `haindy` skill.

The skill serves two purposes:
1. Teach the coding agent the complete `haindy` CLI interface so it can use it correctly without being trained on it.
2. Establish interaction patterns: when to use which command, how to interpret responses, and how to behave on failure.

---

## Skill Placement

The skill file content is agent-agnostic. Placement depends on the coding agent being used:

| Agent | Placement |
|---|---|
| Claude Code | `.claude/skills/haindy.md` - loaded with `/haindy` |
| Codex | Inject as a system prompt addition or project context file per Codex conventions |
| Other agents | Load into context per that agent's context injection mechanism |

The canonical skill content lives at `docs/design/tool-call-mode/SKILL_SPEC.md` (this file). Agent-specific wrappers are derived from it.

---

## Skill Content

The skill teaches the following in order:

### 1. What Haindy Is

A one-paragraph summary. The agent needs to know Haindy is an external testing agent that controls a real device, not a mock or simulation.

> Haindy is an autonomous testing agent that controls a real Android device (via ADB) or a desktop environment (via computer use). You issue natural language commands through its CLI and receive structured JSON results. Haindy runs as a background session daemon that keeps the device connection alive between your commands.

### 2. Session Setup

```bash
# Start a session (Android only)
haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]

# Start a session (desktop after the target app/site is already running)
haindy session new --desktop

# Read `session_id` from the JSON response, then pass it explicitly:
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD "$TEST_PASSWORD" --secret --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# Check current screen
haindy session status --session <SESSION_ID>

# End the session when done
haindy session close --session <SESSION_ID>
```

Rule: Prefer explicit `--session <SESSION_ID>` in coding-agent and tool-runner workflows.

Rule: For web projects, make sure the site or dev server is already running before you start a desktop session. If a browser is not already open, instruct Haindy to open one and navigate to the URL like a human would. Prefer a maximized browser window.

Rule: For native desktop app projects, make sure the app is already running before you start a desktop session. If needed, instruct Haindy to bring the app to the foreground using normal desktop UI actions. Prefer a maximized app window when possible.

Rule: For mobile, only Android is supported in v1. Start the session against a device or emulator that ADB can reach, and pass `--android-serial` / `--android-app` when needed.

Rule: Desktop sessions may downshift resolution for speed and token savings. Maximizing the target browser or app window helps keep screenshots focused on the app instead of surrounding desktop noise.

Rule: Use `haindy session set --secret` for credentials and tokens. Prefer `--value-file` when the value is sensitive, because shell-expanded env vars still become command-line input before Haindy can redact them. Secret values are never echoed in responses or written to logs after Haindy receives them.

### 3. Command Reference (compact)

```
haindy act  "<single action>" --session <SESSION_ID>  # direct device interaction, no validation
haindy test "<scenario>" --session <SESSION_ID>       # planned multi-step test with pass/fail
haindy session status --session <SESSION_ID>          # take screenshot and describe current screen state
haindy session set <NAME> <VALUE> [--secret] --session <SESSION_ID>  # store a session variable
haindy session set <NAME> --value-file <path> [--secret] --session <SESSION_ID>  # read a variable value from a file
```

### 4. The JSON Response

Every command returns JSON. The agent must read `status`, `response`, and `meta.exit_reason`:

```json
{
  "session_id": "...",
  "command": "act|test|session",
  "status": "success|failure|error",
  "response": "What happened in natural language.",
  "screenshot_path": "/absolute/path/to/screenshot.png",
  "meta": {
    "exit_reason": "completed|assertion_failed|max_steps_reached|element_not_found|command_timeout|agent_error|device_error|session_busy",
    "duration_ms": 1243,
    "actions_taken": 3
  }
}
```

- `status: success` - proceed
- `status: failure` - the action/assertion failed; read `response` for what was observed
- `status: error` - Haindy itself failed; read `response` for diagnostic info
- `meta.exit_reason` - why the command ended; use this to decide whether to retry or change approach

### 5. Choosing the Right Command

| Situation | Use |
|---|---|
| You know the exact UI element and action, no validation needed | `act` |
| You want to validate an outcome, whether simple or complex | `test` |
| You just want to see what is on screen | `session status` |

Rule: **Prefer `test` over `act` when you care about the result.** `act` does not validate anything. If the tap succeeds but the expected outcome does not appear, `act` still returns `success`. Use `test` to validate both the action and the result.

### 6. Session Variables

Session variables let you store values once and reference them by name in subsequent commands:

```bash
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD "$TEST_PASSWORD" --secret --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# Reference with $NAME in any instruction string
haindy test "sign in with $USERNAME and $PASSWORD and verify the dashboard appears" --session <SESSION_ID>
haindy act "type '$USERNAME' into the email field" --session <SESSION_ID>
```

The daemon interpolates `$NAME` tokens before passing the instruction to agents. Secret variables appear as `[redacted]` in any response text.

### 7. Handling Failures

On `status: failure`, read `response` carefully - it describes what was observed vs. what was expected. Then check `meta.exit_reason` to decide how to respond:

- `assertion_failed` - the action executed but the expected outcome did not occur. Adjust the `test` scenario or investigate the app state with `session status`.
- `element_not_found` - the target was not visible. The app may still be loading, or the screen state is unexpected. Use `session status` to see the current screen.
- `max_steps_reached` - Haindy ran out of steps without completing. Use `--max-steps` to allow more, or break the scenario into smaller `test` calls.
- `command_timeout` - Haindy hit the command time limit. Retry with a larger `--timeout` or break the scenario into smaller commands.
- `agent_error` or `device_error` - internal failure. Check `response` for details.

Do not retry the same `act` instruction more than twice. If a direct action is not working, switch to `test` with an explicit expected outcome to get more diagnostic information.

### 8. Worked Example

```bash
# 1. Start session and capture `session_id` from the JSON response
haindy session new --android
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# 2. Check what we're looking at
haindy session status --session <SESSION_ID>
# response: "Device is on the home screen of the Android launcher."

# 3. Sign in and reach the dashboard
haindy test "open the Acme app, sign in with $USERNAME and $PASSWORD, and verify the dashboard is shown" --session <SESSION_ID>
# status: success
# response: "Test passed in 3 steps. App opened, credentials entered, dashboard confirmed."

# 4. Validate a specific feature
haindy test "tap on 'My Orders' and verify that a list of past orders is shown" --session <SESSION_ID>
# status: success

# 5. Run a regression test
haindy test "add item 'Blue Widget' to cart, proceed to checkout, enter shipping address '123 Main St', and verify the order summary shows the correct item and address" --session <SESSION_ID>
# status: failure
# meta.exit_reason: assertion_failed
# response: "Failed at step 3. Items added to cart and checkout reached. Entering '123 Main St' showed a validation error: 'Please enter a valid street address with apartment number'. The field requires a more specific format."

# 6. Close session
haindy session close --session <SESSION_ID>
```

### 9. CI / Non-Interactive Use

In CI pipelines, have the caller capture `.session_id` from the JSON response of `session new`, then pass it explicitly:

```bash
haindy session new --android
# Caller stores the returned .session_id as <SESSION_ID>
haindy session set USERNAME "$CI_TEST_USER" --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/ci_test_pass --secret --session <SESSION_ID>
haindy test "complete onboarding flow" --session <SESSION_ID>
haindy session close --session <SESSION_ID>
```

---

## Skill Design Principles

### Conciseness over completeness

The skill is loaded into a context window that the coding agent is already using for other work. Every line costs tokens. The skill should teach correct usage, not document every edge case. Full documentation lives in `docs/design/tool-call-mode/`.

### Teach the contract, not the internals

The coding agent does not need to know about the session daemon, Unix sockets, or the agent architecture. It needs to know: how to start a session, which commands exist, what JSON comes back, and how to behave on failure.

### The `response` field is the interface

The most important thing to teach is that `response` is always a natural language description of what happened, written by the agent that executed the command. On failure it describes the gap between expected and observed. The coding agent should treat this as first-class information, not a debug artifact.

### `meta.exit_reason` is the decision signal

`status` tells the coding agent whether the command succeeded. `meta.exit_reason` tells it *why* it failed, which is what determines the correct recovery action. The skill must teach both.

### Do not teach screenshot handling

The skill mentions `screenshot_path` exists but does not teach the coding agent how to read or display it. Coding agents already know how to read files. If the agent needs to look at the screenshot it will do so on its own.

---

## Skill File Template

The actual skill file content (adapt placement per agent):

```markdown
# Using Haindy for device testing

Haindy is an autonomous testing agent that controls a real Android device (via ADB)
or a desktop environment (via computer use). You issue natural language commands
through its CLI and receive structured JSON results.

## Session setup

    haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]
    # or for desktop after the target app/site is already running:
    haindy session new --desktop

Read `session_id` from the JSON response and pass it explicitly:

    haindy session status --session <SESSION_ID>

Desktop/web startup rules:

- For web projects, make sure the site or dev server is already running before starting the desktop session.
- If a browser is not open, instruct Haindy to open one and navigate to the URL like a human would.
- For native desktop apps, make sure the app is already running before starting the desktop session, then instruct Haindy to bring it to the foreground if needed.
- Prefer a maximized browser or app window when possible. Desktop sessions may run at a reduced resolution for speed and token savings.

Store credentials before using them in commands:

    haindy session set USERNAME alice@example.com --session <SESSION_ID>
    haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

Close the session when done:

    haindy session close --session <SESSION_ID>

## Commands

    haindy act  "<instruction>" --session <SESSION_ID>   # single device interaction, no validation
    haindy test "<scenario>" --session <SESSION_ID>      # planned multi-step test with pass/fail
    haindy session status --session <SESSION_ID>         # take screenshot and describe current screen state
    haindy session set <N> <V> [--secret] --session <SESSION_ID>  # store a session variable
    haindy session set <N> --value-file <path> [--secret] --session <SESSION_ID>  # read a variable value from a file

Reference session variables with $NAME in any instruction string.

## JSON response (always)

    {
      "session_id": "...",
      "command": "act|test|session",
      "status": "success|failure|error",
      "response": "Natural language description of what happened.",
      "screenshot_path": "/absolute/path/to/screenshot.png",
      "meta": {"exit_reason": "...", "duration_ms": N, "actions_taken": N}
    }

- status: success -> proceed
- status: failure -> read response for what was observed vs. expected
- status: error   -> Haindy internal failure, read response for diagnostics
- meta.exit_reason -> why it ended: completed, assertion_failed, max_steps_reached,
                      element_not_found, agent_error, device_error, session_busy

## Choosing the right command

- Use `act` only when you know the exact action and do not care about the outcome.
- Use `test` for everything that requires outcome validation, simple or complex.
- Use `session status` to orient before issuing commands to an unfamiliar screen.

## On failure

Read `response` and `meta.exit_reason` before retrying.
- assertion_failed: action worked but expected outcome did not appear.
- element_not_found: target not visible; check screen state first.
- max_steps_reached: increase --max-steps or split into smaller test calls.
- session_busy: another command is already running; wait and retry.
Do not retry the same `act` more than twice. Switch to `test` for better diagnostics.
```

---

## Versioning

The skill file version-locks to the JSON contract version. If the contract changes in a breaking way, the skill file must be updated in the same commit. The contract stability guarantee (see CLI_SPEC.md) protects the coding agent from silent breakage.
