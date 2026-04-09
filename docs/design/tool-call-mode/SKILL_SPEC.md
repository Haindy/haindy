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

> Haindy is an autonomous testing agent that controls a real Android device (via ADB), iOS device (via idb), or desktop environment (via computer use). You issue natural language commands through its CLI and receive structured JSON results. Haindy runs as a background session daemon that keeps the device connection alive between your commands.

### 2. Session Setup

```bash
# Start a session (pick one backend)
haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]
haindy session new --ios [--ios-udid <UDID>] [--ios-app <BUNDLE_ID>]
haindy session new --desktop

# Read `session_id` from the JSON response, then pass it explicitly:
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# Check current screen
haindy session status --session <SESSION_ID>

# End the session when done
haindy session close --session <SESSION_ID>
```

Rule: Prefer explicit `--session <SESSION_ID>` in coding-agent and tool-runner workflows.

Rule: For web projects, make sure the site or dev server is already running before you start a desktop session. If a browser is not already open, instruct Haindy to open one and navigate to the URL like a human would. Prefer a maximized browser window.

Rule: For native desktop app projects, make sure the app is already running before you start a desktop session. If needed, instruct Haindy to bring the app to the foreground using normal desktop UI actions. Prefer a maximized app window when possible.

Rule: For Android, start the session against a device or emulator that ADB can reach, and pass `--android-serial` / `--android-app` when needed.

Rule: For iOS, start the session against a device or simulator that idb can reach, and pass `--ios-udid` / `--ios-app` when needed.

Rule: Desktop sessions may downshift resolution for speed and token savings. Maximizing the target browser or app window helps keep screenshots focused on the app instead of surrounding desktop noise.

Rule: Use `haindy session set --secret` for credentials and tokens. Prefer `--value-file` when the value is sensitive, because shell-expanded env vars still become command-line input before Haindy can redact them. Secret values are never echoed in responses or written to logs after Haindy receives them.

### 3. Command Reference (compact)

**Synchronous (return immediately with result):**

```
haindy act  "<instruction>" --session <SESSION_ID>      # single device interaction, no validation
haindy screenshot --session <SESSION_ID>                 # capture screen without AI, return path
haindy session status --session <SESSION_ID>             # AI describes current screen state
haindy session set <NAME> <VALUE> [--secret] --session <SESSION_ID>
haindy session set <NAME> --value-file <path> [--secret] --session <SESSION_ID>
```

**Asynchronous (dispatch and poll):**

```
haindy test "<scenario>" --session <SESSION_ID>          # dispatch test, returns immediately
haindy test-status --session <SESSION_ID>                # poll test progress / result

haindy explore "<goal>" --session <SESSION_ID>           # dispatch exploration, returns immediately
haindy explore-status --session <SESSION_ID>             # poll explore progress / result
```

### 4. The JSON Response

Every command returns JSON. Read `status`, `response`, and `meta.exit_reason`:

```json
{
  "session_id": "...",
  "command": "act|test|test-status|explore|explore-status|screenshot|session",
  "status": "success|failure|error",
  "response": "Natural language description of what happened.",
  "screenshot_path": "/absolute/path/to/screenshot.png",
  "meta": {"exit_reason": "...", "duration_ms": N, "actions_taken": N}
}
```

- `status: success` - proceed
- `status: failure` - the action/assertion failed; read `response` for what was observed
- `status: error` - Haindy itself failed; read `response` for diagnostic info
- `meta.exit_reason` - why the command ended; use this to decide what to do next

### 5. Choosing the Right Command

| Situation | Use |
|---|---|
| You know the exact UI element and action, no validation needed | `act` |
| You have a detailed, unambiguous scenario with explicit steps and expected outcomes | `test` |
| You have an open-ended goal and need Haindy to figure out the navigation | `explore` |
| You want to see what is on screen (AI description) | `session status` |
| You want a screenshot without AI processing | `screenshot` |

Rule: **Use `test` when you can write precise steps. Use `explore` when you cannot.** `test` requires detailed, unambiguous requirements. If you cannot describe the exact steps and expected outcomes, use `explore` with a goal instead.

Rule: **`act` does not validate anything.** If the tap succeeds but the expected outcome does not appear, `act` still returns `success`. Use `test` to validate both the action and the result.

### 6. The Async Pattern: `test` and `explore`

Both `test` and `explore` are asynchronous. They return immediately after dispatch. You poll for progress.

**`test` requires detailed, unambiguous requirements.** Do not pass vague descriptions like "test the login flow". Instead, provide explicit steps, specific UI elements, concrete values, and clear expected outcomes. If you do not have enough detail, use `explore` or `session status` first.

**`explore` accepts a goal.** The goal should be achievable by navigating visible UI. It does not need step-by-step instructions, but any context you provide (current app state, expected layout, relevant features) improves quality.

**Polling pattern:**

```bash
# Dispatch
haindy test "Step 1: Tap email field and type '{{USERNAME}}'. Step 2: Tap password field and type '{{PASSWORD}}'. Step 3: Tap 'Sign In'. Step 4: Verify dashboard appears with 'Welcome, Alice' header." --session <SESSION_ID>

# Poll until done (test_status will be "in_progress", "passed", "failed", etc.)
haindy test-status --session <SESSION_ID>
# ... wait, then poll again ...
haindy test-status --session <SESSION_ID>
```

```bash
# Dispatch
haindy explore "find the notification settings screen and report what options are available" --session <SESSION_ID>

# Poll until done (explore_status will be "in_progress", "goal_reached", "stuck", etc.)
haindy explore-status --session <SESSION_ID>
# ... wait, then poll again ...
haindy explore-status --session <SESSION_ID>
```

**`test-status` response fields:**
- `test_status`: `in_progress`, `passed`, `failed`, `error`, `timeout`, `max_steps_reached`
- `current_step`: what Haindy is currently executing (null when done)
- `steps_total`, `steps_completed`, `steps_failed`: progress counters
- `issues_found`: map of step identifiers to failure descriptions
- `elapsed_time_seconds`: wall-clock time since dispatch

**`explore-status` response fields:**
- `explore_status`: `in_progress`, `goal_reached`, `stuck`, `aborted`, `timeout`, `max_steps_reached`, `error`
- `current_focus`: what Haindy is currently trying to do (null when done)
- `todo`: the Awareness Agent's living TODO list. Each entry is `{"action": string, "status": "pending" | "in_progress" | "done" | "skipped"}`. The list is mutable -- items may be added, reordered, or skipped as the agent learns more. Useful for understanding the trajectory and reconstructing what Haindy tried.
- `observations`: accumulating list of factual observations about the app
- `elapsed_time_seconds`: wall-clock time since dispatch

Note: `explore` is driven by an Awareness Agent that maintains the TODO list and calls the Action Agent directly in a tight loop. It does not build a fixed plan up front, so it can freely backtrack when assumptions about the app turn out to be wrong. `aborted` specifically means the Awareness Agent detected that the device is no longer in a state Haindy produced (e.g. the target app lost focus, the emulator restarted, a user touched the device). It is distinct from `stuck`, which means Haindy tried and could not find a way forward on its own.

**Timeout:** `test` defaults to 300s. `explore` has no default timeout -- it runs until the goal is reached, the agent gets stuck, or max-steps is hit. You can pass `--timeout <seconds>` to either command if you want to cap execution time.

**Screenshots:** The dispatch response for `test` and `explore` includes a `screenshot_path` capturing the device state at the moment the command was accepted. Status poll responses include the latest screenshot from the background task. Haindy's screenshot timing may not match the exact moment you need. Use `haindy screenshot --session <SESSION_ID>` to take your own screenshot at a time you choose.

### 7. Session Variables

Session variables let you store values once and reference them by name in subsequent commands:

```bash
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# Reference with {{NAME}} -- shell-safe in both single- and double-quoted strings
haindy test "Step 1: Type {{USERNAME}} into the email field. Step 2: Type {{PASSWORD}} into the password field. Step 3: Tap 'Sign In'. Step 4: Verify the dashboard loads." --session <SESSION_ID>
haindy act "type {{USERNAME}} into the email field" --session <SESSION_ID>
```

The daemon interpolates `{{NAME}}` tokens before passing the instruction to agents. Secret variables appear as `[redacted]` in any response text.

Important: use the exact name you passed to `session set`. If you stored it as `USERNAME`, reference it as `{{USERNAME}}` -- not `{{LOGIN_EMAIL}}`, `{{USER}}`, or any other variation.

### 8. Handling Failures

**For `act`:** On `status: failure`, read `response` and `meta.exit_reason`.
- `element_not_found` - target not visible. Use `session status` or `screenshot` to check screen state.
- `command_timeout` - action took too long. Break into simpler steps.
- Do not retry the same `act` more than twice. Switch to `test` for better diagnostics.

**For `test`:** Check `test-status` response fields.
- `assertion_failed` - action worked but expected outcome did not appear. Adjust the scenario or investigate with `session status`.
- `max_steps_reached` - increase `--max-steps` or split into smaller test calls.
- `timeout` - increase `--timeout` or split into smaller test calls.

**For `explore`:** Check `explore-status` response fields.
- `stuck` - the agent tried and could not find a way forward. Read `observations` and the final `todo` for what was discovered and attempted. Try a different goal or provide more context.
- `goal_reached` - read `observations` for what was discovered along the way.
- `aborted` - something outside Haindy's control moved the device (user touched the emulator, another app stole focus, the emulator restarted). The session itself is still alive. Read the last screenshot and decide whether to restart the exploration or investigate manually. Do not treat this as a bug in the app under test.
- `timeout` / `max_steps_reached` - the goal may be too broad. Try a more focused goal.

**General:** `agent_error` or `device_error` means an internal failure. Read `response` for details.

### 9. Worked Example

```bash
# 1. Start session and set credentials
haindy session new --android
haindy session set USERNAME alice@example.com --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

# 2. Explore to understand the app
haindy explore "open the Acme app and find the sign-in screen" --session <SESSION_ID>
# ... poll ...
haindy explore-status --session <SESSION_ID>
# explore_status: goal_reached
# observations: ["Home screen shows Acme app icon", "App opened to welcome screen", "Welcome screen has 'Sign In' and 'Create Account' buttons"]

# 3. Now we know the UI. Run a detailed test.
haindy test "Step 1: Tap the 'Sign In' button on the welcome screen. Step 2: Tap the email field and type '{{USERNAME}}'. Step 3: Tap the password field and type '{{PASSWORD}}'. Step 4: Tap the 'Log In' button. Step 5: Verify the dashboard screen appears with text 'Welcome, Alice'." --session <SESSION_ID>
# ... poll ...
haindy test-status --session <SESSION_ID>
# test_status: passed
# steps_completed: 5, steps_failed: 0

# 4. Explore a feature area
haindy explore "navigate to the order history and report what orders are listed" --session <SESSION_ID>
# ... poll ...
haindy explore-status --session <SESSION_ID>
# explore_status: goal_reached
# observations: ["Dashboard has bottom nav: Home, Search, Orders, Profile", "Orders tab shows list of 3 past orders", "Most recent: Order #1234 - Blue Widget - Delivered"]

# 5. Close session
haindy session close --session <SESSION_ID>
```

### 10. CI / Non-Interactive Use

In CI pipelines, capture `.session_id` from `session new`, then pass it explicitly:

```bash
haindy session new --android
# Caller stores the returned .session_id as <SESSION_ID>
haindy session set USERNAME "$CI_TEST_USER" --session <SESSION_ID>
haindy session set PASSWORD --value-file /run/secrets/ci_test_pass --secret --session <SESSION_ID>

haindy test "Step 1: Open app. Step 2: Sign in with {{USERNAME}} and {{PASSWORD}}. Step 3: Complete onboarding by tapping 'Next' three times and then 'Done'. Step 4: Verify the dashboard appears." --session <SESSION_ID>
# Poll test-status until terminal state
haindy test-status --session <SESSION_ID>

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

### Teach screenshot ownership

The skill must teach that `screenshot_path` in status responses is Haindy's screenshot, taken at Haindy's timing. The coding agent should use `haindy screenshot` to take its own screenshots when it needs to verify device state at a specific moment. Coding agents already know how to read image files.

### Teach the async pattern explicitly

The async dispatch-and-poll pattern for `test` and `explore` is the most important interaction pattern to get right. The skill must show the polling loop and explain the status fields. If the coding agent treats `test` as synchronous, it will block its own tool-use loop or hit tool call timeouts.

### Teach requirement quality for `test`

The `test` command requires detailed, unambiguous scenario descriptions. Vague inputs produce unreliable plans. The skill must show good examples (with explicit steps and assertions) and bad examples (vague one-liners) so the coding agent understands the quality bar.

---

## Skill File Template

The actual skill file content (adapt placement per agent):

```markdown
# Using Haindy for device testing

Haindy is an autonomous testing agent that controls a real Android device (via ADB),
iOS device (via idb), or desktop environment (via computer use). You issue natural
language commands through its CLI and receive structured JSON results.

## Session setup

    haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]
    haindy session new --ios [--ios-udid <UDID>] [--ios-app <BUNDLE_ID>]
    haindy session new --desktop

Read `session_id` from the JSON response and pass it explicitly on every command.

Device startup rules:
- Web: make sure the site or dev server is running before starting a desktop session.
  If no browser is open, instruct Haindy to open one and navigate.
- Desktop app: make sure the app is running, then instruct Haindy to bring it to foreground.
- Prefer maximized windows. Desktop sessions may run at reduced resolution.

Store credentials before using them:

    haindy session set USERNAME alice@example.com --session <SESSION_ID>
    haindy session set PASSWORD --value-file /run/secrets/test_password --secret --session <SESSION_ID>

Close the session when done:

    haindy session close --session <SESSION_ID>

## Commands

Synchronous (return with result):

    haindy act "<instruction>" --session <SESSION_ID>      # single action, no validation
    haindy screenshot --session <SESSION_ID>                # capture screen, no AI
    haindy session status --session <SESSION_ID>            # AI describes current screen

Asynchronous (dispatch and poll):

    haindy test "<scenario>" --session <SESSION_ID>         # dispatch test
    haindy test-status --session <SESSION_ID>               # poll progress/result

    haindy explore "<goal>" --session <SESSION_ID>          # dispatch exploration
    haindy explore-status --session <SESSION_ID>            # poll progress/result

Reference session variables with {{NAME}} in any instruction string. Use the exact name
you passed to `session set`.

## test: requires detailed requirements

Do NOT pass vague descriptions. Provide explicit steps, specific UI elements, values, and
expected outcomes.

Good:
    haindy test "Step 1: Tap email field, type '{{USERNAME}}'. Step 2: Tap password field,
    type '{{PASSWORD}}'. Step 3: Tap 'Sign In'. Step 4: Verify dashboard shows 'Welcome, Alice'."

Bad:
    haindy test "test the login"

If you do not have enough detail for test, use explore or session status first.

## explore: accepts a goal

Pass a goal achievable by navigating visible UI. Does not need step-by-step instructions.

    haindy explore "find the notification settings and report available options"

explore has no default timeout. It runs until the goal is reached, the agent gets stuck,
or max-steps is hit. Pass --timeout <seconds> to cap execution time.

## Async polling pattern

test and explore return immediately. Poll for progress:

    haindy test "..." --session <SESSION_ID>
    # poll until test_status is not "in_progress"
    haindy test-status --session <SESSION_ID>

test-status fields: test_status (in_progress|passed|failed|error|timeout|max_steps_reached),
current_step, steps_total, steps_completed, steps_failed, issues_found, elapsed_time_seconds.

explore-status fields: explore_status (in_progress|goal_reached|stuck|aborted|timeout|max_steps_reached|error),
current_focus, todo (list of {action, status}), observations (accumulating list), elapsed_time_seconds.

explore is driven by an Awareness Agent that maintains a living TODO list and backtracks freely.
aborted means something external moved the device (user touched it, foreign app in focus, emulator restart).
It is not an app bug -- diagnose and decide whether to restart the exploration.

## Screenshots

The dispatch response for test and explore includes a screenshot of the device at accept time.
Status poll responses include the latest screenshot from the background task. Take your own
screenshots with `haindy screenshot --session <SESSION_ID>` when you need to verify device
state at a specific moment -- Haindy's timing may not match yours.

## JSON response (always)

    {
      "session_id": "...",
      "status": "success|failure|error",
      "response": "Natural language description.",
      "screenshot_path": "/path/to/screenshot.png",
      "meta": {"exit_reason": "...", "duration_ms": N, "actions_taken": N}
    }

- status: success -> proceed
- status: failure -> read response for what was observed vs. expected
- status: error   -> Haindy internal failure, read response for diagnostics

## On failure

For act: element_not_found means target not visible. Do not retry same act more than twice.
For test: read test-status. assertion_failed, max_steps_reached, timeout.
For explore: read explore-status. stuck means Haindy tried but could not find a way forward --
try a different goal. aborted means the device left Haindy's control (user touched it, foreign
app in focus, emulator restart) -- not a bug in the app under test.
session_busy: a background task is running. Poll its status or close the session.
```

---

## Versioning

The skill file version-locks to the JSON contract version. If the contract changes in a breaking way, the skill file must be updated in the same commit. The contract stability guarantee (see CLI_SPEC.md) protects the coding agent from silent breakage.
