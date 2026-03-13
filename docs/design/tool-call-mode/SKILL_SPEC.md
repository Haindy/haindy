# Haindy Agentic Mode - Skill Specification

## Purpose

A Claude Code skill is a prompt fragment loaded into a coding agent's context on demand. This document specifies the design, content, and teaching goals of the `haindy` skill.

The skill serves two purposes:
1. Teach the coding agent the complete `haindy` CLI interface so it can use it correctly without being trained on it.
2. Establish interaction patterns: when to use which abstraction level, how to interpret responses, and how to behave on failure.

---

## Skill Placement

The skill file lives at `.claude/skills/haindy.md` in the project. Coding agents load it with `/haindy` or when instructed to use Haindy for testing.

---

## Skill Content

The skill teaches the following in order:

### 1. What Haindy Is

A one-paragraph summary. The agent needs to know Haindy is an external testing agent that controls a real device or browser, not a mock or simulation.

> Haindy is an autonomous testing agent that controls a real Android device (via ADB) or a desktop browser (via Playwright). You issue natural language commands through its CLI and receive structured JSON results. Haindy runs as a background session daemon that keeps the device connection alive between your commands.

### 2. Session Setup

```bash
# Start a session (Android)
export HAINDY_SESSION=$(haindy session new --android | jq -r .session_id)

# Start a session (desktop browser)
export HAINDY_SESSION=$(haindy session new --desktop --url https://app.example.com | jq -r .session_id)

# Check current screen
haindy session status

# End the session when done
haindy session close
```

Rule: Always set `HAINDY_SESSION` immediately after `session new`. All subsequent commands use it automatically.

### 3. Command Reference (compact)

```
haindy act    "<single action>"          # direct interaction, no validation
haindy step   "<action + expected result>" # action + outcome validation
haindy test   "<full scenario>"          # planned multi-step test
haindy explore "<open-ended goal>"       # situational assessment + auto-plan
```

### 4. The JSON Response

Every command returns JSON. The agent must read `status` and `response`:

```json
{
  "session_id": "...",
  "command": "step",
  "status": "success | failure | error",
  "response": "What happened in natural language.",
  "screenshot_path": "/absolute/path/to/screenshot.png"
}
```

- `status: success` - proceed
- `status: failure` - the action/assertion failed; read `response` for what was observed
- `status: error` - Haindy itself failed; read `response` for diagnostic info

### 5. Choosing the Right Command

This is the most important section. Agents tend to under-use `step` and `test` when they would get better results.

| Situation | Use |
|---|---|
| You know the exact UI element and action, no validation needed | `act` |
| You want to perform an interaction AND verify an outcome | `step` |
| You want to validate a complete user journey | `test` |
| You don't know the current app state or want open-ended exploration | `explore` |
| You just want to see what's on screen | `session status` |

Rule: **Prefer `step` over `act` when you care about the result.** `act` does not validate anything. If you use `act` to tap Login and the app crashes, Haindy will still return `success` because the tap executed.

### 6. Handling Failures

On `status: failure`, the `response` field always describes what was observed vs. what was expected. The agent should:
- Read `response` carefully before retrying
- Check `screenshot_path` if visual context is needed
- Decide whether to retry the same command, adjust the instruction, or escalate to a different abstraction level

Do not retry the same `act` instruction more than twice. If a tap is failing, use `step` with explicit expected outcome to get more diagnostic information. If a `step` is failing, use `explore` to let Haindy assess the situation from scratch.

### 7. Worked Example

This example shows a complete tool call workflow:

```bash
# 1. Start session
export HAINDY_SESSION=$(haindy session new --android | jq -r .session_id)

# 2. Check what we're looking at
haindy session status
# response: "Device is on the home screen of the Android launcher."

# 3. Navigate to the app
haindy explore "open the Acme app and sign in as alice@example.com with password hunter2"
# status: success
# response: "Found Acme app icon on home screen, launched it, signed in. Dashboard is visible."

# 4. Validate a specific feature
haindy step "tap on 'My Orders' and verify that a list of past orders is shown"
# status: success

# 5. Run a regression test
haindy test "add item 'Blue Widget' to cart, proceed to checkout, enter shipping address '123 Main St', and verify the order summary shows the correct item and address"
# status: failure
# response: "Failed at step 3. Items added to cart and checkout reached. Entering '123 Main St' into the address field showed a validation error: 'Please enter a valid street address with apartment number'. The field requires a more specific format."

# 6. Close session
haindy session close
```

### 8. CI / Non-Interactive Use

In CI pipelines, skip `export` and pass `--session` explicitly:

```bash
SESSION=$(haindy session new --android | jq -r .session_id)
haindy step "verify the app launch screen loads within 3 seconds" --session "$SESSION"
RESULT=$(haindy test "complete onboarding flow" --session "$SESSION")
haindy session close --session "$SESSION"
echo "$RESULT" | jq .status
```

---

## Skill Design Principles

### Conciseness over completeness

The skill is loaded into a context window that the coding agent is already using for other work. Every line costs tokens. The skill should teach correct usage, not document every edge case. Full documentation lives in `docs/design/tool-call-mode/`.

### Teach the contract, not the internals

The coding agent does not need to know about the session daemon, Unix sockets, or the agent architecture. It needs to know: how to start a session, which commands exist, what JSON comes back, and how to behave on failure.

### The `response` field is the interface

The most important thing to teach is that `response` is always a natural language description of what happened, written by the agent that executed the command. On failure it describes the gap between expected and observed. The coding agent should treat this as first-class information, not a debug artifact.

### Do not teach screenshot handling

The skill mentions `screenshot_path` exists but does not teach the coding agent how to read or display it. Coding agents already know how to read files. Including file-reading instructions would bloat the skill. If the agent needs to look at the screenshot it will do so on its own.

---

## Skill File Template

The actual skill file to place at `.claude/skills/haindy.md`:

```markdown
# Using Haindy for device/browser testing

Haindy is an autonomous testing agent that controls a real Android device (via ADB)
or a desktop browser (via Playwright). You issue natural language commands through
its CLI and receive structured JSON results.

## Session setup

Start a session before issuing any commands. Set HAINDY_SESSION so you don't need
to pass --session on every call.

    export HAINDY_SESSION=$(haindy session new --android | jq -r .session_id)
    # or for desktop:
    export HAINDY_SESSION=$(haindy session new --desktop --url https://example.com | jq -r .session_id)

Close the session when done:

    haindy session close

## Commands

    haindy act    "<instruction>"    # single device interaction, no validation
    haindy step   "<instruction>"    # action + outcome validation
    haindy test   "<scenario>"       # full planned multi-step test
    haindy explore "<goal>"          # situational assessment + auto-plan + execute
    haindy session status            # see current screen state

## JSON response (always)

    {
      "session_id": "...",
      "command": "act|step|test|explore|session",
      "status": "success|failure|error",
      "response": "Natural language description of what happened.",
      "screenshot_path": "/absolute/path/to/screenshot.png"
    }

- status: success -> proceed
- status: failure -> read response for what was observed vs. expected
- status: error   -> Haindy internal failure, read response for diagnostics

## Choosing the right command

- Use `act` only when you know the exact action and do not care about the outcome.
- Use `step` when you want to perform an interaction AND verify a result. Prefer step over act.
- Use `test` for full multi-step user journeys that need structured pass/fail.
- Use `explore` when the current device state is unknown or the goal is open-ended.
- Use `session status` to orient before issuing commands to an unfamiliar screen.

## On failure

Read `response` carefully. It describes the gap between expected and observed.
Do not retry the same `act` more than twice. Escalate to `step` or `explore`
if a direct action is not working - they provide more diagnostic context.
```

---

## Versioning

The skill file version-locks to the JSON contract version. If the contract changes in a breaking way, the skill file must be updated in the same commit. The contract stability guarantee (see CLI_SPEC.md) protects the coding agent from silent breakage.
