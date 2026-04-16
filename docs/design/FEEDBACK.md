# Feedback URLs Design

## Overview

HAINDY is an open-source, early-stage tool. We want a lightweight way for users to
report issues without running telemetry infrastructure. The first iteration uses
pre-filled GitHub issue URLs. No server, no data leaves the user's machine until
they choose to click the link.

This covers two surfaces:

1. **Batch mode** (`haindy run ...`) — a human invokes HAINDY directly. We can
   print a human-readable call-to-action at the end of the run.
2. **Tool-call mode** (`haindy act`, `haindy test`, `haindy explore`, ...) — an
   AI coding agent (Claude Code, Codex, OpenCode) invokes HAINDY. The agent
   reads the JSON envelope but the human eventually sees the agent's summary.
   We surface the feedback URL as a field in the JSON envelope so agents relay
   it when reporting a failure.

## Design

### Module

New module `haindy/feedback.py` exposes two helpers:

- `build_issue_url(*, command: str, exit_reason: str | None, error: str | None,
  run_id: str | None, extra: dict[str, str] | None = None) -> str`
- `console_feedback_hint(url: str) -> str` — one-line Rich-safe string for batch
  mode output.

The URL points at `https://github.com/Haindy/haindy/issues/new` with query
parameters:

| Param | Source |
|-------|--------|
| `title` | `[haindy <version>] <command>: <exit_reason or "issue">` |
| `body` | Rendered markdown template (see below) |
| `labels` | `user-feedback` |

The body template captures everything a maintainer needs to triage a bug,
pre-filled so users only add the narrative:

```
<!-- Pre-filled by HAINDY. Edit freely. -->

**Command:** haindy <command>
**HAINDY version:** <version>
**Platform:** <platform>
**Python:** <python_version>
**Run ID:** <run_id or "n/a">
**Exit reason:** <exit_reason or "n/a">

### What happened
<truncated error text, if any>

### What I expected

### Steps to reproduce
```

Body length is capped at ~6000 characters before URL encoding to stay under
GitHub's ~8KB URL limit. The error snippet is truncated first.

### Batch mode integration

In `haindy/main.py`, after the report is written (both success and failure
paths) and on the top-level error paths (`ScopeTriageBlockedError`,
`asyncio.TimeoutError`, generic `Exception`), print one line:

```
Feedback or bug? https://github.com/Haindy/haindy/issues/new?<pre-filled>
```

On keyboard interrupt we skip the hint (the user already decided to abort).

### Tool-call mode integration

Add an optional `feedback_url: str | None` field to `ToolCallEnvelope`. The
field is populated only when `status != success`, to keep happy-path envelopes
lean. Populated at the top of `run_tool_call_cli` / `run_tool_call_daemon_cli`
right before `model_dump_json()`.

Agents that relay tool results to humans (Claude Code, Codex) will include the
URL in their summary when a call fails, giving the human a one-click path to
report the problem.

### Opt-out

Respect `HAINDY_NO_FEEDBACK_URL=1`. When set:

- Batch mode prints no hint.
- Tool-call mode leaves `feedback_url` as `None`.

No allow-list or rate limit. The URL is generated locally; there is no
outbound request.

## Non-goals

- **Passive telemetry** (metrics, crash reports, anonymous usage pings). Out of
  scope for v1. A follow-up design can add this if we ever want it.
- **In-app issue submission**. Users always land on github.com to review the
  pre-filled body before submitting — no surprise data exfiltration.
- **Per-agent attribution** in tool-call mode. The envelope already carries
  `session_id`, `run_id`, and `command`; that's enough context in the body.

## Privacy

The pre-filled body contains: HAINDY version, Python version, platform string
(from `platform.platform()`), command name, exit reason, run id, and a
truncated error message. No API keys, file contents, screenshots, or user
prompts. The user still reviews and edits the body on github.com before
submitting.
