# Haindy Tool Call Mode - Overview

## Operating Modes

Haindy has two distinct operating modes. Understanding the difference is essential before reading anything else in this document.

**Standard mode** (existing): A batch pipeline designed for human operators. The human writes a requirements file and an execution context file, runs `haindy run --plan requirements.md --context context.txt`, and reads the generated HTML report when it finishes. The entire planning, execution, and reporting pipeline runs as a single unattended process. The human is outside the loop during execution.

**Tool call mode** (this document): A session-based, command-driven interface designed for AI coding agents (Codex, Claude Code, etc.). Instead of a batch pipeline, the coding agent opens a persistent session and issues individual commands in real time, reading structured JSON responses and making decisions based on them. The coding agent replaces the human operator: it decides what to test, interprets results, and determines what to do next.

This distinction drives every design decision that follows. Tool call mode is not a wrapper around the standard batch pipeline - it is a separate runtime that bypasses the file-based I/O, the scope triage gate, and the WorkflowCoordinator, and instead dispatches commands directly to the agent layer through a persistent session daemon.

---

## Problem

Haindy is a capable autonomous testing agent, but its standard interface is designed for human operators. Coding agents like Codex or Claude Code cannot easily drive it as a tool within their own workflows.

The goal of tool call mode is to expose Haindy as a first-class tool that a coding agent can call from within its tool-use loop, enabling it to perform exploratory testing, validate features, and get structured feedback from a real device - without leaving its own context.

## Design Philosophy

- **CLI over MCP/API**: A well-designed CLI paired with a skill requires less context and has better adoption than an MCP server or custom API. Coding agents are trained to use CLIs and can learn new ones via a skill loaded in-context.
- **Session-based**: A persistent session daemon keeps the device alive between calls, avoiding expensive re-initialization on every command.
- **Independent daemon launch**: `session new` returns only after an independently launched daemon is ready on its Unix socket, so later commands are not coupled to the parent CLI process lifetime.
- **Layered abstraction**: Commands are tiered from direct device actions up to open-ended exploration. The coding agent picks the right level of abstraction for its needs.
- **Async long-running commands**: `test` and `explore` return immediately after dispatch and run in the background. The coding agent polls for status at its own pace, keeping control of its tool-use loop and token budget.
- **Stable JSON contract**: Every command returns the same JSON envelope. The `status` field is machine-readable. The `response` field is natural language that the coding agent can pass directly to the user or reason about.
- **Screenshot on every response**: Agents need visual grounding. Every response includes a path to the latest screenshot.

## Document Index

| Document | What it covers |
|---|---|
| [OVERVIEW.md](./OVERVIEW.md) | This file. Problem, philosophy, architecture diagram, glossary. |
| [CLI_SPEC.md](./CLI_SPEC.md) | Every command, subcommand, flag, argument, and the full JSON contract. |
| [SESSION_DAEMON.md](./SESSION_DAEMON.md) | Session daemon internals, lifecycle, IPC protocol, sequence diagrams. |
| [SKILL_SPEC.md](./SKILL_SPEC.md) | The skill/context file design: what it teaches, example interactions, per-agent placement. |

---

## High-Level Architecture

```mermaid
flowchart TD
    CA["Coding Agent\n(Codex / Claude Code / other)"]

    subgraph CLI ["haindy CLI (tool call mode)"]
        SC[session new/close/list/set/vars/prune]
        ACT[act / screenshot]
        TEST[test / test-status]
        EXP[explore / explore-status]
    end

    subgraph DAEMON ["Session Daemon (per session)"]
        SOCK[Unix Socket Listener]
        DISP[Command Dispatcher]
        BG[Background Task Runner]
    end

    subgraph AGENTS ["Haindy Agent Layer"]
        AA[Action Agent]
        TR[Test Runner]
        TP[Test Planner]
        SIT[Situational Agent]
    end

    subgraph DEVICE ["Device"]
        ADB[Android via ADB]
        IDB[iOS via idb]
        DT[Desktop via CU]
    end

    CA -->|subcommand + args| CLI
    CLI -->|IPC over Unix socket| SOCK
    SOCK --> DISP

    DISP -->|act| AA
    DISP -->|test| BG
    DISP -->|explore| BG
    BG -->|test| TP
    BG -->|explore| SIT

    SIT --> TP
    TP --> TR
    TR --> AA
    AA --> DEVICE

    DISP -->|JSON response| CLI
    CLI -->|stdout JSON| CA
```

### Component Roles

**CLI client** (`haindy <subcommand>`): Thin wrapper. Locates the session daemon socket from the explicit session ID provided on the command, sends the command over IPC, waits for the JSON response, prints to stdout, exits with code 0 (success) or 1 (failure/error).

**Session Daemon**: A long-running Python process launched by `haindy session new` through a dedicated daemonization helper. Owns the device connection for the lifetime of the session. Listens on a Unix socket at `~/.haindy/sessions/<id>/daemon.sock`. Dispatches incoming commands to the appropriate agent and returns JSON. For long-running commands (`test`, `explore`), the daemon runs the work in a background task and returns an acknowledgement immediately.

**Background Task Runner**: An internal component of the daemon that executes `test` and `explore` commands asynchronously. The daemon accepts the command, starts the background task, and returns a response immediately. The coding agent polls `test-status` or `explore-status` to track progress. Only one background task runs at a time per session (the device is sequential).

**Action Agent**: Receives a natural language instruction, takes a screenshot via computer use, and either executes a single interaction (tap, click, type, scroll) or, for `session status`, observes the current screen and returns a natural-language description without taking action. Returns immediately with the result.

**Test Runner**: Drives the Action Agent through a sequence of structured steps produced by the Test Planner, validating each step's expected outcome. Runs as a background task; progress is exposed via `test-status`.

**Test Planner**: Accepts a detailed scenario description with explicit steps and expected outcomes, and produces a structured sequence of steps. Used by the `test` command. Requires unambiguous, detailed input from the coding agent.

**Situational Agent**: Assesses live device state from a screenshot to determine the current screen context and decide how to proceed. Used by the `explore` command to handle unknown or changing screen states during open-ended exploration.

---

## Command Abstraction Hierarchy

Each command maps to a different level of the agent stack:

```
explore       ──►  Situational Agent + Test Planner + Test Runner + Action Agent  (async)
test          ──►  Test Planner + Test Runner + Action Agent                      (async)
act           ──►  Action Agent only                                              (sync)
```

The coding agent should pick the right level of abstraction:
- Use `act` when the exact interaction is known and no validation is needed. Synchronous -- returns when the action completes.
- Use `test` when the scenario is well-defined with explicit steps and expected outcomes. Asynchronous -- returns immediately; poll `test-status` for progress.
- Use `explore` when the goal is open-ended and the current screen state is unknown or unpredictable. Asynchronous -- returns immediately; poll `explore-status` for progress.

---

## JSON Response Envelope

Every command returns a single JSON object on stdout. There are two categories:

### Synchronous commands (`act`, `screenshot`, `session *`)

Return when the operation completes:

```json
{
  "session_id": "string",
  "command": "act | screenshot | session",
  "status": "success | failure | error",
  "response": "Natural language description of what happened. Always present. Especially detailed on failure.",
  "screenshot_path": "/absolute/path/to/latest/screenshot.png",
  "meta": {
    "exit_reason": "completed | element_not_found | command_timeout | agent_error | device_error | session_busy",
    "duration_ms": 4821,
    "actions_taken": 7
  }
}
```

### Async dispatch commands (`test`, `explore`)

Return immediately after the daemon accepts the command. The daemon takes an initial screenshot before dispatching the background task, so the response includes `screenshot_path` capturing the device state at the moment the command was received. The actual work runs in the background:

```json
{
  "session_id": "string",
  "command": "test | explore",
  "status": "success | error",
  "response": "Test dispatched. Poll test-status for progress.",
  "screenshot_path": "/absolute/path/to/initial/screenshot.png",
  "meta": {
    "exit_reason": "dispatched",
    "duration_ms": 52,
    "actions_taken": 1
  }
}
```

### Status poll commands (`test-status`, `explore-status`)

Return the current state of the running background task. See CLI_SPEC.md for the full response shape per command.

### Common envelope fields

| Field | Always present | Notes |
|---|---|---|
| `session_id` | Yes | Echoed from the active session. `null` for `session list`. |
| `command` | Yes | The subcommand that was run. |
| `status` | Yes | Machine-readable signal. `error` means Haindy itself failed (bug/crash), `failure` means the action or assertion failed. |
| `response` | Yes | Human-readable. On success: what happened. On failure: what was expected vs. what was observed. |
| `screenshot_path` | Yes (when session active) | Absolute path to the latest screenshot. `null` only for commands with no device context (e.g. `session list`, `session vars`). Async dispatch includes the initial screenshot taken at accept time. |
| `meta.exit_reason` | Yes | Why the command terminated. Sync: `completed`, `element_not_found`, etc. Async dispatch: `dispatched`. Status polls: reflects the background task state. |
| `meta.duration_ms` | Yes | Wall-clock time for the command in milliseconds. For status polls, this is the poll latency, not the background task elapsed time. |
| `meta.actions_taken` | Yes | Number of atomic device operations performed. For async dispatch this is 1 (the initial screenshot). For status polls this reflects the background task's running total. |

Exit codes mirror status: 0 for `success`, 1 for `failure` or `error`.

---

## Session Filesystem Layout

```
~/.haindy/
  sessions/
    <session-id>/
      daemon.sock        # Unix domain socket (IPC)
      daemon.pid         # Daemon process PID
      session.json       # Session metadata (backend, created_at, etc.)
      screenshots/       # Sequential screenshots from this session
        step_001.png
        step_002.png
        ...
      logs/
        daemon.log       # Structured daemon logs
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Coding agent** | An AI coding assistant (Codex, Claude Code, etc.) using Haindy as a tool. |
| **Session** | A persistent device connection owned by a daemon process, identified by a UUID. |
| **Session daemon** | The background process that owns the device connection and dispatches commands. |
| **Background task** | An async execution of `test` or `explore` running inside the daemon. One at a time per session. Polled via `test-status` or `explore-status`. |
| **Session variable** | A named value stored in the session, referenced as `{{VAR}}` in commands. Secret variables are masked in logs and responses. |
| **Skill** | A context-injection file that teaches a coding agent how to use `haindy` in tool call mode. Placed per-agent (e.g. `.claude/skills/` for Claude Code). |
| **IPC** | Inter-process communication between CLI client and daemon, over a Unix domain socket. |
| **act** | A single direct device interaction with no outcome validation. Synchronous. |
| **test** | A detailed, unambiguous scenario dispatched to the Test Planner and Test Runner. Asynchronous -- returns immediately; poll `test-status` for progress and results. |
| **explore** | An open-ended goal handled by the Situational Agent + Test Planner + Test Runner. Asynchronous -- returns immediately; poll `explore-status` for progress and observations. |
| **test-status** | Polls the progress of a running or completed `test` background task. |
| **explore-status** | Polls the progress of a running or completed `explore` background task. |
| **exit_reason** | The `meta` field explaining why a command terminated. Sync commands: `completed`, `element_not_found`, `command_timeout`, `agent_error`, `device_error`, `session_busy`. Async dispatch: `dispatched`. Background task results: `completed`, `assertion_failed`, `max_steps_reached`, `stuck`, `goal_reached`, `goal_unreachable`, `timeout`, `agent_error`, `device_error`. |
