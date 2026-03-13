# Haindy Agentic Mode - Session Daemon Design

## Why a Daemon

The Playwright browser and ADB device connection are expensive to initialize (1-5 seconds each). In tool call mode, a coding agent may issue dozens of commands in sequence. Re-initializing the device connection on every CLI invocation would make each `haindy act` call unacceptably slow and would break stateful navigation (page/app state is lost on each reconnect).

The session daemon is a long-running Python process that:
- Owns a single device/browser connection for the lifetime of the session
- Listens on a Unix domain socket for commands from CLI clients
- Dispatches commands to the appropriate Haindy agents
- Keeps agent instances warm between calls (no re-instantiation overhead)
- Writes screenshots and logs to the session directory

---

## Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Spawning : haindy session new

    Spawning --> Initializing : daemon process started
    Initializing --> Ready : device/browser connected

    Ready --> Executing : command received
    Executing --> Ready : command complete (response sent)
    Executing --> Ready : command failed (failure response sent)
    Executing --> Error : unrecoverable exception

    Ready --> Closing : haindy session close
    Ready --> Closing : idle timeout reached
    Error --> Closing : forced shutdown

    Closing --> [*] : daemon exits, socket removed
```

---

## Session Initialization Sequence

```mermaid
sequenceDiagram
    participant CA as Coding Agent
    participant CLI as haindy CLI
    participant FS as Filesystem (~/.haindy)
    participant D as Session Daemon
    participant DEV as Device/Browser

    CA->>CLI: haindy session new --android
    CLI->>FS: Create sessions/<uuid>/
    CLI->>D: Spawn daemon process (detached)
    D->>FS: Write daemon.pid
    D->>DEV: Initialize ADB / Playwright connection
    DEV-->>D: Connection ready
    D->>DEV: Take initial screenshot
    DEV-->>D: screenshot_001.png
    D->>FS: Write session.json (metadata)
    D->>FS: Open daemon.sock (Unix socket)
    D-->>CLI: Ready signal (via readiness pipe)
    CLI-->>CA: {"session_id": "...", "status": "success", ...}
    CA->>CA: export HAINDY_SESSION=...
```

**Readiness pipe**: The daemon inherits a write-end file descriptor from the CLI process. When the socket is open and the device is ready, it writes a single byte to signal readiness. The CLI blocks on the read-end until it receives this signal or a startup timeout fires (default: 30s). This avoids polling and race conditions.

---

## Command Dispatch Sequence: `act`

```mermaid
sequenceDiagram
    participant CA as Coding Agent
    participant CLI as haindy CLI
    participant D as Session Daemon
    participant AA as Action Agent
    participant DEV as Device/Browser

    CA->>CLI: haindy act "tap the Login button"
    CLI->>D: Connect to daemon.sock
    CLI->>D: Send JSON request: {command: "act", instruction: "tap the Login button"}
    D->>DEV: Take screenshot
    DEV-->>D: current_screen.png
    D->>AA: execute(instruction, screenshot)
    AA->>AA: Identify target element via computer-use model
    AA->>DEV: Execute tap at coordinates
    DEV-->>AA: Action complete
    AA->>DEV: Take post-action screenshot
    DEV-->>AA: result_screen.png
    AA-->>D: ActionResult(status, description)
    D->>FS: Save screenshot as step_NNN.png
    D-->>CLI: JSON response
    CLI-->>CA: {"status": "success", "response": "...", "screenshot_path": "..."}
```

---

## Command Dispatch Sequence: `step`

```mermaid
sequenceDiagram
    participant CA as Coding Agent
    participant CLI as haindy CLI
    participant D as Session Daemon
    participant TR as Test Runner
    participant AA as Action Agent
    participant DEV as Device/Browser

    CA->>CLI: haindy step "tap login and verify welcome screen"
    CLI->>D: {command: "step", instruction: "..."}
    D->>TR: run_step(instruction)

    loop Action loop (max N iterations)
        TR->>DEV: Take screenshot
        DEV-->>TR: screen.png
        TR->>TR: Interpret next action from instruction + current screen
        TR->>AA: execute(next_action)
        AA->>DEV: Perform action
        DEV-->>AA: Done
        AA-->>TR: ActionResult
        TR->>TR: Evaluate: outcome achieved?
        alt Outcome achieved
            TR-->>D: StepResult(passed, description)
        else Max actions reached
            TR-->>D: StepResult(failed, "Max actions reached. Last state: ...")
        end
    end

    D->>FS: Save final screenshot
    D-->>CLI: JSON response
    CLI-->>CA: {"status": "success|failure", "response": "..."}
```

---

## Command Dispatch Sequence: `explore`

```mermaid
sequenceDiagram
    participant CA as Coding Agent
    participant CLI as haindy CLI
    participant D as Session Daemon
    participant SIT as Situational Agent
    participant TP as Test Planner
    participant TR as Test Runner
    participant AA as Action Agent
    participant DEV as Device/Browser

    CA->>CLI: haindy explore "sign in as alice@example.com"
    CLI->>D: {command: "explore", goal: "..."}

    D->>DEV: Take screenshot
    DEV-->>D: current_screen.png

    D->>SIT: assess(goal, current_screen)
    SIT-->>D: Assessment(context, starting_state_description)

    D->>TP: plan(goal, assessment)
    TP-->>D: MiniTestPlan(steps[])

    D->>TR: execute_plan(plan)

    loop For each step in plan
        TR->>AA: execute step actions
        AA->>DEV: Perform actions
        DEV-->>AA: Done
        AA-->>TR: Results
        TR->>TR: Verify step outcome
    end

    TR-->>D: PlanResult(status, summary)
    D->>FS: Save final screenshot
    D-->>CLI: JSON response
    CLI-->>CA: {"status": "success|failure", "response": "..."}
```

---

## IPC Protocol

Communication between the CLI client and daemon uses newline-delimited JSON over a Unix domain socket.

### Request format

```json
{
  "command": "act | step | test | explore | session_status | session_close",
  "instruction": "string (for act/step/test/explore)",
  "goal": "string (alias for instruction in explore)",
  "options": {
    "max_actions": 10,
    "max_steps": 20
  }
}
```

### Response format

The full JSON response envelope (as defined in CLI_SPEC.md). Sent as a single line terminated by `\n`. The CLI reads until `\n` and exits.

### Error handling

If the daemon crashes mid-command, the socket connection is closed before a response is sent. The CLI detects EOF on the socket and emits an error envelope:

```json
{
  "session_id": "...",
  "command": "...",
  "status": "error",
  "response": "Haindy daemon connection lost mid-command. The daemon may have crashed. Check ~/.haindy/sessions/<id>/logs/daemon.log.",
  "screenshot_path": null
}
```

---

## Session Daemon Process Management

### Spawning

The CLI spawns the daemon with `subprocess` using `start_new_session=True` so it is detached from the CLI's process group. The daemon receives:
- `--session-id <uuid>` - its own session ID
- `--session-dir <path>` - base directory for this session
- `--backend <android|desktop>` - device backend to initialize
- A file descriptor number for the readiness pipe (via env var `HAINDY_READINESS_FD`)

### Idle timeout

The daemon tracks the last command time. If no command is received within `--idle-timeout` seconds (default: 1800), it initiates a clean shutdown. This prevents leaked daemon processes after a coding agent session ends unexpectedly.

### Crash recovery

If the daemon exits unexpectedly (crash, OOM, SIGKILL), the session directory and socket file remain on disk. Subsequent CLI calls detect that the socket path exists but the connection is refused (daemon is gone) and return a `status: error` response with guidance to run `haindy session close <id>` to clean up.

### Clean shutdown

`haindy session close` sends a `session_close` command over the socket. The daemon:
1. Finishes any in-progress command
2. Closes the device/browser connection
3. Writes a final summary to `session.json`
4. Removes `daemon.sock`
5. Exits

---

## Session Directory Lifecycle

```
~/.haindy/sessions/<uuid>/
    daemon.sock    # Created when daemon is ready. Removed on close.
    daemon.pid     # Written immediately on spawn. Used for orphan detection.
    session.json   # Written on ready. Updated on close with final stats.
    screenshots/
        step_001.png   # Numbered sequentially across all commands.
        step_002.png
        ...
    logs/
        daemon.log     # Rotating structured log for this session.
```

Session directories are not automatically deleted after close. They serve as an audit trail. A separate `haindy session prune --older-than <days>` command can clean them up.

---

## Concurrency

The daemon is single-threaded per session. It processes one command at a time. If a second CLI process connects while a command is executing, it receives a `status: error` response:

```json
{
  "status": "error",
  "response": "Session is busy executing a previous command. Retry when the current command completes."
}
```

This is intentional. Device state is inherently sequential. Parallel commands on the same session would produce undefined behavior.

---

## Implementation Notes

- The daemon is implemented as a Python asyncio server using `asyncio.start_unix_server`.
- Agent instances (ActionAgent, TestRunner, etc.) are created once at daemon startup and reused across commands, preserving any in-memory caches (e.g., coordinate caches).
- The `WorkflowCoordinator` is not used in tool call mode. The daemon dispatches directly to agents. This avoids the planning overhead that is part of the full `run_test` flow.
- Screenshots are taken by the daemon (not the CLI) and written to the session screenshots directory. The response includes the absolute path. The coding agent is responsible for reading the file if it needs the image content.
