# Tool Call Mode Daemon Survival Follow-Up Plan

## Status
Drafted on 2026-03-19.

## Relationship to Existing Plan
This document is a follow-up to [docs/plans/tool-mode/completed/tool-call-mode-v1-plan.md](/home/fkeegan/src/haindy/haindy/docs/plans/tool-mode/completed/tool-call-mode-v1-plan.md).
It narrows the next work item to one concrete V1 reliability gap: keeping the tool-call session daemon alive after `haindy session new` exits when the caller is a short-lived agent terminal wrapper.

## Summary

- Preserve the V1 session-daemon contract from `docs/design/tool-call-mode/SESSION_DAEMON.md`.
- Replace the current detached-child spawn with a true daemon launcher so the session process is not tied to the lifetime of the `session new` CLI process.
- Promote `haindy` to the canonical user-facing CLI entrypoint and document an install/PATH story that makes it behave like a normal command-line tool instead of a `python -m ...` workflow.
- Add explicit shutdown/signal observability so unexpected daemon death is diagnosable.
- Add regression coverage that proves the daemon survives parent-process exit in a stubbed but live end-to-end flow.
- Keep the hidden foreground daemon entrypoint as an intentional fallback for hostile wrappers that kill an entire process container or cgroup.

## Problem

Tool-call mode currently assumes that spawning the daemon with `start_new_session=True` is enough to detach it from the `haindy session new` process.
That is not reliable in some coding-agent terminal harnesses.

Separately, the repository already declares a `haindy` console script in packaging metadata, but much of the repo-facing guidance still teaches `.venv/bin/python -m src.main ...`.
That makes Haindy feel like an internal module entrypoint instead of a first-class CLI tool and keeps the install/PATH story under-specified.

Observed failure mode:

1. `haindy session new --desktop` or `--android` returns a success JSON envelope.
2. The daemon finishes startup and may even initialize the device/app successfully.
3. The parent CLI process exits.
4. The wrapper reaps or terminates the daemon anyway.
5. The next command returns `No active session found`.

This breaks the main V1 value proposition of session reuse and makes tool-call mode unreliable exactly in the environments it was built for.

## Current Behavior Snapshot

- [src/tool_call_mode/cli.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/cli.py) starts the daemon with:
  - `asyncio.create_subprocess_exec(...)`
  - `start_new_session=True`
  - stdio redirected to `DEVNULL`
  - a readiness pipe passed through `HAINDY_READINESS_FD`
- [pyproject.toml](/home/fkeegan/src/haindy/haindy/pyproject.toml) already declares:
  - `[project.scripts]`
  - `haindy = "src.main:main"`
- [README.md](/home/fkeegan/src/haindy/haindy/README.md) and [docs/RUNBOOK.md](/home/fkeegan/src/haindy/haindy/docs/RUNBOOK.md) still primarily show `.venv/bin/python -m src.main ...` examples.
- [src/tool_call_mode/daemon.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/daemon.py) correctly owns session startup, socket binding, idle timeout, and graceful close once it is alive.
- [tests/test_tool_call_mode_cli.py](/home/fkeegan/src/haindy/haindy/tests/test_tool_call_mode_cli.py) verifies spawn flags, but not daemon survival after the parent command exits.
- The bundled HAINDY skill already documents this as a known wrapper-lifetime issue.

## Design Decision

Introduce a dedicated daemon launcher that fully daemonizes the session process before control returns to `haindy session new`.

Recommended mechanism:

1. Parent CLI creates the readiness pipe and any other startup plumbing.
2. A launcher path performs `fork`.
3. The intermediate child calls `setsid()`.
4. The intermediate child performs a second `fork`.
5. The grandchild:
   - becomes the long-lived daemon process
   - redirects stdin/stdout/stderr to `/dev/null`
   - keeps only the readiness FD and any explicitly required FDs
   - `exec`s the hidden `__tool_call_daemon` entrypoint
6. The original parent waits only for readiness or startup failure, then exits.

Why this is the preferred fix:

1. It matches the actual requirement: the daemon must outlive the spawning CLI process.
2. It reduces dependence on the parent-child relationship that some wrappers appear to police.
3. It preserves the existing daemon/runtime design instead of redesigning IPC or session ownership.
4. It keeps the current public CLI contract intact.
5. It creates a natural place to centralize how Haindy resolves its own executable path when it needs to re-exec itself.

## Non-Goals

1. Do not redesign the socket protocol or JSON envelope contract.
2. Do not add MCP, HTTP, or other new control surfaces.
3. Do not make `haindy session new` responsible for web-server or desktop-app startup.
4. Do not remove the hidden `__tool_call_daemon` command.
5. Do not promise that HAINDY can survive environments that kill the entire cgroup/container; document that case as a wrapper limitation and keep a foreground fallback path.
6. Do not turn this pass into a native single-binary packaging effort; the goal is a first-class `haindy` command on PATH, not yet a compiled standalone executable.

## Target Behavior

After the fix:

1. `haindy session new` returns success only after the daemon is independently alive and ready on its socket.
2. A later `haindy session status --session <id>` works even after the original `session new` process has exited.
3. Unexpected daemon death leaves a diagnosable trace in the session log, including shutdown reason where available.
4. The hidden direct daemon mode remains usable from a long-lived PTY for wrappers that cannot support detached daemons safely.
5. Existing commands and exit codes remain unchanged for callers.
6. Repo docs and examples use `haindy ...` as the canonical invocation, with `python -m src.main ...` demoted to an internal/dev fallback.
7. After installation, users can invoke `haindy` from PATH in the same style as tools such as `gh` or `aws`.

## Implementation Plan

### 1. Add a narrow daemon-launch helper

Create a launcher helper dedicated to the `session new` flow.

Possible homes:

- new module such as `src/tool_call_mode/launcher.py`
- or a tightly scoped helper inside `src/tool_call_mode/cli.py` if it remains small

Responsibilities:

1. Create the readiness pipe.
2. Launch the daemon through a double-fork path.
3. Close inherited file descriptors correctly in each process.
4. Return enough startup information for the parent CLI to wait for readiness and produce the same JSON envelope as today.

Important rule:

- keep daemon-launch mechanics separate from envelope construction and command parsing logic

### 2. Add one canonical CLI executable resolver

Add a small helper that answers two related questions:

1. what users should invoke as the public CLI command
2. what path or argv Haindy should use when it needs to launch itself

Recommended behavior:

1. Prefer the installed `haindy` console-script path when available.
2. Preserve a fallback to `sys.executable -m src.main` for development or environments where the console script is not available yet.
3. Keep the hidden daemon entrypoint private even if the public command becomes `haindy`.

This keeps the user-facing story and self-reexec story aligned without making PATH lookup a hard requirement for local development.

### 3. Convert `session new` to use the launcher

Update [src/tool_call_mode/cli.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/cli.py) so `_handle_session_new(...)` no longer relies on `asyncio.create_subprocess_exec(..., start_new_session=True)` as the primary daemonization mechanism.

Requirements:

1. Preserve the current 30-second startup timeout behavior.
2. Preserve readiness signaling via `HAINDY_READINESS_FD`.
3. Preserve the current startup error envelopes and log-path guidance.
4. Preserve explicit backend, idle-timeout, Android serial, Android app, and debug flag propagation.
5. Route daemon launch through the canonical executable resolver rather than hardcoding only `python -m ...`.

### 4. Promote `haindy` as the canonical CLI in docs and install guidance

Update the repo-facing CLI contract so Haindy reads like a normal installed command-line tool.

Update together:

1. [README.md](/home/fkeegan/src/haindy/haindy/README.md)
2. [docs/RUNBOOK.md](/home/fkeegan/src/haindy/haindy/docs/RUNBOOK.md)
3. command examples embedded in active design docs where appropriate
4. top-level usage/help text in [src/main.py](/home/fkeegan/src/haindy/haindy/src/main.py)

Documentation goals:

1. `haindy ...` is the first example everywhere.
2. Development setup makes it obvious that activating `.venv` or installing the package exposes `haindy`.
3. The docs include a simple PATH-based install story, such as:
   - editable dev install plus activated virtualenv
   - or a user install path such as `pipx` once the project is ready for it
4. `python -m src.main ...` is retained only as a fallback/debugging note, not the primary UX.

### 5. Harden daemon shutdown observability

Update [src/tool_call_mode/daemon.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/daemon.py) to make unexpected exits easier to diagnose.

Recommended changes:

1. Install signal handlers for `SIGTERM` and `SIGHUP` where supported by the event loop.
2. Log that a shutdown signal was received before beginning cleanup.
3. Persist a final metadata note or status when shutdown is externally requested.
4. Keep graceful `session close` behavior distinct from external termination.

This does not prevent every hostile wrapper from killing the daemon, but it makes the failure mode visible instead of silent.

### 6. Keep the foreground daemon entrypoint as an intentional fallback

Do not remove the direct hidden entrypoint:

- `python -m src.main __tool_call_daemon ...`

Document it as the fallback path for:

1. local debugging
2. integration testing
3. wrappers that cannot preserve detached background processes but can keep a long-lived PTY open

This fallback should be documented as operationally useful, not as the primary V1 path.

### 7. Add end-to-end regression coverage

Add a live but stubbed integration test that proves parent exit does not kill the daemon.

Recommended test shape:

1. Stub controller and agent creation so no real X11 or ADB dependency is required.
2. Invoke `haindy session new` through the real CLI path.
3. Let the spawning command fully exit.
4. Assert the daemon is still reachable with `session status`.
5. Close the session cleanly at the end.

Important:

- this test must verify survival semantics, not just subprocess flags

### 8. Add executable smoke coverage

Add at least one targeted test that proves the installed CLI surface is wired correctly.

Good candidates:

1. a subprocess smoke test that runs `.venv/bin/haindy --help`
2. a test that verifies the command examples in docs match the installed entrypoint shape
3. a narrow unit test around the executable resolver

The goal is not to exhaustively test packaging, but to prevent drift back toward module-only invocation.

### 9. Update docs and skill guidance

After the runtime behavior changes, update the docs that describe daemon lifecycle so they reflect the new launch model accurately.

Update together:

- [docs/design/tool-call-mode/SESSION_DAEMON.md](/home/fkeegan/src/haindy/haindy/docs/design/tool-call-mode/SESSION_DAEMON.md)
- [docs/design/tool-call-mode/OVERVIEW.md](/home/fkeegan/src/haindy/haindy/docs/design/tool-call-mode/OVERVIEW.md)
- the bundled HAINDY skill at [.agents/skills/haindy/SKILL.md](/home/fkeegan/src/haindy/haindy/.agents/skills/haindy/SKILL.md)

Documentation goals:

1. Describe the daemon as independently launched rather than merely `start_new_session=True`.
2. Keep the wrapper-limitation note, but downgrade it from expected behavior to fallback guidance.
3. Document the foreground-daemon fallback for debugging and hostile wrappers.

## Code Areas

- [src/tool_call_mode/cli.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/cli.py)
- new module such as [src/tool_call_mode/launcher.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/launcher.py)
- new helper such as `src/tool_call_mode/executable.py` if the executable resolver is split out
- [src/tool_call_mode/daemon.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/daemon.py)
- [src/tool_call_mode/paths.py](/home/fkeegan/src/haindy/haindy/src/tool_call_mode/paths.py) if metadata or cleanup helpers need minor support changes
- [pyproject.toml](/home/fkeegan/src/haindy/haindy/pyproject.toml) if packaging metadata or install guidance needs minor alignment
- [src/main.py](/home/fkeegan/src/haindy/haindy/src/main.py)
- [README.md](/home/fkeegan/src/haindy/haindy/README.md)
- [docs/RUNBOOK.md](/home/fkeegan/src/haindy/haindy/docs/RUNBOOK.md)
- [tests/test_tool_call_mode_cli.py](/home/fkeegan/src/haindy/haindy/tests/test_tool_call_mode_cli.py)
- a new targeted integration-style test file if needed
- [docs/design/tool-call-mode/SESSION_DAEMON.md](/home/fkeegan/src/haindy/haindy/docs/design/tool-call-mode/SESSION_DAEMON.md)
- [docs/design/tool-call-mode/OVERVIEW.md](/home/fkeegan/src/haindy/haindy/docs/design/tool-call-mode/OVERVIEW.md)
- [.agents/skills/haindy/SKILL.md](/home/fkeegan/src/haindy/haindy/.agents/skills/haindy/SKILL.md)

## Test Plan

Add or update tests to prove:

1. `session new` still returns the same success envelope on healthy startup.
2. The daemon survives after the spawning CLI process exits.
3. `session status` works against the surviving daemon from a later process.
4. `session close` still performs graceful shutdown and removes session artifacts.
5. External termination paths record a diagnostic signal/shutdown reason when possible.
6. The codebase no longer relies on `start_new_session=True` as the sole daemon-survival mechanism.
7. The installed `haindy` command remains functional and becomes the documented primary invocation path.

Standard validation before merging implementation:

- `.venv/bin/ruff check .`
- `.venv/bin/ruff format .`
- `.venv/bin/mypy src`
- `.venv/bin/pytest`

## Acceptance Criteria

1. Tool-call sessions survive the end of `haindy session new` in environments that allow independently daemonized descendants.
2. The daemon launch contract is implemented by a dedicated helper instead of inline detached-child subprocess flags.
3. Session startup, readiness, and close behavior stay backward-compatible for callers.
4. Unexpected daemon death becomes visible in logs and session metadata.
5. Docs and skill guidance describe the new launch model truthfully and keep the fallback path documented.
6. `haindy` is the canonical CLI shown to users, with a clear install/PATH story and smoke coverage to keep it that way.

## Risks

1. Double-fork logic can be subtle around file descriptor ownership and startup failure propagation.
2. Cross-environment behavior may still differ if the host kills an entire cgroup rather than child processes.
3. Tests can become flaky if they rely on real device or desktop startup instead of stubbed runtime pieces.
4. Over-rotating into a generic process-supervision framework would add complexity without solving the core issue.
5. Switching user-facing docs from `python -m ...` to `haindy` without a clear install story could confuse contributors unless setup docs are updated in the same pass.

## Mitigation

1. Keep the launcher small, Unix-specific, and narrowly focused on one daemon startup path.
2. Preserve the existing readiness-pipe handshake so startup success remains explicit.
3. Use stubbed agents/controllers in integration tests to isolate process-lifetime behavior from runtime backends.
4. Retain the foreground-daemon fallback and document the hard limit around full-container/cgroup termination.
5. Update packaging and install docs together with command examples so the new `haindy` UX is immediately reproducible.
