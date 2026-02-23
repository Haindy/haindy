# Teal Backport Audit Issues

Date: 2026-02-20

## Confirmed Decisions
- Provider fallback should be **per call** (do not permanently mutate provider state after one failure).
- Desktop automation is **Linux-only** for now.
- Screenshot retention should cover **all screenshot outputs** (not only debug-run screenshots).

## Findings

### 1) High: Google max-turn exits can be reported as success
- Evidence: `src/agents/computer_use/session.py:683`, `src/agents/computer_use/session.py:730`, `src/agents/action_agent.py:518`
- Problem: Google loop/max-turn termination does not always append a failed turn or explicit failure status, so the action can be treated as successful.
- Impact: False-positive step passes on incomplete actions.
- Suggested fix: Emit a terminal failed/system turn on max-turn hit (same behavior as OpenAI path) and set an explicit failure flag consumed by `ActionAgent`.

### 2) High: Desktop driver start is not idempotent
- Evidence: `src/desktop/driver.py:59`, `src/desktop/driver.py:63`, `src/agents/computer_use/session.py:1672`
- Problem: `start()` always creates a new `VirtualInput` device; `ComputerUseSession` always calls `start()`.
- Impact: Repeated runs can create multiple input devices and destabilize long sessions.
- Suggested fix: Guard `start()` with `_started` and avoid reinitializing input/resolution when already active.

### 3) Medium: Safety fail-fast setting is unused
- Evidence: `src/config/settings.py:286`, `src/agents/computer_use/session.py:1847`
- Problem: `actions_computer_tool_fail_fast_on_safety` exists but safety behavior is driven only by `cu_safety_policy`.
- Impact: Confusing configuration surface and misleading behavior.
- Suggested fix: Either wire the setting into policy behavior or remove/deprecate it.

### 4) Medium: Vertex project/location settings are currently unused
- Evidence: `src/config/settings.py:246`, `src/config/settings.py:251`, `src/agents/computer_use/session.py:1667`
- Problem: `VERTEX_PROJECT` and `VERTEX_LOCATION` are documented/configured but ignored in client setup.
- Impact: Configuration ambiguity and operator confusion.
- Suggested fix: Wire both settings into Google client initialization or remove from runtime docs/config until supported.

### 5) Medium: Replay can store `press_key` payloads that Playwright cannot execute
- Evidence: `src/agents/action_agent.py:768`, `src/desktop/execution_replay.py:157`, `src/browser/driver.py:200`
- Problem: Replay actions may preserve list-valued keys and pass them directly to `press_key(str)`.
- Impact: Replay failures on valid recorded sessions.
- Suggested fix: Canonicalize replay `press_key` into a string sequence before driver execution.

### 6) Medium: Screenshot retention is partial; desktop capture naming can overwrite
- Evidence: `src/runtime/evidence.py:22`, `src/agents/test_runner.py:2159`, `src/utils/model_logging.py:96`, `src/desktop/screen_capture.py:67`
- Problem: Retention pruning is only applied for TestRunner-registered screenshots; model-log screenshots and desktop captures are not pruned. Desktop captures use second-level filenames and can collide.
- Impact: Unbounded storage growth and possible artifact loss by overwrite.
- Suggested fix: Centralize retention policy for debug/model/desktop screenshot roots and switch desktop filenames to microsecond/uuid suffixes.

### 7) Low: Desktop `scroll()` silently accepts invalid directions
- Evidence: `src/desktop/driver.py:118`
- Problem: Non-standard directions fall through instead of raising.
- Impact: Silent behavior drift and hard-to-debug wrong scrolling.
- Suggested fix: Validate and raise on invalid direction values.

### 8) Low: Provider fallback is sticky across calls
- Evidence: `src/agents/computer_use/session.py:265`
- Problem: On one Google failure, session mutates provider/model to OpenAI permanently.
- Impact: Unexpected cross-step behavior.
- Suggested fix: Keep fallback local to a single `run()` call and restore original provider afterward.

### 9) Low: Optional GNOME screen recorder exists but is not wired into runtime flow
- Evidence: `src/desktop/screen_recorder.py:20`, `src/desktop/__init__.py:13`
- Problem: Recorder is implemented/exported but unused by runner/session/main.
- Impact: Feature appears available in docs/plans but is not operational.
- Suggested fix: Add explicit runtime toggle + lifecycle hooks in test run orchestration.

### 10) Medium: Test coverage gaps for critical backport behaviors
- Evidence: `tests/test_google_computer_use_smoke.py:20`, `tests/test_computer_use_session.py:205`
- Problem: Smoke tests do not cover Google max-turn failure reporting, sticky provider fallback, screenshot retention across all sinks, or replay key-shape compatibility.
- Impact: Regressions likely in the highest-risk paths.
- Suggested fix: Add targeted tests for the above cases and a small end-to-end desktop replay round-trip.

## Notes
- Runtime test execution was blocked initially because `pytest` and project dependencies were not installed in this environment.
