# Teal Backport Plan (Gemini + Desktop + Caching + Observability)

## Summary
Backport the core improvements from the teal fork into haindy while preserving the existing planner and reporting architecture. The priority is a Gemini computer-use provider (default), OS-level desktop control (soft keyboard/mouse), cache upgrades (coordinate, task plan, execution replay), and stronger model-call/run tracing.

## Scope and Priorities
1. Gemini computer-use provider with safe defaults and provider switching.
2. Desktop automation backend that operates the full OS (soft input, screen capture).
3. Cache system upgrades that reduce model calls and stabilize repeatable steps.
4. Improved logging and reporting artifacts, integrated with existing reporters.

## Non-goals (initial)
- Rewriting planner/runner behavior or the test report UX.
- Removing Playwright or the grid-based fallback path.
- Migrating teal's Slack-specific orchestration into haindy.

## Key Architecture Choices
- Add a desktop driver that implements the existing `BrowserDriver` interface (or a shared base) so `ComputerUseSession` can run against either Playwright or the desktop.
- Introduce a computer-use provider abstraction inside `src/agents/computer_use/` to support OpenAI and Google Gemini consistently.
- Store caches in `data/` or `cache_dir` with append-only JSON for easy inspection and invalidation.
- Keep haindy reporting as the source of truth; add new artifacts as optional attachments.

## Proposed Plan
### Phase 1: Desktop Backend (OS-level control)
Deliverables:
- `src/desktop/driver.py` implementing `BrowserDriver` methods used by computer-use (click, type, scroll, screenshot, wait, get_viewport_size).
- `src/desktop/virtual_input.py` using `/dev/uinput` for soft keyboard/mouse.
- `src/desktop/screen_capture.py` using `ffmpeg`/`x11grab`.
- `src/desktop/resolution_manager.py` using `xrandr` to optionally downshift resolution.
- Settings to configure resolution, keyboard layout, scancodes, and screenshot paths.
- New controller or factory to select Playwright vs Desktop driver.

Integration points:
- `src/core/interfaces.py` (driver contract)
- `src/browser/controller.py` or new `src/desktop/controller.py`
- `src/main.py` (CLI/runtime selection)

### Phase 2: Multi-Provider Computer Use (Gemini default)
Deliverables:
- Provider abstraction for computer-use loops.
- Google Gemini provider that uses the native computer-use tool and normalized coords.
- Settings for `CU_PROVIDER`, `GOOGLE_CU_MODEL`, `VERTEX_API_KEY`, `VERTEX_PROJECT`, `VERTEX_LOCATION`, `CU_SAFETY_POLICY`.
- Default provider to `google` (Gemini) with fallback to OpenAI.
- Auto-set preferred resolution to 1440x900 for Gemini when supported.

Integration points:
- `src/agents/computer_use/session.py`
- `src/config/settings.py`

### Phase 3: Cache Upgrades
Deliverables:
- Coordinate cache keyed by label/action/resolution with invalidation on failure.
- Task plan cache keyed by step name + context hash to reuse action plans.
- Execution replay cache to record driver actions on successful steps and replay them before invoking computer-use.
- Trace events for cache hits/stores/invalidations.

Integration points:
- `src/agents/action_agent.py` (return driver action log)
- `src/agents/test_runner.py` (plan cache + replay cache)
- `src/core/types.py` (action result schema)
- New cache modules under `src/core/` or `src/runtime/`

### Phase 4: Logging and Reporting Enhancements
Deliverables:
- Model-call JSONL log with optional screenshots.
- Per-run trace artifact with cache events, step summaries, and model calls.
- Evidence management to prune screenshots by retention limit.
- Optional desktop screen recording (GNOME gdbus) for long runs.

Integration points:
- `src/monitoring/debug_logger.py` (extend or add a model-call logger)
- `src/monitoring/reporter.py` (link to new artifacts)
- `src/main.py` (run metadata and artifact paths)

### Phase 5: Docs, Runbook, and Tests
Deliverables:
- Update `docs/RUNBOOK.md` for OS-level prerequisites (`ffmpeg`, `xrandr`, `/dev/uinput`).
- Add config docs for Gemini provider and desktop settings.
- Add smoke tests for desktop driver and Gemini provider (stubbed where needed).
- Unit tests for caches and replay logic.

## Acceptance Criteria
- Gemini provider can execute a desktop scenario end-to-end with `CU_PROVIDER=google`.
- Desktop driver works with computer-use to click/type/scroll outside the browser.
- Cache replay bypasses computer-use on repeatable steps and falls back on validation failure.
- Logs and reports include links to model-call logs, trace JSON, and screenshots.

## Risks and Mitigations
- OS-level dependencies (uinput, ffmpeg, xrandr): document clearly and add pre-flight checks.
- Coordinate drift on resolution changes: lock resolution or include resolution in cache keys.
- Provider instability: keep OpenAI provider as fallback and expose runtime switch.

## Open Questions
- Should the desktop driver live under `src/browser/` to reuse controller logic, or as `src/desktop/` with a thin adapter?
- Should cache storage default to `data/` or `cache_dir` for parity with existing haindy artifacts?
- Do we want replay cache in `test_runner` only, or also in `action_agent` for finer granularity?
