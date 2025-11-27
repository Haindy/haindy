# LinkedIn Desktop Automation POC

## Objectives
- Control an already-open browser with a signed-in LinkedIn session (no new Playwright context).
- Execute the experiment workflow: search person A and send connect invite; search person B (existing contact) and send a message; open person B profile and generate a sourcer-style summary.
- Operate at the OS layer (keyboard, mouse, screenshots) with zero DOM access.
- Cache stable UI coordinates for LinkedIn elements to minimize repeated computer-use cycles.
- Keep the work compatible with Haindy so learnings can be folded back into the core flow.

## Scope & Constraints
- Environment: Ubuntu with sudo available for uinput and resolution changes. 4K primary display today; prefer 1080p during runs.
- Browser: reuse an already-open window (likely Chrome/Chromium) with an authenticated LinkedIn session.
- Do not rely on Playwright; all interaction and screenshots come from the OS-level desktop path.
- No DOM access anywhere; automation is 100% visual (screenshots + absolute coordinates).

## Proposed Architecture

### 1) Desktop control surface
- Add `src/desktop/` module to encapsulate OS-level automation:
  - `DesktopController`: tracks window geometry, exposes viewport metadata, and provides a hook for “window acquisition” via the computer-use agent (model drives focus/raise/maximize using screenshots instead of shelling out to window tools).
  - `ScreenCapture`: captures the current screen or target window using `python-mss` or `ffmpeg`/`xwd`; stores artifacts under `debug_screenshots/desktop`.
  - `VirtualInput`: manages a virtual keyboard/mouse using `evdev` + `uinput` devices only (no `xdotool` fallback). Requires pre-provisioned uinput access.
  - `ResolutionManager`: optional helper that can downshift to 1080p via `xrandr`, persists previous mode, and restores on exit.

### 2) Native computer-use session
- Mirror the existing `ComputerUseSession` flow but swap the executor:
  - Use OS screenshots instead of Playwright captures.
  - Translate computer-tool actions (`click`, `type`, `scroll`, `keypress`) into `VirtualInput` calls at absolute screen coordinates.
  - Supply allowed actions and an “interaction_mode=observe_only” guard similar to the current flow.
  - Persist raw action traces and screenshots for debugging parity with the browser path.

### 3) Coordinate + run caches
- Coordinate cache (`src/desktop/cache.py`):
  - Cache entries: `{label, action, coordinates (x,y), resolution, timestamp, screenshot_hash}` stored in `data/desktop_cache/linkedin.json`.
  - Lookup before each desktop action; prefer reuse when resolution/screenshot hash match, fall back to computer-use when stale.
- Run cache (`src/core/run_cache.py`):
  - Persist plan + step interpretations keyed by a deterministic signature (requirements + settings + viewport + models).
  - Reuse cached plans and step actions when screenshots align; invalidate on failures and retry live.

### 4) Planner/Runner integration
- Add a “desktop_mode” flag to plans/test cases. When true:
  - Skip Playwright startup in `TestRunner`.
  - Inject Desktop controller metadata (window title hint, preferred resolution) into action prompts.
  - Supply initial screenshot from `ScreenCapture`.
  - Route `ActionAgent` to the desktop computer-use executor and coordinate cache before invoking the model.
- Thread the run cache through planning and execution; leverage cached plans/step actions to cut latency while auto-invalidating on failure.
- Extend test planner prompt templates to allow single-app scenarios (LinkedIn) with cached coordinates/visual targets to cut latency (no DOM cues).

### 5) Model & SDK updates
- SDK: bump `openai` to the latest 2.8.x line to stay current with Responses API and computer-use tooling. Update `pyproject.toml` + client instantiation to match.
- Models (based on current OpenAI release notes):
  - Planning/Test Runner: `gpt-5.1` (reasoning high).
  - Action Agent: `gpt-5.1` (low reasoning, text+vision) paired with `computer-use-preview` or its 5.1-aligned successor.
  - Cost-sensitive utilities: `gpt-5.1-mini` (text/vision) where heavy reasoning is unnecessary.
- Keep reasoning_level/temperature defaults aligned with existing settings and add env overrides for the new models.

### 6) Resolution handling
- Optional pre-step that records current mode, switches to 1920x1080 via `xrandr`, and restores on teardown.
- Guardrails: only act when a single display is present; otherwise log a warning and proceed at native resolution.

### 7) Observability & safety
- Extend debug logger to tag desktop runs (`mode=desktop`) and persist OS screenshots alongside action traces.
- Safety rails: block input if the focused window is not LinkedIn; stop on rapid repeated identical actions (reuse loop detection from `ComputerUseSession`).
- Add a dry-run/observe-only mode for assertion steps to avoid unintended sends.
- Typing guardrails: Action prompts now instruct select-all + single replacement for text fields; `VirtualInput` maps shifted number-row symbols (e.g., `!@#$%^&*()`) to prevent dropped punctuation.

## Experiment Plan (high level)
1) Bootstrap desktop modules (`DesktopController`, `ScreenCapture`, `VirtualInput`, `ResolutionManager`).
2) Implement `DesktopComputerUseSession` that reuses the existing action orchestration surface.
3) Add coordinate cache service and wire it into `ActionAgent` lookup before model calls.
4) Thread `desktop_mode` through planner → runner → action agent; add LinkedIn-specific test scenario covering the four target steps.
5) Manual shakedown on Ubuntu: resolution downshift, window focus, invite flow, message flow, profile summary extraction.
6) Add docs/runbook entry for prerequisites (packages, sudo setup, resolution caveats).

## Open Questions / Risks
- Permissions: uinput requires sudo; must be provisioned manually before runs (no automation of root steps).
- Multi-monitor behavior: cache needs per-display awareness; initial version may constrain to single display.
- LinkedIn UI churn: cached coordinates may stale quickly; may need cheap validation (pHash on nearby text/icon) before reuse.
- Computer-use model drift: monitor any new 5.1-specific tool behaviors and adjust allowed actions list.

## Operational prerequisites (manual)
- Ensure uinput is available and writable: load `uinput` module and configure permissions/groups for the run user (manual, may require sudo).
- Verify a single active display for resolution switching; otherwise skip the downshift step.
- Have a Firefox/LinkedIn window already open; the agent will bring it forward/maximize via computer-use, not shell commands.

## Deliverables
- New design notes (this doc).
- Desktop automation modules under `src/desktop/`.
- Updated settings/prompts to opt into desktop mode and new model defaults.
- LinkedIn POC scenario and runbook for Ubuntu execution.
