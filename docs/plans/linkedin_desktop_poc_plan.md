# LinkedIn Desktop POC Implementation Plan

## Goals
- Deliver a visual-only, OS-level LinkedIn automation flow (search/connect/message/summarize) using the computer-use tool.
- Keep Playwright unused; all interaction via screenshots, coordinates, and uinput-driven input.
- Make the work mergeable back into Haindy with clear toggles and guardrails.

## Inputs & Constraints
- Ubuntu, sudo available; uinput provisioned manually (no automated root).
- Single-display preferred; optional downshift to 1920x1080 via xrandr.
- Browser window already open (Firefox/LinkedIn); window focus/maximize driven by computer-use, not shell.
- Models: gpt-5.1/gpt-5.1-mini + computer-use-preview; Responses API with truncation auto.

## Milestones & Tasks

### M1: Desktop substrate
- Create `src/desktop/` package:
  - `resolution_manager.py`: detect active output, save/restore mode, switch to 1080p with guardrails.
  - `screen_capture.py`: full-screen (and optional window-region) capture to bytes, saving under `debug_screenshots/desktop`.
  - `virtual_input.py`: uinput-based mouse/keyboard (click/move/scroll/type/keypress); config for device names and rate limits.
  - `window_state.py`: read window geometry metadata (via computer-use-provided screenshots + inferred viewport); no wmctrl/xdotool calls.
- Add settings for desktop mode (enable flag, screenshot dir, input pacing, resolution toggle).

### M2: Desktop computer-use executor
- Implement `DesktopComputerUseSession` mirroring the existing browser session:
  - Provide initial screenshot and viewport metadata to Responses API with `computer-use-preview`.
  - Translate computer_call actions to VirtualInput at absolute coords; handle stabilization delays and action timeouts.
  - Loop detection, turn limits, safety events identical to browser flow; store action traces + screenshots.
- Wire ActionAgent to select desktop executor when `desktop_mode` is set; keep existing path untouched otherwise.

### M3: Coordinate cache
- Add `desktop/cache.py`:
  - Append-only JSON store at `data/desktop_cache/linkedin.json` with entries: label, action, (x,y), resolution, timestamp, screenshot hash.
  - Lookup-first strategy with lightweight visual validation (crop + pHash); on miss/fail, fall back to computer-use and refresh cache.
- Integrate into ActionAgent: check cache before model call; record success/failure to update cache. (TTL dropped; failure-driven invalidation instead.)

### M3.5: Run cache for plans and steps
- Add `src/core/run_cache.py` with deterministic run signatures (requirements + settings + viewport + models).
- Cache test plans and per-step action interpretations; reuse when screenshots match; invalidate on step/test failures and retry live.

### M4: Planner/Runner threading
- Add `desktop_mode` to plans/test cases; propagate through TestRunner → ActionAgent.
- Pre-run hooks:
  - Optional resolution downshift.
  - Initial desktop screenshot capture.
  - Window acquisition via a short computer-use goal (“bring LinkedIn Firefox window front and maximize”).
- Update planner prompts to prefer cached coordinates/visual targets (no DOM) and to skip Playwright setup in desktop mode.
- Thread run cache through planning/execution so repeat runs can skip re-interpretation unless a prior failure invalidated entries.

### M5: Model/SDK alignment
- Bump `openai` to latest 2.8.x; update client wrappers for Responses API defaults (truncation auto, verbosity/reasoning parameters).
- Update `settings.py` defaults/envs for gpt-5.1/gpt-5.1-mini and computer-use-preview model ids.

### M6: Test and runbook
- Add a LinkedIn POC scenario under `test_scenarios/` describing the four target steps.
- Smoke test script to exercise desktop mode end-to-end (manual verification expected). Scenario updated to send “Portate bien!” exactly.
- Runbook in `docs/runbook` covering prerequisites (uinput provisioning, single display, resolution toggle) and launch instructions.

## Risks & Mitigations
- uinput access: document manual setup; fail fast with clear error if missing.
- Multi-monitor: gate resolution change and coordinate cache by display count; warn and proceed native when >1.
- UI drift: rely on cache validation via screenshot hashes; quick fallback to computer-use when stale.
- Computer-use variance: keep tight action limits and stabilization waits; log traces for replay.

## Success Criteria
- Desktop mode executes the four LinkedIn steps visually with no DOM or Playwright usage.
- Coordinate cache reduces repeated computer-use calls on subsequent runs at the same resolution.
- Toggle-able integration that leaves existing browser path unaffected when desktop_mode is off. 

## Manual prerequisites (operator steps)
Follow before running the POC; none of these are automated:
- **Enable uinput** (root): `sudo modprobe uinput`; ensure persistence by adding `uinput` to `/etc/modules` or equivalent. If `/dev/uinput` permissions are restrictive, add your user to the appropriate group or set udev rules (e.g., `/etc/udev/rules.d/99-uinput.rules` with `KERNEL=="uinput", MODE="0660", GROUP="input"`), then reload udev and re-login.
- **Confirm single active display**: `xrandr --listmonitors` should show 1. If more than one, either disable extras or be aware the plan will skip resolution downshift and cache entries will be display-specific.
- **Resolution downshift readiness**: Verify your primary output supports 1920x1080@60 via `xrandr --query`. If unsupported, the run will keep native resolution.
- **Open and sign in**: Manually open Firefox (or chosen browser) with an authenticated LinkedIn session and a visible window. Leave it in a normal/maximizable state; the agent will bring it forward via computer-use.
- **Environment vars/keys**: Ensure `OPENAI_API_KEY` is set and that your account has access to `gpt-5.1`, `gpt-5.1-mini`, and `computer-use-preview`.
