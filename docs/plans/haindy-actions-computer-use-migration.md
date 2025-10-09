# Haindy Actions: Migration to OpenAI Computer Use Tool

## Summary
Vision-first plan to replace the Actions agent's custom grid planner with OpenAI's Computer Use tool (via the Responses API) while keeping Playwright as the actuator and preserving Haindy's existing Planner → Runner → Actions architecture.

## Today's Date
October 9, 2025 (Europe/Madrid)

## Background & Goals
- Current Actions agent converts Runner instructions into grid-based coordinates using bespoke prompts and refinement; reliability varies with DOM shifts and visual noise.
- OpenAI's `computer-use-preview` model adds first-party planning for GUI interactions using screenshots and simulated mouse/keyboard actions.
- Goal: delegate planning to the Computer Use tool, keep execution in Haindy/Playwright, and maintain self-contained orchestration without ChatGPT Agent Mode or MCP dependencies.

## Constraints & Non-Goals
- **Must** remain DOM-independent; all clicks/scrolls use screen coordinates derived from screenshots.
- **Must** keep Playwright for action execution and screenshot capture across Chromium, Firefox, WebKit, headful CI containers, and developer laptops.
- **Must** detect `pending_safety_checks` and fail fast without continuing automated execution.
- **Must not** rely on hosted Agent Mode, hosted Operator, or MCP connectors.
- **Non-goals:** redesigning Planner/Runner logic, adopting DOM selectors, or adding new third-party automation frameworks.

## References
- OpenAI, *Computer Use Tool Guide* (retrieved Oct 9, 2025; beta guidance for `computer-use-preview`, safety checks, action loop).
- OpenAI, *Responses API Reference* (retrieved Oct 9, 2025; request/response schema, tool configuration, `truncation="auto"`, `previous_response_id`).
- OpenAI, *New Tools for Building Agents* announcement (Sept 25, 2025 release; positions Responses API + built-in tools as production surface, pricing per standard token/tool rates).
- OpenAI Agents SDK Docs, *Tools* overview (Published Oct 8, 2025; clarifies hosted tool behavior, including ComputerTool parameters and observability hooks).
- OpenAI, *Computer-Using Agent Overview* (May 20, 2025 research preview; context on model capabilities, human confirmation expectations, benchmark performance).

## Gating Requirements
- Confirm organization/project has access to `computer-use-preview` (beta) and agree to preview usage terms; monitor published rate limits and pricing (standard token billing plus per-call tool charges).
- Requests **must** set `truncation="auto"` and include the `computer_use_preview` tool with accurate `display_width`/`display_height` matching the active Playwright viewport; coordinate system is pixel-based with origin at top-left.
- Provide PNG screenshots encoded as `data:image/png;base64,...` (RGB) in both initial `input_image` and follow-up `computer_call_output`.
- `environment` parameter must be `"browser"` to align with Playwright context; set optional `current_url` each turn to improve safety checks.
- Supply stable `safety_identifier` per Responses API guidance; enforce action budgets to stay within preview quotas and monitor x-ratelimit headers for throttling.

## Architecture
### Current
- **Planner** agent decomposes goals into `TestPlan`/`TestStep` objects.
- **Runner** agent iterates steps, delegating each to **Actions** agent.
- **Actions** agent:
  - Generates grid overlay from Playwright screenshot.
  - Prompts OpenAI (text/vision) to pick cell coordinates, refines to pixel positions.
  - Executes clicks/typing via Playwright, captures after-state screenshots, logs metrics.
  - Returns `EnhancedActionResult` (validation, coordinate metadata, execution logs, screenshots).

### Target
- **Planner & Runner** continue unchanged, still producing `TestStep` inputs.
- **Actions** agent responsibilities split:
  - **Planning**: delegate to Responses API with `computer-use-preview` tool; maintain conversation state via `previous_response_id`.
  - **Execution**: continue using Playwright to realize each model-issued action (click, scroll, type, wait, screenshot).
  - **State capture**: send current screenshot (PNG base64) and optional `current_url` back to model as `computer_call_output` each turn.
- Internal orchestrator manages loop: screenshot → tool request → action execution → new screenshot until no `computer_call` items remain.
- Additional supervisor layer handles safety acknowledgements, error recovery, and fallback to legacy grid planner.

## Message Contracts
- **Runner → Actions (input)**: existing `TestStep`, `test_context`, optional pre-screenshot unchanged. Add optional `action_budget` and `safety_level` hints for Computer Use loop.
- **Actions internal → Responses API**:
  - Initial call: `model="computer-use-preview"`, `tools=[{"type":"computer_use_preview","display_width":W,"display_height":H,"environment":"browser"}]`, `input` containing system framing + user goal + latest screenshot (base64 PNG).
  - Each follow-up: `previous_response_id`, `input=[{"type":"computer_call_output","call_id":...,"output":{"type":"computer_screenshot","image_url":"data:image/png;base64,..."},"current_url":... ,"acknowledged_safety_checks":[...] }]`, `truncation="auto"`.
- **Actions → Runner (output)**: extended `EnhancedActionResult` capturing
  - Sequence of `ComputerCall` actions (type, coordinates, args, timestamps, execution status, Playwright result).
  - Final reasoning summary/message from model when available.
  - All screenshots (before/after, per-turn) persisted to `debug_screenshots/` with trace IDs.
- Safety events (pending checks, automatic failures).
  - Failure metadata (API errors, execution errors, fallback usage).

## Execution Loop
1. Runner requests Actions to execute `TestStep`.
2. Actions captures or receives initial screenshot, normalizes viewport metrics (`display_width/height`, DPI if needed).
3. Actions sends Responses API call with goal text + screenshot.
4. Upon receiving `computer_call`:
   - Parse `action` dict (e.g., `click`, `double_click`, `scroll`, `type`, `keypress`, `wait`, `screenshot`).
   - Validate coordinates are within viewport, optionally clip to `[0, W/H)`.
   - Execute via Playwright (cursor move, click, keyboard type, scroll).
   - Capture Playwright artifacts (DOM snapshot optional, network/console logs if configured).
5. Sleep configurable stabilization delay; take fresh screenshot (PNG).
6. Post screenshot back via `computer_call_output`, including `current_url`, optional text status (e.g., detected validation error), and any `acknowledged_safety_checks`.
7. Repeat until model response lacks `computer_call` or exceeds budget.
8. Collate final model message or status; return aggregated result to Runner.

## Safety & Confirmation Handling
- Surface any `pending_safety_check` items (codes `malicious_instructions`, `irrelevant_domain`, `sensitive_domain`) via Runner logs; automatically treat them as hard failures unless configuration explicitly allows acknowledgement.
- Runner enforces allowlist/blocklist for domains, keystrokes, clipboard use per doc guidance; abort if violation.
- When `computer-use-preview` suggests sensitive operations (credential entry, destructive actions), Runner marks the step failed and short-circuits remaining actions.
- Record safety check metadata alongside step results for auditability even without human intervention.

## Observability & Artifacts
- Persist per-turn screenshots, annotated with attempted action and response IDs, under `debug_screenshots/<test_run>/<step>/turn_<n>.png`.
- Capture structured trace (JSONL) containing:
  - Request/response metadata (response IDs, latency, tokens, action payloads, safety flags).
  - Playwright execution outcome (success, exceptions, durations).
  - Network/console logs (via Playwright tracing) when enabled.
- Summaries stored in `reports/` linking to screenshots, actions, and safety outcomes.

## Error Handling & Retries
- **API failures (HTTP, rate limits):** exponential backoff with jitter, bounded retries; if persistent, fall back to legacy grid planner and tag result as degraded.
- **Invalid action payloads:** clip coordinates, map unsupported action types to best effort (e.g., treat `double_click` as double `click`), or request correction via `computer_call_output` message describing error.
- **Execution failures:** capture exception from Playwright, include in next `computer_call_output` as `output":{"type":"input_text","text":"Execution failed: ..."}`, prompting model to re-plan.
- **Loop guardrails:** max tool turns per step, cumulative timeout, and screenshot diff detection to avoid infinite loops.

## Test Plan
- **Happy paths:**
  - Wikipedia search regression scenario (existing JSON) across Chromium, Firefox, WebKit.
  - Multi-step form fill (navigate, type, click submit, validate text).
- **Edge cases:**
  - Prompt injection page with malicious banner to ensure safety check handling.
  - Slow-loading page forcing waits/timeouts.
  - Scroll-heavy page requiring multiple scroll actions.
  - Modal dialogs/pop-ups requiring precise coordinates.
- **Multi-browser coverage:** execute Phase 0 spike flows on headful Chromium/Firefox/WebKit locally and in CI container; verify viewport alignment and coordinate accuracy.
- **Regression:** run `pytest -m "not slow"` plus targeted integration tests that stub Responses API (fixtures) to simulate computer_call loops.

## Acceptance Criteria
- Actions agent completes defined spike flows end-to-end using Computer Use planner in all three browsers without manual retries.
- Safety acknowledgements are surfaced and gated per policy.
- Artifacts (screenshots, traces) are generated and linked in action results.
- Fallback toggle allows returning to legacy grid planner without code changes to Planner/Runner.
- Unit/integration tests cover action-loop orchestrator, safety handling, and fallback switch.

## Phase 0 Spike Plan
1. Implement minimal Actions prototype invoking `computer-use-preview` with fixed viewport (1024×768) and manual Playwright wiring.
2. Validate two flows end-to-end:
   - Search + result click on Wikipedia.
   - Form input + submission on example site (e.g., playwright.dev contact).
3. Run flows on Chromium, Firefox, WebKit in headful mode (CI container + local) ensuring coordinate fidelity and screenshot loop.
4. Instrument basic logging of actions and screenshots.
5. Exit criteria to graduate to Pilot:
   - ≥90% action success over 10 consecutive runs per browser.
   - Safety checks surfaced at least once and correctly gated.
   - Recorded performance metrics (avg actions/step, latency) to baseline future tuning.

## Rollout Plan
- **Phase 0 (Spike):** isolated prototype behind feature flag, manual runs only.
- **Phase 1 (Pilot):** enable for select scenarios (Wikipedia, smoke tests) with enhanced monitoring; capture metrics, tune timeouts.
- **Phase 2 (Beta rollout):** expand to broader scenario set once pilot meets SLA; run side-by-side with legacy planner to compare outcomes.
- **Phase 3 (Full migration):** default to Computer Use for all actions; retain legacy fallback flag for one release cycle before deprecation.
- **Phase 4 (Retrospective):** document learnings in `GRID_TEST_PROGRESS.md`, update docs, and tune Playwright instrumentation.

## Risks & Mitigations
- **Model availability / rate limits (preview status):** Monitor quota dashboards; implement graceful degradation to legacy planner.
- **Coordinate drift due to viewport mismatch:** Enforce single source of truth for viewport (Playwright `page.viewport_size`), update request payload each turn.
- **Latency & cost increase:** Track per-turn tokens/actions; add caching for static instructions; consider summarizing loops via `reasoning.summary="concise"` to debug without extra turns.
- **Prompt injection / unsafe actions:** Strict domain allowlist, automatic failure on suspicious actions, carry `safety_identifier` per user run, leverage safety checks.
- **Browser compatibility gaps:** Pre-flight cross-browser tests; add per-browser overrides if model struggles (e.g., Firefox UI differences).
- **Automated shutdown on safety events:** Because we skip manual approvals, ensure Runner clearly reports failed steps so humans can review afterwards.

## Fallback & Revert
- Feature flag (`HAINDY_ACTIONS_USE_COMPUTER_TOOL=true/false`) toggles between Computer Use loop and legacy grid planner at runtime.
- Retain legacy grid code paths and tests; ensure new orchestration wraps old implementation for quick swap.
- Revert procedure: set flag false, redeploy; if recovery requires code revert, roll back commit introducing Computer Use integration (documented in PR) using standard git revert.

## Open Questions
- **Do we need additional account configuration (e.g., project-level feature enablement) for `computer-use-preview` beyond default access?** (Resolved: no extra configuration required.)
- Should we stream Responses API events for realtime UI, or poll per turn?
- What automated heuristics should trigger Runner to abort future actions within the same test case?

## Next Steps
1. Confirm account-level availability/quota for `computer-use-preview` and update secrets management (no additional configuration needed, but verify quotas).
2. Build minimal Phase 0 prototype with feature flag and logging.
3. Implement stubbed integration tests for computer-action loop.
4. Prepare rollout dashboards (actions per step, success rate, token usage) to track pilot.
