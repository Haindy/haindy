# Visual-State + OpenAI Realtime Computer-Use Refactor

## Summary

- Refactor Haindy around a single visual-state pipeline that removes duplicate screenshot capture, adds provider-owned cartography on keyframes only, sends cropped patches when safe, and falls back to full-frame keyframes automatically when confidence drops.
- Scope decisions are locked: OpenAI realtime first, Google/Anthropic stay on request/response in v1; the new path is default-on; cartography is produced by the active CU provider; patch/keyframe logic applies to both execution and verification.
- This refactor stays strictly visual. No DOM, widget tree, AX tree, accessibility metadata, or other semantic platform state is introduced anywhere.

## Interfaces

- Add a CU-local transport interface under `src/agents/computer_use/*`: `ComputerUseTransport.start(...)`, `continue_turn(...)`, and `close()`. `ComputerUseSession` remains the orchestrator; only provider transport state moves behind the interface.
- Add visual-state types: `VisualFrame`, `CartographyMap`, `VisualTarget`, and `VisualFramePackage`. Extend `ComputerUseFollowUpBatch`, `ComputerUseSessionResult`, `EnvironmentState`, and `EnhancedActionResult` to carry `frame_kind`, `keyframe_id`, `parent_keyframe_id`, `patch_bbox`, `cartography`, and final provider-returned frame artifacts.
- Add CU-specific settings in `src/config/settings.py`: `HAINDY_OPENAI_CU_TRANSPORT` with `responses_websocket|responses_http`, `HAINDY_CU_VISUAL_MODE` with `keyframe_patch|legacy_full_frame`, `HAINDY_CU_CARTOGRAPHY_MODEL`, `HAINDY_CU_KEYFRAME_MAX_TURNS` default `3`, `HAINDY_CU_PATCH_MAX_AREA_RATIO` default `0.35`, and `HAINDY_CU_PATCH_MARGIN_RATIO` default `0.12`. Default mode is `responses_websocket` + `keyframe_patch`.
- Add one new dependency: `Pillow` for deterministic PNG decode, crop, encode, and diff support. Use existing `numpy` only where it helps local diffing.

## Implementation

### Phase 1: Unify screenshot ownership

- Make `TestRunnerArtifacts` the single owner of the latest visual state. It should store the current full keyframe, derived patch metadata, provenance, and persisted paths.
- Replace unconditional action-level `before` capture with “reuse the latest valid frame unless invalidated.” `TestRunnerExecutor` should use the last provider-returned `after` frame as the next decomposed action’s `before` frame instead of always calling `automation_driver.screenshot()`.
- Remove duplicate `after` recapture. `ActionAgent` should stop taking a fresh post-session screenshot when the provider session already produced the final frame, and `TestRunnerStepProcessor` should stop taking a separate step-level `after` screenshot unless no valid final frame exists.
- Preserve replay cache behavior. Execution replay remains driver-action-based and does not include cartography or patch artifacts in the replay key. Coordinate cache remains a secondary point-level optimization, updated from executed turns and invalidated on failed steps.

### Phase 2: Add cartography and keyframe/patch selection

- Cartography is generated only on keyframes by the active CU provider. Default cartography model is the active provider’s configured CU model; `HAINDY_CU_CARTOGRAPHY_MODEL` allows override without changing provider ownership.
- Cartography output schema is fixed: each target returns `target_id`, `label`, `bbox`, `interaction_point`, and `confidence`. Target IDs are stable only within a keyframe; cross-keyframe matching is best-effort using label + IoU and must not be treated as a durable semantic identifier.
- Keyframe refresh triggers are fixed: first frame of an action run; any scroll, navigation, or `reset_app`; any failed or non-executed action; any safety event; any patch decode/parse failure; any verifier full-frame fallback; and every `3` CU turns even if nothing else invalidates.
- Patch eligibility is fixed: the union of the current target bbox and the local changed-region bbox must fit within `35%` of the full-frame area. Patch margin is `12%` on each side, clamped to `24-160` pixels. If the union exceeds the threshold or no trustworthy local region exists, send a full keyframe.
- Local diff is screenshot-only and provider-independent. Use local full-frame capture to compute changed-region bounding boxes, but transmit cropped patches from the original-resolution pixels. Tiny deltas are not ignored; if the diff is too small to anchor reliably, use a target-centered crop instead of a delta-only crop.
- Patch mode is never used for `group_assert`, global navigation, `reset_app`, or multi-action steps that touched multiple far-apart regions. Those paths always use full keyframes.

### Phase 3: Add OpenAI realtime transport

- Implement `OpenAIResponsesHttpTransport` as the compatibility transport and `OpenAIResponsesWebSocketTransport` as the default transport. Both consume the same existing request builders and return the same normalized response shape the current mixins already expect.
- Do not move CU transport into `BaseAgent` or `OpenAIClient`. CU transport stays under `src/agents/computer_use/*` so it can manage persistent session state, event reassembly, and image-heavy follow-up turns without entangling planner/verifier APIs.
- Reuse the current OpenAI request semantics: initial turn stays task-only; follow-up turns continue to use `previous_response_id`; the only wire change is that OpenAI follow-up screenshots can now be full keyframes or patches rather than always full screenshots.
- WebSocket failure policy is fixed: on connection setup failure, protocol mismatch, event reassembly failure, timeout, or duplicate-response ambiguity, log the error, close the socket, switch the current action run to HTTP transport, and continue with the same visual-state package instead of aborting the whole test run.
- Google and Anthropic move behind the same transport interface but keep their current request/response behavior in v1. They must keep working on the old transport semantics while adopting the new visual-state packaging hooks.

### Phase 4: Refactor verification

- Verifier input selection becomes deterministic. For single-target, local-change steps, send cropped before/after patches plus machine-readable metadata text with original frame size, keyframe IDs, patch bbox, target bbox, and CU observation summary. For scroll/navigation/global-change/multi-region steps, send full before/full after keyframes.
- `assert` and `group_assert` default to full-frame verification unless a single, local, non-scrolling region was the entire subject of the step. This keeps verification conservative where missing context is most dangerous.
- The verifier continues to use the existing generic OpenAI path in v1; only its image selection logic changes. Realtime is scoped to computer-use transport, not to planner/interpreter/verifier text calls.
- Keep the current rule that transient CU observations remain strong evidence even if the final screenshot no longer shows the toast or banner; patch-mode verification should include those textual observations exactly as today.

### Phase 5: Ship default-on and stabilize

- Default the local runtime to the new behavior, but keep emergency fallbacks: `HAINDY_OPENAI_CU_TRANSPORT=responses_http` and `HAINDY_CU_VISUAL_MODE=legacy_full_frame`.
- The supplied validation command remains unchanged:

```bash
python -m src.main --mobile --plan test_scenarios/playerup/sign-in-test-plan-short.md --context test_scenarios/playerup/playerup_context.md
```

- After implementation, run targeted unit/integration tests first, then run the PlayerUp mobile command in the venv, and iterate fixing until the run completes without internal errors introduced by this refactor. App/test failures are acceptable; transport, screenshot, cartography, verifier, and artifact-pipeline crashes are not.
- Blocking point criteria are fixed: stop only if the remaining failure is external to the refactor, such as provider outage, emulator/runtime breakage, or app behavior unrelated to the new code. In that case, capture the last clean stack trace, the last model call logs, and the last persisted visual artifacts.

## Test Plan

- Add unit tests for local diff bbox detection, patch margin clamping, patch eligibility threshold, keyframe invalidation triggers, and cartography JSON parsing/normalization.
- Add CU transport tests for OpenAI WebSocket happy path, OpenAI WebSocket fallback to HTTP, duplicate event handling, socket close/reopen behavior, and unchanged Google request/response behavior.
- Add execution-pipeline tests proving screenshot reuse: no duplicate action `before` capture when a valid latest frame exists, no duplicate action/step `after` capture when the provider session already returned a final frame, and correct reuse of that frame in verification.
- Add verifier tests for patch-mode selection, full-frame fallback on global changes, `group_assert` full-frame behavior, and reuse of CU observations alongside cropped images.
- Keep existing OpenAI and Google CU regression suites green, especially guardrails, max-turn handling, safety handling, and follow-up payload tests.
- Acceptance criteria are fixed: the PlayerUp mobile run reaches normal test completion/failure reporting with no uncaught exceptions, no missing-artifact errors, no OpenAI transport/session errors, no patch/cartography parsing errors, and no verifier crashes.

## Assumptions

- The current local runtime already resolves `CU_PROVIDER=openai`, so the provided validation command will exercise the new default-on OpenAI realtime path.
- The interpreter stays full-frame and latest-screenshot based in v1; patch-awareness is intentionally limited to execution and verification.
- Cartography remains screenshot-derived only and does not attempt to become a semantic UI model beyond boxes, interaction points, and confidence.
- Google and Anthropic must retain current functional behavior in v1; only their screenshot packaging and transport abstraction seams change, not their end-to-end execution contract.
