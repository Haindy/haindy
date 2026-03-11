# Execution-Replay Fallback Clean Retry

## Summary

- Fix execution replay fallback so a failed cached action sequence becomes a real retry from the replay-failure state, not a continuation on a mutated UI with stale planning context.
- The retry must capture a fresh full screenshot at the moment replay validation fails, invalidate the replay cache entry, increment the step attempt, and force fresh interpretation using that new screenshot.
- This is primarily a test-runner / replay-orchestration fix. It does not change provider APIs.

## Problem Context

### Latest failing run

- Run: `20260310T094259Z_2a571ecf`
- Failure: `TC001` step 3, `Type "qaagent@playerup.co" in the Email field.`
- Main artifact bundle:
  - `reports/20260310T094259Z_2a571ecf/a6b78d78-a326-4397-813a-26cda20c95d7_20260310_094557-actions.json`
  - `data/traces/20260310T094259Z_2a571ecf.json`
  - `debug_screenshots/20260310T094259Z_2a571ecf/`

### What actually happened

- Step 3 hit an execution replay cache entry with `17` cached driver actions.
- The trace recorded:
  - `execution_replay_cache_hit`
  - `execution_replay_cache_invalidate` with reason `validation_failed`
  - `execution_replay_fallback_to_cu`
- The replayed `17` actions exactly matched the prior cached step-3 action trace from:
  - `reports/20260310T001713Z_207fe7e7/d219a42b-29c5-4aa4-9d27-9bb379ed9204_20260310_002207-actions.json`
- By the time live Google CU started, the initial screenshot already showed a duplicated email field:
  - `data/model_logs/screenshots/computer_use_initial_20260310_094438_731996.png`
- Google then operated on an already-corrupted field, not on the original clean step state.

### Current replay fallback behavior

- Replay failure invalidates the replay cache entry and returns `fallback_to_cu=True`.
- The step processor interprets that as “continue with normal step execution now”.
- The system does **not** restore the UI to the pre-replay state.
- The system does **not** increment the step attempt counter.
- The system does **not** force fresh interpretation from the replay-failure screenshot.
- Result: live CU resumes on a mutated screen while still conceptually being in the same attempt.

## Why this matters

- The current fallback is not a real retry.
- The interpreter does not get a truthful new “before” image that reflects the damaged UI after replay failure.
- That prevents the runner from planning cleanup or restoration actions before re-attempting the original step.
- This especially hurts Google CU because it is more iterative on follow-up turns than OpenAI, so it is more likely to keep acting in a dirty state instead of concluding immediately.

## OpenAI vs Google Insight

- Both providers share the same low-level text-entry executor in `src/agents/computer_use/action_mixin.py`.
- OpenAI tends to succeed here because it usually completes the step in one atomic action and stops.
- Google is more turn-heavy and therefore exposes orchestration weaknesses more often:
  - bad replay artifacts
  - continuing from dirty UI after replay fallback
  - provider-side input blocking after extra text-entry turns
- The replay cache key is currently provider-agnostic, so a messy provider-specific action trace can poison future reruns for the same step.

## Target Behavior

When replay validation fails for a step:

1. Capture a full screenshot immediately at failure time.
2. Invalidate the replay cache entry.
3. Increment the step attempt counter.
4. Force fresh interpretation with `use_cache=False`.
5. Use the new replay-failure screenshot as the active `screenshot_before` for that retry.
6. Start a fresh step session from that new screenshot and execute the newly interpreted actions.

This allows the interpreter to plan:

- cleanup of corrupted UI state
- restoration to the expected starting point
- then execution of the intended step

## Non-Goals

- Do not redesign the replay cache storage format in this pass.
- Do not add provider-specific replay keys in this pass.
- Do not change Google/OpenAI provider request payloads in this pass.
- Do not fix Android `type_text_at` clearing behavior in this pass.

## Implementation Plan

### 1. Extend replay fallback result

- Add replay-failure screenshot bytes/path to `ReplayExecutionResult`.
- When replay validation fails, capture and return the full failure screenshot alongside `fallback_to_cu=True`.
- Keep the existing replay action record so reporting still shows the replay attempt.

### 2. Make fallback a real retry in step processor

- In `TestRunnerStepProcessor.execute_step(...)`:
  - detect replay fallback with screenshot data
  - replace the active `screenshot_before` and `step_result.screenshot_before` with the replay-failure screenshot
  - increment `attempt`
  - bypass plan cache for the next `_interpret_step(...)` call
- Do not return from replay fallback straight into live CU on the same attempt state.

### 3. Force fresh interpretation

- Ensure the retry path calls `_interpret_step(..., use_cache=False)`.
- The retry should behave like a new planning pass over the replay-failure screenshot.
- This should let the interpreter emit recovery actions if the UI is already damaged.

### 4. Preserve reporting clarity

- Keep the replay attempt visible in step artifacts and trace events.
- Make the final step record show:
  - replay hit
  - replay invalidation
  - fallback retry
  - new attempt number

## Code Areas

- `src/runtime/execution_replay_service.py`
  - replay failure handling
  - replay result payload
- `src/agents/test_runner_step_processor.py`
  - replay fallback control flow
  - attempt increment
  - forcing fresh interpretation
  - replacing the active screenshot for retry
- `src/runtime/trace.py`
  - only if extra replay-retry events need to be recorded
- `tests/test_test_runner_replay_execution.py`
- `tests/test_test_runner_step_processor.py`

## Test Plan

- Add a replay-fallback unit test proving:
  - replay validation failure invalidates the replay cache
  - a fresh failure screenshot is captured
  - the next interpretation uses that new screenshot
  - `attempt` increments before live execution continues
  - `use_cache=False` is enforced on the retry
- Add a step-processor test proving the fallback retry does not reuse the old pre-replay screenshot.
- Add a test proving the replay action record is still preserved in reporting even when fallback retries live CU.

## Acceptance Criteria

- After replay validation failure, live CU no longer starts from the stale replay-mutated UI without reinterpretation.
- The interpreter receives the replay-failure screenshot as the new planning context.
- The next step execution is recorded as a new attempt.
- Existing replay success cases remain unchanged.

## Follow-Up Work

- Consider adding provider to the replay cache key.
- Consider tightening what kinds of “passed” driver traces are eligible for replay storage.
- Consider fixing `mobile_adb` `type_text_at` so `clear_before_typing=True` actually clears text deterministically instead of relying on triple-tap selection.
