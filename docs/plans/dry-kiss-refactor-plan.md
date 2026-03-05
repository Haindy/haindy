# DRY and KISS Refactor Plan

## Status
Drafted on 2026-03-05.

## Objectives
1. Centralize environment/backend semantics so `desktop`, `browser`, and `mobile_adb` are resolved in one place.
2. Reduce `TestRunner` to a coordination role and move execution mechanics into smaller collaborators.
3. Remove orchestration ceremony that does not provide runtime value.

## Non-Goals
1. Changing provider behavior or prompt contracts as part of the initial refactor.
2. Reworking report schemas beyond what is required to preserve behavior after extraction.
3. Introducing compatibility shims for old internal APIs unless a migration step strictly requires them.

## Current-State Findings
1. Environment normalization is duplicated across settings, action execution, computer-use session handling, and test execution flow.
2. `TestRunner` mixes orchestration, screenshot management, AI interpretation, action execution, replay caching, evidence handling, and summary generation.
3. The message bus adds structure around coordinator flows, but the subscribed handlers are stubs and the coordinator still performs direct orchestration.

## Principles
1. One owner per concept.
2. Prefer small helpers with clear contracts over multi-purpose classes.
3. Preserve behavior while reducing branching and file size.
4. Make validation easy: each phase must have explicit acceptance checks.

## Phase 1: Centralize Environment Semantics
1. Create a shared module, for example `src/runtime/environment.py`, with the canonical environment contract.
2. Move these decisions into that module:
- environment normalization
- target-type derivation
- backend alias handling
- coordinate-cache-path selection
- provider-facing environment mapping where needed
3. Replace duplicated logic currently spread across:
- `src/config/settings.py`
- `src/agents/action_agent.py`
- `src/agents/computer_use/session.py`
- `src/agents/test_runner.py`
- `src/main.py`
4. Add focused unit tests for normalization and mapping behavior.

Acceptance checks:
1. Runtime code no longer hand-rolls `desktop`/`browser`/`mobile_adb` string normalization outside the shared module.
2. Existing desktop and mobile flows keep the same effective behavior.
3. Tests cover aliases such as `web`, `browser`, `mobile`, `android`, and invalid fallback values.

## Phase 2: Extract TestRunner Collaborators
1. Keep `TestRunner` as the orchestration entrypoint only.
2. Extract step interpretation logic into a dedicated collaborator, for example `src/agents/test_runner_interpreter.py`.
3. Extract action execution and action-result assembly into a dedicated collaborator, for example `src/agents/test_runner_executor.py`.
4. Extract replay cache handling and related invalidation/store logic into a dedicated collaborator, for example `src/runtime/execution_replay_service.py`.
5. Extract evidence/screenshot/report helper behavior into a smaller artifact helper if it remains materially coupled.
6. Keep collaborator contracts narrow:
- input models only
- return typed results
- no direct writes into `TestRunner` private state except through explicit return values

Acceptance checks:
1. `TestRunner` becomes primarily a coordinator and no longer owns screenshot persistence, replay cache internals, and action execution details directly.
2. `TestRunner` size drops materially from its current 3k+ lines.
3. Existing tests for planning, execution, replay, and reporting continue to pass without behavioral regressions.

## Phase 3: Simplify Coordinator Orchestration
1. Audit every `MessageBus` usage and separate diagnostics from control flow.
2. Remove publish/subscribe paths that only target stub handlers.
3. If event logging is still useful, keep a thin event sink or event recorder API instead of a general-purpose bus.
4. Convert coordinator internals to direct method calls for planning, execution, pause, resume, stop, and status transitions unless asynchronous decoupling is actually needed.
5. Keep externally visible behavior unchanged.

Acceptance checks:
1. Coordinator no longer publishes to handlers that do nothing.
2. The message bus is either deleted or reduced to a minimal diagnostics/event facility.
3. Coordinator initialization becomes smaller and easier to follow.

## Recommended Execution Order
1. Phase 1 first because it is the smallest, safest, and removes active logic drift.
2. Phase 2 second because the shared environment contract reduces extraction churn inside `TestRunner`.
3. Phase 3 last because orchestration simplification is easier once runner boundaries are clear.

## Work Breakdown

### Step 1
1. Add the shared environment module and tests.
2. Migrate settings, main entrypoint, action agent, computer-use session, and test runner to use it.
3. Remove obsolete normalization helpers after callers are cut over.

### Step 2
1. Extract `TestRunner` interpretation code without changing prompt content.
2. Extract action execution/result assembly without changing action schemas.
3. Extract replay cache logic without changing cache keys or persistence format.
4. Run full regression tests after each extraction slice.

### Step 3
1. Remove stub subscription handlers.
2. Replace unnecessary bus-mediated calls with direct calls.
3. Retain only the minimum event reporting surface needed by logs or diagnostics.

## Risks
1. Environment behavior drift between settings validation and runtime resolution.
2. Hidden coupling inside `TestRunner` private state during collaborator extraction.
3. Replay cache regressions if key generation or invalidation timing changes.
4. Accidental removal of message-bus behavior that tests do not currently cover.

## Mitigations
1. Freeze current environment behavior with tests before broad cutover.
2. Extract one `TestRunner` concern at a time and keep interfaces typed.
3. Preserve existing replay cache key construction until after the extraction stabilizes.
4. Add direct coordinator tests around pause/resume/stop/status flows before removing bus behavior.

## Validation
1. Run:
- `.venv/bin/ruff check .`
- `.venv/bin/ruff format .`
- `.venv/bin/mypy src`
- `.venv/bin/pytest`
2. Add targeted tests for:
- environment normalization and mapping
- `TestRunner` collaborator boundaries
- coordinator direct orchestration behavior
3. Perform one desktop and one `mobile_adb` smoke scenario after Phase 1 lands.

## Done Criteria
1. There is exactly one canonical place to resolve runtime environment semantics.
2. `TestRunner` is substantially smaller and reads as orchestration code rather than a utility bucket.
3. Coordinator orchestration no longer depends on a general-purpose message bus unless that indirection has a verified need.
4. All required validation commands pass.
