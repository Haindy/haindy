# DRY and KISS Refactor Follow-Up Plan

## Status
Drafted on 2026-03-06.

## Relationship to Existing Plan
This document is a follow-up to `docs/plans/dry-kiss-refactor/completed/dry-kiss-refactor-plan.md`.
It narrows the work to the three highest-value improvements currently visible in the codebase and puts them in implementation order.

## Top 3 Improvements
1. Reduce `TestRunner` to orchestration-only responsibilities.
2. Remove `MessageBus` from coordinator control flow and keep only minimal diagnostics if still needed.
3. Centralize runtime bootstrap so agent creation and execution-context assembly are owned in one place.

## Why These 3 First
1. `TestRunner` is still the largest concentration of mixed responsibilities and the biggest source of change risk.
2. The coordinator/message-bus split adds indirection without runtime value in the current implementation.
3. CLI/bootstrap duplication is smaller, but it keeps agent wiring and execution metadata inconsistent across entrypoints.

## 1. Simplify TestRunner

### Current Problem
`src/agents/test_runner.py` still mixes orchestration, mutable execution state, screenshot persistence, replay validation, cache coordination, verification, and bug-report creation.

### Goal
Keep `TestRunner` as the coordinator for a test plan and move implementation detail into focused collaborators.

### Proposed Extractions
1. Extract replay execution and replay validation flow from `_try_execution_replay`.
2. Extract step artifact handling and screenshot persistence from `_execute_test_step` and related helpers.
3. Extract bug-report planning/context gathering and bug-report assembly into a dedicated collaborator.
4. Keep `TestRunner` responsible only for test-plan sequencing, step lifecycle, and collaborator composition.

### Candidate Targets
1. `src/runtime/execution_replay_service.py`
2. `src/agents/test_runner_artifacts.py`
3. New module such as `src/agents/test_runner_bug_reports.py`
4. New module such as `src/agents/test_runner_step_execution.py` if step flow remains too large

### Acceptance Checks
1. `TestRunner` no longer owns replay cache invalidation/store details directly.
2. `TestRunner` no longer writes screenshot/evidence artifacts directly outside artifact helpers.
3. Bug-report creation logic is isolated behind a typed helper/service.
4. `src/agents/test_runner.py` becomes materially smaller and reads as orchestration code.

### Risks
1. Hidden coupling through `self._current_*` mutable state.
2. Regressions in replay validation or artifact registration.
3. Step-result assembly drifting during extraction.

### Mitigation
1. Extract one concern at a time and preserve typed request/response objects.
2. Add focused tests around replay hit, replay invalidation, and screenshot registration behavior before moving code.

## 2. Remove MessageBus Ceremony

### Current Problem
`WorkflowCoordinator` publishes diagnostics events, but production code does not use subscriptions or queues for actual control flow.
The message bus currently behaves like an event log with extra abstraction cost.

### Goal
Use direct coordinator method calls for execution control.
If diagnostics history is still useful, retain a minimal event recorder instead of a general-purpose bus.

### Proposed Changes
1. Remove `MessageBus` dependency from `WorkflowCoordinator` initialization.
2. Replace `_publish_event` with a narrow diagnostics sink or simple in-memory event recorder.
3. Delete unused queue/subscription behavior from `src/orchestration/communication.py` if no longer needed.
4. Update coordinator status reporting to read from the new diagnostics surface instead of message-bus statistics.

### Acceptance Checks
1. Coordinator pause/resume/stop/status behavior works without `MessageBus`.
2. No production path depends on `register_agent`, `subscribe`, `unsubscribe`, or per-agent queues.
3. Coordinator initialization becomes smaller and easier to follow.
4. Remaining diagnostics API is obviously simpler than the current bus.

### Risks
1. Existing tests may overfit to bus internals instead of actual behavior.
2. Diagnostics consumers may silently rely on message statistics.

### Mitigation
1. Rewrite tests around coordinator behavior, not bus mechanics.
2. Preserve only the diagnostics fields that are actually surfaced to users or logs.

## 3. Centralize Runtime Bootstrap

### Current Problem
Agent construction and execution-context assembly are repeated in multiple places.
The CLI currently knows too much about agent wiring, setup payload structure, and coordinator composition.

### Goal
Create one owner for runtime bootstrap so configuration, agent creation, and execution context stay consistent.

### Proposed Changes
1. Add an `AgentFactory` or equivalent bootstrap helper for creating configured agents.
2. Add an execution-context builder for the planning/test context payload now assembled in `src/main.py`.
3. Update `src/main.py` and `src/orchestration/coordinator.py` to consume shared factories/builders instead of hand-rolling agent setup.
4. Keep environment resolution delegated to `src/runtime/environment.py`; do not reintroduce per-entrypoint branching.

### Candidate Targets
1. New module such as `src/runtime/agent_factory.py`
2. New module such as `src/runtime/execution_context_builder.py`
3. Possibly a single `src/runtime/bootstrap.py` if that stays small

### Acceptance Checks
1. Agent model config lookup and instantiation happen in one shared place.
2. Planning context and test execution context are assembled from shared helpers, not duplicated dict literals.
3. CLI entrypoint shrinks and loses direct knowledge of coordinator internals.
4. Coordinator initialization reuses the same agent/bootstrap path as the CLI.

### Risks
1. Over-abstracting small bootstrap code into an unclear framework.
2. Accidentally coupling CLI-only concerns into coordinator internals.

### Mitigation
1. Keep helpers data-oriented and explicit.
2. Prefer one or two narrow modules over a broad bootstrap layer.

## Recommended Order
1. Simplify `TestRunner` first.
2. Remove `MessageBus` ceremony second.
3. Centralize runtime bootstrap last.

## Why This Order
1. `TestRunner` is the highest-value simplification and reduces the largest maintenance hotspot.
2. Coordinator cleanup is easier once runner boundaries are clearer.
3. Bootstrap consolidation is safer after the main runtime responsibilities are already disentangled.

## Validation
1. Run `.venv/bin/ruff check .`
2. Run `.venv/bin/ruff format .`
3. Run `.venv/bin/mypy src`
4. Run `.venv/bin/pytest`
5. Add or update focused tests for:
   - `TestRunner` collaborator boundaries
   - coordinator pause/resume/stop/status flows without bus internals
   - shared agent/bootstrap construction

## Done Criteria
1. `TestRunner` reads as an orchestrator instead of a utility bucket.
2. Coordinator control flow no longer depends on unused bus abstraction.
3. Runtime bootstrap has one canonical owner for agent wiring and context assembly.
4. The refactor reduces branching and file size without changing runtime behavior.
