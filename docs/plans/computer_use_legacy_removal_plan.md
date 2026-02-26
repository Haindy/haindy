# Computer Use Legacy Removal Plan

## Status
Drafted on 2026-02-24.

## Locked Constraints
1. Keep `gpt-5.2` as orchestration model.
2. Keep `computer-use-preview` as OpenAI computer action model.
3. Support both providers: OpenAI and Google Gemini.
4. No provider fallback at runtime.
5. No backward compatibility for grid-era runtime behavior or schemas.
6. Remove all legacy grid code paths.

## Source Contracts
- OpenAI Computer Use guide: https://platform.openai.com/docs/guides/tools-computer-use
- OpenAI computer-use-preview model: https://platform.openai.com/docs/models/computer-use-preview
- OpenAI Responses API: https://api.openai.com/v1/responses
- Google Gemini Computer Use docs: https://ai.google.dev/gemini-api/docs/computer-use

## Objectives
1. Remove all grid-based planning/execution from runtime.
2. Enforce a single computer-use execution path.
3. Maintain feature parity between OpenAI and Google providers.
4. Align request/response loops with current official provider contracts.

## Non-Goals
1. Preserving old grid fields in reports/journals.
2. Maintaining environment toggles for legacy action execution.
3. Keeping legacy tests/demos/docs that depend on grid overlays.

## Current-State Findings (Verified)
1. Legacy grid runtime is active in action execution and desktop helpers.
2. Provider behavior is asymmetric across looping, safety, and follow-up semantics.
3. Google provider currently falls back to OpenAI in-session.
4. Computer-use-only mode is not enforced because flag/branching still exists.

## Target Architecture
1. One mandatory computer-use execution loop.
2. One canonical internal action/result contract used by both providers.
3. Provider adapter layer for request serialization and response normalization.
4. Provider-neutral telemetry schema for journaling/reporting.

## Implementation Phases

## Phase 1: Contract Freeze and Adapter Boundary
1. Define provider-agnostic internal types for:
- model action request
- action execution result
- safety event
- follow-up payload input
2. Define OpenAI adapter contract for:
- `computer_call` parsing
- `computer_call_output` follow-up
- `previous_response_id` chaining
- safety acknowledgment path
3. Define Google adapter contract for:
- function-call extraction
- function-response follow-up
- safety decision handling where applicable
4. Create parity matrix doc with one row per action type.

Self-check:
1. Every field in adapter contracts maps to official docs.
2. No provider-specific object leaks above adapter boundary.

## Phase 2: Enforce Computer-Use-Only Runtime
1. Remove Google->OpenAI fallback behavior.
2. Remove `actions_use_computer_tool` and `HAINDY_ACTIONS_USE_COMPUTER_TOOL` branching.
3. Remove `_should_use_computer_tool` split logic and make computer-use path mandatory.
4. Keep explicit terminal failure semantics when provider execution fails.

Self-check:
1. One selected provider per run, no runtime provider swap.
2. No legacy planner branch remains reachable.

## Phase 3: Remove Grid Runtime
1. Delete `src/grid/overlay.py` and `src/grid/refinement.py`.
2. Remove grid-dependent execution/prompt parsing paths from `ActionAgent`.
3. Remove grid-dependent desktop controller behavior.
4. Remove grid screenshot generation/highlighting utilities.

Self-check:
1. Runtime has no `GridOverlay` or `GridRefinement` imports.
2. Runtime has no `GRID_CELL` parsing path.

## Phase 4: Schema and Telemetry Cutover
1. Remove grid-shaped fields from core runtime models.
2. Remove grid-shaped fields from journal/report adapters.
3. Replace with provider-neutral action telemetry fields.
4. Update downstream consumers to new schema.

Self-check:
1. No `grid_coordinates`/`grid_screenshot_*` in runtime result schemas.
2. Journal/reporting reads only provider-neutral action telemetry.

## Phase 5: Provider Parity Hardening
1. Standardize action coverage and behavior across providers.
2. Standardize error classification and terminal codes.
3. Standardize loop detection and max-turn policy output surface.
4. Standardize safety handling surface in result metadata.

Self-check:
1. Same acceptance scenarios pass for both providers.
2. Any intentional differences are explicit and documented.

## Phase 6: Delete Legacy Tests, Demos, Docs
1. Remove grid-specific unit/integration tests.
2. Remove grid demos/examples.
3. Rewrite docs/runbooks as computer-use-only.
4. Update migration and architecture plans to reflect final state.

Self-check:
1. No docs describe legacy runtime toggles.
2. No demos reference grid overlays.

## Acceptance Gates (Must Pass)
1. `rg` zero-match gate for legacy runtime tokens:
- `GridOverlay`
- `GridRefinement`
- `GRID_CELL`
- `HAINDY_ACTIONS_USE_COMPUTER_TOOL`
- `actions_use_computer_tool`
2. `rg` zero-match gate for runtime fallback token:
- `falling back to OpenAI`
3. OpenAI path uses Responses computer-use flow with `computer_use_preview` and `computer_call_output`.
4. Google path uses official Gemini computer-use loop.
5. Parity matrix scenarios execute at same capacity across providers.

## Risk Register
1. Schema breakage in journals/reports due to grid field removal.
2. Hidden transitive imports of grid types from shared modules.
3. Behavioral drift between providers during adapter extraction.
4. Safety handling divergence if not normalized at adapter boundary.

## Mitigations
1. Perform adapter extraction before deleting runtime code.
2. Replace grid fields with explicit provider-neutral equivalents in one cutover.
3. Keep a strict parity matrix and require dual-provider scenario validation.
4. Treat provider failure as terminal and explicit, never silent.

## Execution Order Summary
1. Freeze contracts.
2. Remove fallback and legacy toggles.
3. Introduce adapters and switch runtime to canonical contract.
4. Delete grid runtime.
5. Cut over schemas and telemetry.
6. Remove legacy tests/examples/docs.
7. Enforce acceptance gates.
