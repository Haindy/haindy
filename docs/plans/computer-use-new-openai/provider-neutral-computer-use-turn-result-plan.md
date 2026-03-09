# Provider-Neutral Computer Use Turn Result Plan

## Status
Drafted on 2026-03-06.

## Relationship to Existing Plan
This document is a follow-up to [openai-gpt-5-4-computer-use-migration-plan.md](/home/fkeegan/src/haindy/haindy/docs/plans/openai-gpt-5-4-computer-use-migration-plan.md).
That migration moved the OpenAI path onto GPT-5.4.
This follow-up narrows the next problem: the three Computer Use providers still receive different amounts of post-action context, and the shared behavior tuning is not expressed in one provider-neutral model.

## Goal
Introduce one shared provider-neutral turn-result model for Computer Use execution and render it into the provider-specific follow-up payloads required by:
1. OpenAI
2. Google
3. Anthropic

## Non-Goals
1. Do not replace human-like CU interaction with non-CU launch or control paths.
2. Do not change the actual automation-driver behavior for click, type, scroll, drag, wait, or keypress.
3. Do not force all providers onto one wire format.
4. Do not broadly rewrite agent orchestration outside `src/agents/computer_use/`.

## Current Problem
The runtime behavior is only partially shared today.

1. Action execution is mostly shared, but follow-up payload construction is provider-specific and uneven.
2. Google receives richer structured action-result feedback in follow-ups:
   - `status`
   - `url`
   - `x/y`
   - clipboard/error fields
   - screenshot inline with the function response
3. OpenAI currently receives a thinner follow-up contract:
   - `computer_call_output`
   - screenshot payload
   - reminder/error text as separate user input
4. Anthropic currently receives yet another variant:
   - `tool_result`
   - screenshot-only content on success
   - text-only error content on failure
5. Provider-specific follow-up builders now carry behavior differences that should be shared semantically, even when the final rendering must stay provider-specific.

## Why This Matters
1. Shared CU tuning currently lands in whichever provider path happened to need it first.
2. OpenAI, Google, and Anthropic are not benefiting equally from the same action-result semantics.
3. Debugging is harder because execution meaning is spread across three payload builders.
4. The current shape encourages drift: one provider can become more grounded simply because its adapter exposes more result detail.

## Design Principles
1. DRY: one canonical internal description of what happened after a CU action.
2. KISS: keep provider adapters thin and declarative.
3. Preserve legality: each provider still gets the exact wire format its API expects.
4. Preserve human-like interaction: the refactor changes result transport, not the interaction model.
5. Prefer additive internal structure over clever abstraction. The shared model should be explicit and typed.

## Proposed Target Architecture

### 1. Shared Turn Result
Create one provider-neutral turn-result object assembled immediately after `_execute_tool_action()` and snapshot capture.

Candidate fields:
1. `call_id`
2. `action_type`
3. `status`
4. `screenshot_base64`
5. `x`
6. `y`
7. `start_x`
8. `start_y`
9. `end_x`
10. `end_y`
11. `duration_ms`
12. `current_url`
13. `clipboard_text`
14. `clipboard_truncated`
15. `clipboard_error`
16. `error_message`
17. `pending_safety_checks`
18. `acknowledged_safety_checks`
19. `provider_metadata`
20. `observation_text`

`observation_text` should stay short and optional. It is not a second evaluator. It is only a compact provider-neutral summary of what the harness knows happened, for example:
- `Action executed successfully. Latest screenshot attached.`
- `Action execution failed: <error>.`

### 2. Shared Renderer Input
Build one shared renderer input object from the executed turns in a model turn.

This should answer:
1. What actions were executed?
2. What is the latest screenshot state?
3. What execution errors occurred?
4. What short reminder should the model receive for execute vs observe-only mode?
5. What structured metadata is safe and useful to pass back?

### 3. Thin Provider Renderers
Each provider should only convert the shared turn-result model into its own wire format.

OpenAI renderer:
1. Render legal `computer_call_output` items.
2. Attach screenshot in the required OpenAI shape.
3. Add extra grounded context as separate `input_text` items when needed.
4. Keep provider-specific safety acknowledgement handling.

Google renderer:
1. Render `FunctionResponse` payloads.
2. Include structured fields already supported by the Google path.
3. Source those fields from the shared turn-result object instead of ad hoc provider assembly.

Anthropic renderer:
1. Render `tool_result` blocks.
2. Include screenshot on success.
3. Include execution error text on failure.
4. Add minimal structured grounding text where allowed, sourced from the shared turn-result object.

## Proposed File Boundaries
1. Add a new shared module, likely one of:
   - `src/agents/computer_use/turn_result.py`
   - `src/agents/computer_use/provider_feedback.py`
2. Keep action execution in:
   - `src/agents/computer_use/action_mixin.py`
3. Keep provider loops in:
   - `src/agents/computer_use/openai_mixin.py`
   - `src/agents/computer_use/google_mixin.py`
   - `src/agents/computer_use/anthropic_mixin.py`
4. Keep low-level shared helpers in:
   - `src/agents/computer_use/support_mixin.py`
   - `src/agents/computer_use/common.py`

## Implementation Plan

## Phase 1: Define the Shared Model
Purpose: create one internal contract before changing any provider rendering.

Tasks:
1. Add a typed provider-neutral turn-result structure.
2. Add a typed provider-neutral batch/follow-up structure for one model turn.
3. Move shared reminder/error-summary assembly into shared helpers.
4. Keep provider-specific raw metadata in a single escape hatch field instead of leaking it into the shared schema.

Self-check:
1. The new types are simple enough to understand without reading provider code.
2. There is one obvious place to look for post-action semantics.

## Phase 2: Centralize Turn-Result Assembly
Purpose: make all providers use the same post-action facts.

Tasks:
1. Build the shared turn-result object from `ComputerToolTurn` after `_record_turn_snapshot()`.
2. Centralize extraction of:
   - screenshot
   - current URL
   - coordinates
   - clipboard state
   - execution status
   - error text
3. Add a shared helper that converts executed turns into one provider-neutral follow-up batch.
4. Ensure loop detection, logging, and existing `ComputerToolTurn` journaling continue to work unchanged.

Self-check:
1. The same executed action yields the same semantic result object regardless of provider.
2. No provider builds action-result context directly from raw `turn.metadata` unless strictly necessary.

## Phase 3: Re-render OpenAI from the Shared Model
Purpose: fix the thinnest provider path first.

Tasks:
1. Rewrite the OpenAI follow-up builder to consume the shared turn-result batch.
2. Keep the legal OpenAI wire format:
   - `computer_call_output`
   - screenshot payload
   - allowed companion text inputs
3. Move OpenAI reminder/error text generation to the shared layer where possible.
4. Keep OpenAI-specific safety acknowledgement handling in the renderer.

Self-check:
1. OpenAI renderer reads like serialization, not business logic.
2. Provider-neutral semantics are visible in the rendered follow-up, even if OpenAI must receive them via screenshot plus text.

## Phase 4: Re-render Google from the Shared Model
Purpose: preserve Google behavior while removing adapter-local semantic drift.

Tasks:
1. Rewrite the Google follow-up builder to source all structured fields from the shared turn-result batch.
2. Keep Google-specific `FunctionResponse` mechanics and ID correlation logic.
3. Preserve current useful Google grounding:
   - `status`
   - `url`
   - `x/y`
   - clipboard/error fields
4. Remove duplicated reminder/error assembly from the Google adapter.

Self-check:
1. Google keeps its current grounding quality.
2. The Google adapter becomes thinner, not more abstract.

## Phase 5: Re-render Anthropic from the Shared Model
Purpose: bring Anthropic into the same semantic model as OpenAI and Google.

Tasks:
1. Rewrite Anthropic follow-up building to consume the shared turn-result batch.
2. Preserve Anthropic `tool_result` mechanics.
3. Use the shared result model to decide:
   - screenshot content on success
   - text error content on failure
   - short shared reminder text if appropriate
4. Keep Anthropic action translation separate from turn-result rendering.

Self-check:
1. Anthropic becomes a pure renderer of shared turn-result semantics plus its own action-translation layer.
2. Success and failure follow-ups are explained consistently with the other providers.

## Phase 6: Remove Drift and Tighten Tests
Purpose: lock in the shared architecture.

Tasks:
1. Delete duplicated provider-local helpers once the shared layer owns them.
2. Add tests that compare provider-neutral batch creation independently from provider rendering.
3. Add renderer tests for:
   - OpenAI
   - Google
   - Anthropic
4. Add regression coverage for:
   - successful click/wait turns
   - failed action turns
   - observe-only reminders
   - safety acknowledgements
   - multi-action batches
   - multi-call turns

Self-check:
1. The same synthetic executed turn can be rendered by all three providers.
2. New tuning can be added once in the shared layer and reflected everywhere.

## Risks
1. Over-designing the shared model and making adapters harder to read.
2. Accidentally erasing provider-specific information that genuinely matters.
3. Smuggling provider-specific quirks into the shared core until it stops being provider-neutral.
4. Breaking logging or loop detection by moving screenshot ownership around.

## Mitigation
1. Keep the shared model minimal and action-result oriented.
2. Preserve a `provider_metadata` escape hatch instead of polluting the common fields.
3. Require each provider adapter to stay mostly serialization-only.
4. Keep `ComputerToolTurn` as the execution record of truth; the shared turn-result is a rendering model, not a replacement for execution history.

## Validation
1. Run `.venv/bin/ruff check .`
2. Run `.venv/bin/ruff format .`
3. Run `.venv/bin/mypy src`
4. Run `.venv/bin/pytest`
5. Add focused tests for:
   - provider-neutral turn-result construction
   - provider-neutral follow-up batch construction
   - OpenAI rendering from shared turn results
   - Google rendering from shared turn results
   - Anthropic rendering from shared turn results

## Done Criteria
1. OpenAI, Google, and Anthropic all render follow-ups from the same provider-neutral turn-result model.
2. Provider adapters are thin and mostly declarative.
3. Shared CU tuning no longer has to be copied three times.
4. The refactor improves grounding consistency without changing the human-like interaction contract.
