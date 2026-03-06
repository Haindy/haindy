# OpenAI GPT-5.4 Computer Use Migration Plan

## Status
Drafted on March 6, 2026.

## Goal
Update HAINDY's OpenAI computer-use implementation from the legacy preview integration to the current GPT-5.4 built-in computer tool.

Out of scope for this migration:
- changing non-computer-use OpenAI agent defaults
- changing generic OpenAI orchestration from `gpt-5.2`
- broad OpenAI client cleanup that is unrelated to the computer-use path

## Locked Decisions
1. Scope is limited to the OpenAI computer-use path.
2. `computer-use-preview` will be hard replaced. No compatibility mode, fallback, or transition window.
3. HAINDY should use the moving alias `gpt-5.4`.

## Method And Confidence
### Step 1: Official source review
- Attempted to fetch the requested docs via the OpenAI docs MCP first.
- The docs MCP could list API endpoints, but the specific page fetches for the provided URLs did not resolve cleanly.
- Fell back to direct browsing on official OpenAI domains only: `openai.com` and `developers.openai.com`.
- Confidence: high on the headline contract changes, medium on a few lower-level safety field details that still need one live API confirmation.

### Step 2: Repo inspection
- Inspected the requested repo areas directly:
  - `src/agents/computer_use/`
  - `src/config/settings.py`
  - `src/models/openai_client.py`
  - `README.md`
  - relevant tests under `tests/`
- Ran targeted repo-wide searches for `gpt-5.2`, `computer-use-preview`, `computer_use_preview`, `computer_call_output`, `computer_call`, and related settings/docs strings.
- Confidence: high. The current OpenAI path is consistently preview-era and the pinning is explicit.

### Step 3: Comparison discipline
- Treated the docs as the target contract and the current code as the baseline.
- Marked anything that is directly evidenced by the docs as a required change.
- Marked anything that depends on an incomplete public example as an assumption to validate in a small spike before switching the OpenAI computer-use default.

## Official Sources Reviewed
- OpenAI announcement: <https://openai.com/index/introducing-gpt-5-4/>
- Latest model guide: <https://developers.openai.com/api/docs/guides/latest-model>
- Computer use guide: <https://developers.openai.com/api/docs/guides/tools-computer-use>
- Changelog: <https://developers.openai.com/api/docs/changelog/>
- Responses API reference: <https://developers.openai.com/api/reference/resources/responses/>
- Supplementary model page surfaced from official-domain search: <https://developers.openai.com/docs/models/gpt-5-4>

## What The Latest OpenAI Docs Change
1. GPT-5.4 is now the current flagship OpenAI model, and the changelog entry dated March 5, 2026 positions it as the new latest model.
2. Computer use is now built into `gpt-5.4` via the built-in `computer` tool.
3. OpenAI explicitly keeps `computer-use-preview` available only through a legacy integration path.
4. The current computer-use guide shows the new custom loop shape:
   - request model: `gpt-5.4`
   - tool type: `computer`
   - follow-up input still uses `computer_call_output`
   - follow-up payload now carries an `actions` array
   - `current_url` is nested inside the screenshot output object, not passed as a top-level sibling
5. The repo is already on the Responses API, which is still the right surface. The migration is not a surface-area rewrite; it is a contract migration inside the existing surface.

## Current Repo Findings
### 1. Broader OpenAI model pinning is still hard-coded to `gpt-5.2`, but that is now out of scope
- `src/config/settings.py`
  - `SUPPORTED_OPENAI_MODEL = "gpt-5.2"`
  - all default agent model configs use `gpt-5.2`
  - `openai_model` defaults to `gpt-5.2`
  - validation rejects anything except that exact string
- `src/models/openai_client.py`
  - `SUPPORTED_OPENAI_MODEL = "gpt-5.2"`
  - constructor default is `gpt-5.2`
  - validation rejects anything else
- adjacent defaults still embed `gpt-5.2` in `src/core/interfaces.py`, `src/agents/base_agent.py`, `.env.example`, and multiple tests
- keep these unchanged unless the Phase 0 spike proves the SDK pin or shared dependency wiring must move for GPT-5.4 computer use to function

### 2. The OpenAI computer-use adapter is entirely preview-era
- `src/agents/computer_use/openai_mixin.py`
  - initial request uses `tools=[{"type": "computer_use_preview", ...}]`
  - follow-up request uses the same preview tool type
  - confirmation request also uses the preview tool type
  - follow-up output is a single `computer_call_output` item with one screenshot result and no `actions[]`
- `src/agents/computer_use/common.py`
  - parser extracts `computer_call` items, which is likely still usable, but it assumes the current output contract only
- `src/agents/computer_use/support_mixin.py`
  - confirmation builder still emits preview tool payloads

### 3. The loop assumes one tool action per turn
- `src/agents/computer_use/openai_mixin.py` pulls `computer_calls[0]` and executes only the first call from a response.
- The new guide shows follow-up input shaped around `actions[]`, so the OpenAI path should be upgraded to handle multiple actions per model turn even if the common case remains one action.

### 4. The follow-up payload shape does not match the GPT-5.4 guide
- Current code sends:
  - one `computer_call_output`
  - screenshot payload
  - `current_url` at the top level of the item
  - optional `acknowledged_safety_checks`
- The current guide example uses:
  - one `computer_call_output`
  - screenshot payload
  - `current_url` nested under `output`
  - `actions: [...]`
- This is the single most important protocol delta in the repo.

### 5. Docs, env templates, and tests still describe the legacy path
- `README.md` says OpenAI computer use is typically `computer-use-preview`.
- `.env.example` still sets `HAINDY_COMPUTER_USE_MODEL=computer-use-preview`.
- `docs/RUNBOOK.md` still teaches legacy environment setup and still references `HAINDY_ACTIONS_USE_COMPUTER_TOOL`, which appears to be a stale docs/config artifact rather than an active runtime switch.
- OpenAI computer-use tests and support fixtures are written against the preview request contract.

### 6. `OpenAIClient` is adjacent, but not on the critical path for this migration
- `src/models/openai_client.py` is the generic wrapper used by the non-computer-use agent path.
- `ComputerUseSession` uses `AsyncOpenAI` directly, so the main GPT-5.4 computer-use migration does not require a generic OpenAI-client rewrite.
- The only reason to touch `src/models/openai_client.py`, `pyproject.toml`, or `requirements.lock` in this migration is if the Phase 0 spike proves the current OpenAI SDK pin does not support the GPT-5.4 computer tool cleanly.

## Repo-To-Docs Gap Matrix
| Area | Current repo | Latest docs | Required migration action |
| --- | --- | --- | --- |
| OpenAI computer-use model | `computer-use-preview` | built into `gpt-5.4` | switch OpenAI CU default to `gpt-5.4` |
| Tool declaration | `computer_use_preview` | `computer` | update request builders |
| Follow-up contract | one screenshot output item, no `actions[]`, top-level `current_url` | `computer_call_output` with screenshot output plus `actions[]`, `current_url` nested under `output` | rewrite follow-up builder |
| Turn execution model | executes only `computer_calls[0]` | guide is shaped for action batches | support multiple actions/calls per turn |
| Compatibility posture | preview-only | preview is legacy, GPT-5.4 is current | hard replace preview with no compatibility path |
| Generic OpenAI orchestration models | `gpt-5.2` defaults | unrelated to computer use | leave unchanged in this migration |

## Affected Areas And Files
### Primary runtime changes
- `src/agents/computer_use/openai_mixin.py`
  - rewrite initial, follow-up, and confirmation payload builders for GPT-5.4
  - change loop to support multiple actions/tool calls per turn
- `src/agents/computer_use/common.py`
  - update extraction and normalization helpers for the GPT-5.4 computer-tool response contract
- `src/agents/computer_use/support_mixin.py`
  - update confirmation request builder and any shared helper that still emits preview payloads
- `src/agents/computer_use/session.py`
  - mostly configuration and model-selection wiring; verify no preview assumptions remain

### Configuration and generic OpenAI client changes
- `src/config/settings.py`
  - change the OpenAI computer-use default from `computer-use-preview` to `gpt-5.4`
  - add an explicit hard-fail path for legacy OpenAI preview values
- `pyproject.toml`
- `requirements.lock`
  - only if the Phase 0 spike proves the OpenAI SDK pin must move

### Docs and environment templates
- `README.md`
- `.env.example`
- `docs/RUNBOOK.md`

### Tests to update or add
- `tests/computer_use_session_support.py`
- `tests/test_computer_use_session_openai_flow.py`
- `tests/test_computer_use_session_guardrails.py`
- `tests/test_action_agent.py`
- any tests or fixtures that still hard-code `computer-use-preview` for the OpenAI path

## Migration Plan
## Phase 0: Contract Spike Before Refactor
Purpose: remove ambiguity before touching the production adapter.

Tasks:
1. Add a small temporary spike script or test that calls `client.responses.create` directly with:
   - `model="gpt-5.4"`
   - `tools=[{"type": "computer", ...}]`
   - one desktop screenshot
2. Capture the raw response shape and confirm:
   - actual `computer_call` output item shape
   - whether multiple `computer_call` items can appear in one response
   - exact `computer_call_output` follow-up schema accepted by the live API
   - whether `pending_safety_checks` / `acknowledged_safety_checks` are unchanged
   - whether `openai==2.23.0` works cleanly or needs a version bump
3. Confirm the moving alias `gpt-5.4` works cleanly enough to use directly in HAINDY.

Self-check before leaving Phase 0:
- We have one captured successful GPT-5.4 computer-use turn.
- We have evidence for the exact follow-up payload shape.
- We know whether the current SDK pin is acceptable.

## Phase 1: Introduce A GPT-5.4 OpenAI Computer-Use Adapter
Purpose: migrate protocol handling without mixing it with model-default cleanup.

Tasks:
1. In `src/agents/computer_use/openai_mixin.py`, replace preview tool declarations with the GPT-5.4 built-in `computer` tool.
2. Rewrite `_build_initial_request()` to emit the GPT-5.4 tool contract.
3. Rewrite `_build_follow_up_request()` to emit:
   - `type: "computer_call_output"`
   - nested screenshot output
   - `current_url` under `output`
   - `actions: [...]`
4. Rewrite the loop so it can execute every tool action returned in a turn instead of just `computer_calls[0]`.
5. Preserve existing HAINDY guardrails:
   - observe-only policy
   - domain allow/block rules
   - loop detection
   - max-turn caps
   - action logging
   - safety identifier propagation
6. Revisit the auto-confirmation path. The current preview-era behavior depends on free-form assistant text like "Do you want me to continue?". That may not remain the right control flow under GPT-5.4.

Self-check before leaving Phase 1:
- No OpenAI request builder emits `computer_use_preview`.
- No OpenAI follow-up builder emits preview-era field placement.
- The loop can process more than one tool action per turn.

## Phase 2: Normalize Shared Parsing And Internal Action Contracts
Purpose: keep the rest of the codebase stable while the external provider contract changes.

Tasks:
1. Update `src/agents/computer_use/common.py` so response parsing is explicitly versioned and documented instead of implicitly preview-specific.
2. Decide whether HAINDY needs an internal batch representation, or whether it will simply append one `ComputerToolTurn` per executed action while processing a single model turn.
3. Ensure safety metadata survives the new turn shape.
4. Keep `previous_response_id` chaining intact.
5. Verify model logging still redacts screenshots correctly after the payload shape changes.

Self-check before leaving Phase 2:
- Shared helpers describe the new OpenAI contract clearly.
- Internal turn records still capture all executed actions and safety events.
- Logging and journaling still work with the new payload shape.

## Phase 3: Switch OpenAI Computer-Use Configuration To GPT-5.4
Purpose: make the OpenAI computer-use path use GPT-5.4 by default and reject the preview model outright.

Tasks:
1. In `src/config/settings.py`:
   - change the default OpenAI computer-use model from `computer-use-preview` to `gpt-5.4`
   - update any descriptions or help text that still say preview
2. Add an explicit validation or runtime guard so `CU_PROVIDER=openai` with `computer-use-preview` fails fast instead of silently working.
3. Keep generic OpenAI orchestration settings unchanged in this migration.
4. If the Phase 0 spike proves the OpenAI SDK pin is too old, bump `openai` in `pyproject.toml` and `requirements.lock`. Otherwise leave the generic client layer alone.

Self-check before leaving Phase 3:
- A clean settings load defaults the OpenAI computer-use model to `gpt-5.4`.
- `computer-use-preview` is rejected for the OpenAI provider.
- Non-computer-use OpenAI defaults remain unchanged.

## Phase 4: Rewrite Tests Around The New Contract
Purpose: make the migration durable and catch future regressions.

Tasks:
1. Update OpenAI computer-use fixtures in `tests/computer_use_session_support.py`.
2. Rewrite request-shape assertions in:
   - `tests/test_computer_use_session_openai_flow.py`
   - `tests/test_computer_use_session_guardrails.py`
3. Add explicit tests for:
   - GPT-5.4 request tool type is `computer`
   - follow-up payload contains `actions[]`
   - `current_url` placement matches the new contract
   - multiple actions or multiple tool calls in one model turn
   - safety acknowledgements if still supported
4. Add a failure-path test proving the OpenAI provider rejects `computer-use-preview` after the hard replacement.

Self-check before leaving Phase 4:
- Tests fail if preview payloads are reintroduced accidentally.
- Tests fail if the code falls back to one-action-only behavior.
- Tests fail if the OpenAI path still accepts the preview model after the replacement.

## Phase 5: Docs, Templates, And Developer Workflow Cleanup
Purpose: make the repo teach the new behavior.

Tasks:
1. Update `README.md` to describe OpenAI computer use as GPT-5.4-based rather than `computer-use-preview`.
2. Update `.env.example`:
   - `HAINDY_COMPUTER_USE_MODEL=gpt-5.4`
3. Update `docs/RUNBOOK.md` examples to remove stale preview instructions.
4. Remove or correct `HAINDY_ACTIONS_USE_COMPUTER_TOOL` references if they are truly dead.
5. Keep this migration plan in `docs/plans/` as the implementation reference.

Self-check before leaving Phase 5:
- A new contributor following the docs will configure GPT-5.4 for OpenAI computer use, not the preview model.
- No top-level doc still presents the legacy path as the default.

## Testing And Verification Plan
### Required repo checks
Run after implementation:
- `.venv/bin/ruff check .`
- `.venv/bin/ruff format .`
- `.venv/bin/mypy src`
- `.venv/bin/pytest`

### Targeted automated verification
1. Unit tests for request builders:
   - initial OpenAI GPT-5.4 payload
   - follow-up payload with `actions[]`
   - confirmation payload, if confirmation remains relevant
2. Unit tests for parsing:
   - one `computer_call`
   - multiple `computer_call` items
   - safety metadata propagation
3. Regression tests for session behavior:
   - click
   - drag/drop
   - observe-only rejection
   - safety fail-fast
   - loop detection
   - max-turn handling
4. Negative tests:
   - OpenAI provider rejects `computer-use-preview`
   - no silent fallback or compatibility path remains

### Live verification before default switch
1. Run one small desktop browser scenario with `CU_PROVIDER=openai`.
2. Verify at least:
   - initial tool call succeeds
   - one action executes end-to-end
   - follow-up payload is accepted
   - final assistant response completes without any preview fallback
3. If mobile OpenAI execution is supported, run one mobile smoke test too.

## Rollout, Compatibility, And Deprecation
### Recommended rollout sequence
1. Complete the Phase 0 contract spike.
2. Run targeted live OpenAI smoke tests on GPT-5.4.
3. Merge the adapter rewrite and set the OpenAI computer-use default to `gpt-5.4`.
4. Remove preview-only docs, tests, and config examples in the same change.

### Compatibility guidance
- Google and Anthropic computer-use providers are not part of this migration and should remain behaviorally unchanged.
- There is no compatibility path for OpenAI preview after this migration.
- Any deployment still setting `HAINDY_COMPUTER_USE_MODEL=computer-use-preview` should fail fast and be corrected immediately.

### Deprecation exit criteria
- No default config points at `computer-use-preview`.
- No docs recommend `computer-use-preview`.
- No runtime OpenAI request path emits `computer_use_preview`.
- No tests depend on preview-only payload shape.

## Risks
1. The live GPT-5.4 computer-tool follow-up schema may differ in small but important ways from the guide excerpt.
2. The current OpenAI SDK pin may not fully support the new computer-tool flow cleanly.
3. The internal action model may need batch-aware changes because the new follow-up contract is centered on `actions[]`.
4. Hard replacement means any deployment still configured with `computer-use-preview` will fail immediately after rollout.
5. Hidden tests or docs outside the inspected paths may still reference the preview model and need cleanup.

## Assumptions
1. HAINDY should stay on the Responses API for OpenAI.
2. The migration should change the OpenAI provider only, not Google or Anthropic behavior.
3. The desired end state is GPT-5.4 as the default OpenAI computer-use model, not just an optional override.
4. Non-computer-use OpenAI defaults stay on their current path in this migration.

## Open Questions
1. Does the live GPT-5.4 computer-tool API still expose `pending_safety_checks` / `acknowledged_safety_checks` exactly as the current HAINDY logic expects?
2. Is `openai==2.23.0` sufficient for the GPT-5.4 computer tool, or does the dependency pin need to move?

## Recommended Execution Order
1. Phase 0 contract spike.
2. Phase 1 OpenAI adapter rewrite.
3. Phase 2 shared parsing normalization.
4. Phase 3 OpenAI computer-use config switch.
5. Phase 4 tests.
6. Phase 5 docs cleanup.
7. Live smoke test, then merge the hard replacement.
