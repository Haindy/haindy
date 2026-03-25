# Universal Failed Model-Call Logging

## Status
Implemented on 2026-03-25.

## Summary

- Make failed model calls durable first-class log entries in `data/model_logs/model_calls.jsonl`.
- Apply the same failure-logging contract across all providers and all call surfaces in HAINDY:
  - non-CU agent calls
  - all Computer Use provider paths
  - situational/setup calls
  - test-runner interpretation, verification, and bug-reporting calls
  - streaming and non-streaming requests
- Log failures at the attempt level, not just at the final workflow level.
- Exclude durable failure entries for expected rate-limit-style failures by default:
  - HTTP `429`
  - `resource_exhausted`
  - provider-specific rate-limit equivalents

## Problem

HAINDY currently treats model-call logging as a mostly success-path concern.

In multiple places, the code:

1. builds a request payload
2. awaits the provider call
3. only then calls `ModelCallLogger.log_call(...)`

That means a model call that fails before a response object is returned often has:

- runtime logger output
- step-level failure propagation
- maybe a report-side inference about what happened

but no durable structured entry in `data/model_logs/model_calls.jsonl`.

This creates several concrete problems:

1. Forensics become incomplete exactly where we need them most: provider failures.
2. Error logging behavior is inconsistent across providers and across agent surfaces.
3. Pre-response failures, request rejections, and mid-stream failures are easy to miss in run artifacts.
4. Some downstream components infer product or automation failures from missing execution state when the original failure was actually a model/provider error.
5. There is no single contract that defines which model-call failures must be preserved and which are intentionally suppressed as retry noise.

## Current Behavior Snapshot

### Durable model logging is mostly post-success

- `haindy/agents/test_runner_interpreter.py`
  - logs after `_call_model(...)` succeeds
- `haindy/agents/test_runner_verifier.py`
  - logs after the response is available
- `haindy/agents/test_runner_bug_reports.py`
  - logs after the response is available
- `haindy/agents/situational_agent.py`
  - has a partial fallback path, but this pattern is not universal
- `haindy/agents/computer_use/openai_mixin.py`
  - logs after `_create_response(...)` returns
- `haindy/agents/computer_use/google_mixin.py`
  - logs after `_create_google_response(...)` returns
- `haindy/agents/computer_use/anthropic_mixin.py`
  - logs after `_create_anthropic_response(...)` returns

### Non-CU provider clients are not the logging boundary

- `haindy/models/openai_client.py`
- `haindy/models/google_client.py`
- `haindy/models/anthropic_client.py`
- `haindy/agents/base_agent.py`

Today these layers issue provider requests, but they do not enforce one provider-neutral durable failure-logging contract for all call sites.

### Google retry policy is narrower than the broader failure-logging need

`haindy/agents/computer_use/google_mixin.py` currently treats only rate-limit-style failures as retryable:

- `429`
- `resource_exhausted`
- `"rate limit"` text matches

Non-retryable failures are written to the runtime logger and re-raised, but not durably captured in `model_calls.jsonl` when no response object exists.

## Goal

Define and implement one provider-neutral logging contract:

- every model call attempt that fails must produce a durable structured log entry
- except rate-limit-style failures that are intentionally excluded from durable failure logging

This contract must hold across:

- OpenAI
- Google
- Anthropic
- every agent surface
- every execution mode
- streaming and non-streaming flows
- initial calls and continuation/follow-up calls

## Non-Goals

- Do not redesign the entire artifact layout.
- Do not change report generation semantics beyond consuming better failure evidence.
- Do not change provider retry policy in the same pass except where needed to classify log suppression for excluded rate-limit failures.
- Do not remove existing runtime logger output.
- Do not weaken sanitization or redaction rules when logging failures.

## Working Policy Assumption

For this plan, treat the following as excluded from durable failed-call logging:

1. HTTP `429`
2. `resource_exhausted`
3. provider-specific rate-limit equivalents

They may still appear in normal runtime logs and retry handling, but they should not produce durable failed-call entries in `model_calls.jsonl` by default.

Everything else should be durably logged when the call attempt fails.

## Desired Behavior

For any model-call attempt in HAINDY:

1. A successful call produces the current success-style log entry.
2. A failed call attempt produces a failure log entry even if the provider never returned a normal response object.
3. If the provider returned an error payload or response body, preserve that payload in the durable log entry after the normal sanitization/redaction pass.
4. If the provider failed with an exception before a response body was available, preserve:
   - exception type
   - exception string
   - status/code when available
   - any provider error payload/body reachable from the exception
5. Streaming failures are logged even when they happen after partial output has already been emitted.
6. Retry noise in the excluded rate-limit family does not produce durable failed-call entries.
7. If a non-excluded failed attempt is later followed by a successful retry, both events are still representable:
   - the failed attempt as a failure entry
   - the later successful attempt as a success entry

## Design Decision

Introduce a provider-neutral model-call outcome logging contract at the request wrapper boundary, not at scattered success-only call sites.

The core rule should be:

- model-call logging happens around the provider invocation
- not only after a response object is returned

That means HAINDY needs a common way to record three outcome classes:

1. `success`
2. `failure`
3. `suppressed_retryable_failure`

Only the first two become durable `model_calls.jsonl` entries under this plan.

## Proposed Logging Contract

Each durable model-call entry should explicitly record an outcome.

Recommended fields:

- `outcome`: `success` or `failure`
- `failure_kind`: one of:
  - `provider_http_error`
  - `provider_sdk_error`
  - `stream_error`
  - `response_parse_error`
  - `response_validation_error`
  - `unknown_error`
- `request_payload`
  - sanitized as today
- `response`
  - full normalized success payload on success
  - full normalized error payload on failure when available
- `error`
  - exception metadata for failures:
    - type
    - message
    - status_code if present
    - provider code if present
    - raw/normalized provider error payload if available
- `metadata`
  - provider
  - agent surface
  - run/test/step metadata
  - attempt index
  - retry classification
  - stream flag when applicable

The important contract detail is that failure entries must still contain a `response` field when the provider gave us one, not just a stringified exception.

## Implementation Plan

### 1. Add explicit failure-entry support to `ModelCallLogger`

Extend `haindy/utils/model_logging.py` so failure logging is a first-class operation rather than an improvised fallback.

Recommended shape:

- keep `log_call(...)` for successful or generic entries if useful
- add a dedicated failure-capable path such as:
  - `log_call_attempt(...)`
  - or `log_failure_call(...)`
  - or one unified `log_outcome(...)`

Requirements:

1. The logger must accept either a normal response object or an exception-backed failure payload.
2. The logger must preserve the normalized provider error payload when available.
3. The logger must preserve exception metadata separately from the response payload.
4. The logger must continue to use the existing sanitization/redaction pipeline.

### 2. Introduce one provider-neutral failure-classification helper

Add a small shared helper that decides:

- is this failure excluded retry noise?
- if not, how should it be classified?

This helper should normalize provider-specific error representations into one internal classification scheme.

At minimum it should recognize:

- HTTP `429`
- `resource_exhausted`
- rate-limit-family text/status/code variants

Everything else should default to durable failure logging.

Possible homes:

- `haindy/utils/model_logging.py`
- `haindy/models/`
- or a small new module such as `haindy/runtime/model_call_failures.py`

### 3. Move logging responsibility closer to the provider invocation boundary

The current codebase spreads success logging across many agent surfaces.
That is too late for universal failure capture.

The durable logging contract should move to the layer that directly wraps provider calls.

Recommended target boundaries:

- non-CU:
  - `haindy/models/openai_client.py`
  - `haindy/models/google_client.py`
  - `haindy/models/anthropic_client.py`
- CU:
  - `haindy/agents/computer_use/openai_mixin.py`
  - `haindy/agents/computer_use/google_mixin.py`
  - `haindy/agents/computer_use/anthropic_mixin.py`

This gives us one place per provider surface where both:

- the sanitized request payload
- and any returned error object/exception metadata

are still available.

### 4. Remove success-only assumptions from agent-level call sites

Once provider-boundary logging is in place, simplify or tighten the higher-level call sites so they do not remain responsible for durable failure logging.

Relevant areas include:

- `haindy/agents/base_agent.py`
- `haindy/agents/test_runner_interpreter.py`
- `haindy/agents/test_runner_verifier.py`
- `haindy/agents/test_runner_bug_reports.py`
- `haindy/agents/situational_agent.py`

The goal is:

- provider/client layer guarantees durable attempt logging
- higher-level agent code adds metadata and handles business logic
- higher-level agent code does not need ad hoc try/except logging just to avoid losing failures

### 5. Cover streaming as a first-class failure mode

The non-CU OpenAI, Google, and Anthropic clients all have streaming paths.

The logging contract should explicitly handle:

1. stream fails before any output
2. stream fails after partial text
3. stream fails after partial usage data

Failure entries should preserve:

- partial text if available
- usage observed so far if available
- stream phase metadata
- exception payload/metadata

### 6. Normalize provider error extraction

Create helper logic that extracts as much structured failure detail as possible from provider exceptions.

Examples:

- OpenAI SDK exceptions
  - status code
  - request id if available
  - response body if available
- Google SDK exceptions
  - status code/code
  - provider message
  - attached response/error payload when reachable
- Anthropic SDK exceptions
  - status code
  - error type/body when available

The logger should receive a normalized failure payload instead of only `str(exc)`.

### 7. Make Computer Use follow the same contract as non-CU calls

Computer Use should not remain a special case.

For CU initial, continuation, follow-up, re-ask, and step-reflection requests:

1. success entries stay durable
2. non-excluded failures become durable failure entries
3. missing-response failures no longer disappear from `model_calls.jsonl`

This includes:

- OpenAI CU response creation
- Google CU interactions and generate-content surfaces
- Anthropic CU message creation

### 8. Preserve attempt ordering and correlation

To make failure forensics useful, each entry should be correlateable at the attempt level.

Recommended metadata additions:

- `attempt_number`
- `previous_response_id` or equivalent conversation linkage
- `call_phase`
  - `initial`
  - `continuation`
  - `follow_up`
  - `step_reflection`
  - `stream`
  - `reask`
- `suppressed_retryable_failure`
  - `true` only for excluded failures, and only in runtime handling, not durable logs

### 9. Update docs so the contract is explicit

The repository should document that:

- `data/model_logs/model_calls.jsonl` is for both successful and failed model calls
- excluded rate-limit-family failures are intentionally suppressed from durable failure entries
- provider/runtime logs may still contain additional retry noise

Update:

- `README.md` if it mentions model logs
- `docs/RUNBOOK.md`
- active design docs that describe model-call or artifact behavior

## Code Areas

- `haindy/utils/model_logging.py`
- `haindy/agents/base_agent.py`
- `haindy/models/openai_client.py`
- `haindy/models/google_client.py`
- `haindy/models/anthropic_client.py`
- `haindy/agents/situational_agent.py`
- `haindy/agents/test_runner_interpreter.py`
- `haindy/agents/test_runner_verifier.py`
- `haindy/agents/test_runner_bug_reports.py`
- `haindy/agents/computer_use/openai_mixin.py`
- `haindy/agents/computer_use/google_mixin.py`
- `haindy/agents/computer_use/anthropic_mixin.py`
- tests covering model logging, streaming, and CU provider behavior

## Validation Plan

Add or update tests to prove:

1. A non-CU OpenAI failure with a provider error body produces a durable failure entry.
2. A non-CU Google failure before a normal response object still produces a durable failure entry.
3. A non-CU Anthropic failure does the same.
4. A CU OpenAI failure during initial request produces a durable failure entry.
5. A CU Google failure during `interactions.create(...)` produces a durable failure entry.
6. A CU Anthropic failure during initial or follow-up request produces a durable failure entry.
7. A streaming failure after partial output produces a durable failure entry that includes partial text and error metadata.
8. `429` / `resource_exhausted` / rate-limit-style failures do not produce durable failure entries.
9. Redaction still applies to failure payloads.
10. Existing success logging behavior remains intact.

Standard validation before merging implementation:

- `.venv/bin/ruff check .`
- `.venv/bin/ruff format .`
- `.venv/bin/mypy src`
- `.venv/bin/pytest`

## Acceptance Criteria

1. Every non-excluded failed model-call attempt produces a durable structured entry in `data/model_logs/model_calls.jsonl`.
2. The durable entry contains the normalized full error response payload when the provider exposed one.
3. Pre-response exceptions no longer disappear from model-call artifacts.
4. Streaming failures are durably logged.
5. The behavior is consistent across OpenAI, Google, and Anthropic.
6. The behavior is consistent across non-CU and Computer Use call surfaces.
7. Rate-limit-family failures remain excluded from durable failure entries by policy.

## Rollout

1. Define the provider-neutral failure entry schema and classification rules.
2. Implement the logging contract in `ModelCallLogger`.
3. Apply it to the non-CU provider clients.
4. Apply it to all three CU provider paths.
5. Remove or simplify scattered success-only failure workarounds in higher-level agents.
6. Add provider-specific and streaming regression tests.
7. Update docs to reflect the final contract.

## Risks

- Provider SDK exceptions do not all expose error payloads in the same shape.
- Some failures may expose huge nested payloads; logging may need size guardrails without violating the “entire response” intent.
- Streaming error paths can be easy to miss if logging remains attached only to final aggregation steps.
- Duplicate logging is possible if both provider wrappers and higher-level agents write failure entries without a clear ownership boundary.

## Open Questions To Confirm During Implementation

1. Should an exhausted sequence of excluded rate-limit retries remain completely absent from durable model logs, or should there be one final summarized non-call event elsewhere?
2. For very large provider error payloads, do we preserve the entire sanitized payload as-is, or preserve it under a size cap with an explicit truncation marker?
3. Do we want one unified logger API used by both provider clients and CU mixins, or a lower-level helper that both call into separately?
