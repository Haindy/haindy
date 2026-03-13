---
name: haindy-test-forensics
description: Investigate failed HAINDY runtime test executions by reconstructing the sequence of events from `data/traces/*.json`, `data/model_logs/model_calls.jsonl`, screenshots, and `reports/run-id/` artifacts, then explain what happened and suggest fixes. Use when a user asks for forensics on the last or a specific HAINDY run, wants a high-level failure timeline, issue analysis, or candidate fixes from repo-local artifacts in `data/`. Do not use this skill for ordinary `pytest` stack-trace debugging.
---

# Haindy Test Forensics

## Overview

Analyze a failed HAINDY execution run from local artifacts and explain it in plain English.
Prioritize `data/` artifacts, reconstruct the run sequence, identify the most likely failure category, and propose fixes without inventing non-visual root causes.

## Guardrails

- Treat HAINDY Computer Use execution as screenshot-driven and visual-only.
- Do not infer DOM state, widget tree state, accessibility metadata, selectors, or hidden programmatic state unless a non-CU artifact explicitly provides it.
- Do not default to "this needs deterministic checks/selectors" or similar explanations. That is usually the wrong diagnosis for this repo.
- Mention replay cache, coordinate cache, or validation-only mechanics only if the trace or model logs show they directly caused the failure.
- Separate three questions:
  - What the app or site visibly did.
  - What HAINDY visibly did.
  - What the verifier/reporting layer concluded.
- Prefer the raw trace, action JSON, model logs, and screenshots over prose in the HTML report.

## Evidence Order

Read artifacts in this order unless the user points you somewhere else:

1. `data/traces/<run_id>.json`
This is the authoritative run timeline: backend, start/end times, step ordering, pass/fail status, expected vs actual results, and before/after screenshot paths.

2. `reports/<run_id>/*-actions.json`
Use this for per-action detail: interpreted action type, prompts, automation calls, verification summaries, and bug-report payloads.

3. `data/model_logs/model_calls.jsonl`
Filter by `run_id` to inspect planner/interpreter/executor/verifier reasoning, prompt text, and screenshot attachments.

4. `data/model_logs/screenshots/`
Use these as visual evidence referenced by model logs. They are often the best way to confirm what the CU model actually saw.

5. `debug_screenshots/<run_id>/`
Use these when the trace or action JSON points to step-level before/after screenshots outside `data/`.

6. `reports/<run_id>/*.html`
Treat the HTML report as a convenience view, not the source of truth.

## Workflow

1. Resolve the run.
Default to the latest failed run unless the user names a `run_id` or specific artifact path.
Use the helper:

```bash
python .codex/skills/haindy-test-forensics/scripts/locate_run_artifacts.py --latest-failed
python .codex/skills/haindy-test-forensics/scripts/locate_run_artifacts.py --run-id <run_id>
```

2. Reconstruct the timeline from the trace.
Identify:
- setup steps that passed
- the first failing step or first suspicious divergence
- whether the failure is execution-time, verification-time, or downstream fallout
- what immediately preceded the failure

3. Read the matching action JSON.
Confirm:
- action type (`click`, `type`, `wait`, `skip_navigation`, `reset_app`, etc.)
- prompt given to the CU model
- any automation calls and coordinates
- verification verdict and reasoning
- any generated bug report or blocker classification

4. Filter model calls by `run_id`.
Look for:
- interpreter intent drift
- executor observations
- verifier reasoning
- explicit auth/server/UI errors
- evidence of internal tool failure vs product failure

Typical command:

```bash
rg -n "\"run_id\": \"<run_id>\"" data/model_logs/model_calls.jsonl
```

5. Check the screenshots tied to the failing step.
Use screenshots to answer:
- Was the expected control/screen actually visible?
- Did HAINDY tap/type on the right target?
- Did the app show a business error, auth error, loading state, or unchanged screen?
- Did the verifier interpret the final screen correctly?

6. Classify the failure before proposing fixes.
Use one primary category:
- `app_or_backend`: HAINDY reached the correct visible state, inputs were correct, and the product/backend returned the wrong outcome.
- `automation_execution`: HAINDY acted on the wrong visible target, used the wrong coordinates, or never achieved the intended visible interaction.
- `verification_or_reporting`: execution appears correct, but the verifier/reporting path marked it wrong.
- `environment_or_setup`: device, app reset, account state, backend selection, or test environment made the run invalid.
- `test_plan_or_expectation`: the plan's expected result does not match the product's intended or current behavior.

7. Write the answer in four parts:
- High-level sequence of events.
- What the issue is.
- Why that explanation best fits the artifacts.
- Possible fixes the user can review.

## Root-Cause Rules

- Prefer the simplest explanation that fits the visible evidence.
- If the app shows an auth or business error after a correct visible interaction, call that an app/backend/account/environment issue first, not a CU determinism issue.
- If the action prompt, target, or coordinates are visibly wrong, call it an automation execution issue.
- If the screenshots look successful but the verifier says fail, call it a verification/reporting issue.
- If a failure in one step makes later failures inevitable, say so explicitly and separate the first-cause failure from downstream failures.
- If evidence is ambiguous, say what is known, what is uncertain, and which artifact would disambiguate it.

## Output Contract

Keep the response concise and decision-useful.
Include:

- `Sequence of events`
- `Issue`
- `Evidence`
- `Possible fixes`

When citing evidence, prefer concrete artifact paths and step numbers.
Do not overstate confidence.

## Example Trigger Phrases

- `The last test run failed. I need you to do forensics and tell me what happened.`
- `Analyze the latest failed HAINDY run from data/.`
- `Do forensics on run 20260313T144056Z_447777a8 and suggest fixes.`
- `Explain the sequence of events and root cause for the last mobile_adb failure.`
