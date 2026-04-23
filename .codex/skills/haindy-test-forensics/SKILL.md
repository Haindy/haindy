---
name: haindy-test-forensics
description: Investigate failed HAINDY batch or tool-call executions by reconstructing events from the effective data root, model logs, reports, and tool-call session artifacts under `~/.haindy/sessions/<session-id>/`. Use when a user asks for forensics on the last or a specific HAINDY run/session, wants a high-level failure timeline, issue analysis, or candidate fixes from local HAINDY artifacts. Do not use this skill for ordinary `pytest` stack-trace debugging.
---

# Haindy Test Forensics

## Overview

Analyze a failed HAINDY execution run or tool-call session from local artifacts and explain it in plain English.
Prioritize artifacts under the effective HAINDY data root and, for tool-call mode, the session-local artifact directory. Reconstruct the run or session sequence, identify the most likely failure category, and propose fixes without inventing non-visual root causes.

By default, the data root is `~/.haindy/data/projects/<project-id>` for the resolved current working directory. `HAINDY_DATA_DIR` or `storage.data_dir` overrides it exactly. Tool-call session state is separate under `~/.haindy/sessions/<session-id>/`, or under `HAINDY_HOME/sessions/<session-id>` when `HAINDY_HOME` is set. If a user has old copied env vars pointing at `data/...`, artifacts may still be under `./data` until those overrides are removed; do not scan legacy `./data` unless it is the effective configured root.

## Guardrails

- Treat HAINDY Computer Use execution as screenshot-driven and visual-only.
- Do not infer DOM state, widget tree state, accessibility metadata, selectors, or hidden programmatic state unless a non-CU artifact explicitly provides it.
- Do not default to "this needs deterministic checks/selectors" or similar explanations. That is usually the wrong diagnosis for this repo.
- Mention replay cache, coordinate cache, or validation-only mechanics only if the trace or model logs show they directly caused the failure.
- Separate three questions:
  - What the app or site visibly did.
  - What HAINDY visibly did.
  - What the verifier/reporting layer concluded.
- Prefer raw traces, tool-call action artifacts, model logs, session logs, and screenshots over prose in the HTML report.

## Evidence Order

Read artifacts in this order unless the user points you somewhere else.

### Batch runs and tool-call background `test`

1. `<data-root>/traces/<run_id>.json`
This is the authoritative test timeline: backend, start/end times, step ordering, pass/fail status, expected vs actual results, and before/after screenshot paths.

2. `<reports-dir>/<run_id>/*-actions.json`
Use this for per-action detail: interpreted action type, prompts, automation calls, verification summaries, and bug-report payloads.

3. `<data-root>/model_logs/model_calls.jsonl`
Filter by `run_id` to inspect planner/interpreter/executor/verifier reasoning, prompt text, and screenshot attachments.

4. `<data-root>/model_logs/screenshots/`
Use these as visual evidence referenced by model logs. They are often the best way to confirm what the CU model actually saw.

5. `debug_screenshots/<run_id>/`
Use these when the trace or action JSON points to step-level before/after screenshots outside the data root.

6. `<reports-dir>/<run_id>/*.html`
Treat the HTML report as a convenience view, not the source of truth.

### Tool-call sessions

1. `<haindy-home>/sessions/<session-id>/session.json`
Use this for `latest_background_run_id`, latest phase/progress fields, latest screenshot, backend, command counters, and session status.

2. `<haindy-home>/sessions/<session-id>/action_artifacts/*.json`
For background `test`, these are the best per-step tool-call artifacts: action, verification, phase, status transitions, and step result.

3. `<data-root>/traces/<run_id>.json`
Use this when `session.json` or an action artifact provides a `run_id`. Tool-call `test` should usually have this trace; `explore` and one-off `act` may not.

4. `<data-root>/model_logs/model_calls.jsonl`
Filter by `run_id` when one exists. For session-only evidence, also inspect session-local AI logs.

5. `<haindy-home>/sessions/<session-id>/screenshots/`
Use these for command/status screenshots and promoted background-task screenshots.

6. `<haindy-home>/sessions/<session-id>/logs/ai_interactions.jsonl` and `logs/daemon.log`
Use AI interactions and daemon logs to understand session-only `explore`/`act`, daemon failures, timeouts, or missing trace cases.

## Workflow

1. Resolve the run or session.
Default to the latest failed trace unless the user names a `run_id`, `session_id`, or specific artifact path. For tool-call session issues without a `run_id`, resolve the session first.
Use the helper:

```bash
python .codex/skills/haindy-test-forensics/scripts/locate_run_artifacts.py --latest-failed
python .codex/skills/haindy-test-forensics/scripts/locate_run_artifacts.py --run-id <run_id>
python .codex/skills/haindy-test-forensics/scripts/locate_run_artifacts.py --session-id <session_id>
python .codex/skills/haindy-test-forensics/scripts/locate_run_artifacts.py --latest-session
```

The helper uses effective HAINDY settings for `data_dir`, `reports_dir`, and `haindy_home`. It does not auto-scan legacy `./data` unless that path is configured.

2. Reconstruct the timeline from the authoritative source.
For batch and tool-call `test`, start with the trace. For tool-call `explore` or `act` without a trace, start with `session.json`, session screenshots, AI interactions, and daemon logs.
Identify:
- setup steps that passed
- the first failing step or first suspicious divergence
- whether the failure is execution-time, verification-time, or downstream fallout
- what immediately preceded the failure
- whether a tool-call session has only session-local evidence and no durable trace

3. Read the matching action JSON or tool-call action artifact.
Confirm:
- action type (`click`, `type`, `wait`, `skip_navigation`, `reset_app`, etc.)
- prompt given to the CU model
- any automation calls and coordinates
- verification verdict and reasoning
- any generated bug report or blocker classification
- tool-call phase/status transitions and latest action artifact path when present

4. Filter model calls by `run_id`.
Look for:
- interpreter intent drift
- executor observations
- verifier reasoning
- explicit auth/server/UI errors
- evidence of internal tool failure vs product failure

Typical command:

```bash
rg -n "\"run_id\": \"<run_id>\"" <data-root>/model_logs/model_calls.jsonl
```

For session-only `explore` or `act`, inspect session-local logs instead:

```bash
sed -n '1,200p' <haindy-home>/sessions/<session-id>/logs/ai_interactions.jsonl
sed -n '1,200p' <haindy-home>/sessions/<session-id>/logs/daemon.log
```

5. Check the screenshots tied to the failing step or session command.
Use screenshots to answer:
- Was the expected control/screen actually visible?
- Did HAINDY tap/type on the right target?
- Did the app show a business error, auth error, loading state, or unchanged screen?
- Did the verifier interpret the final screen correctly?
- Did the session stop because the app/device state changed outside HAINDY's control?

6. Classify the failure before proposing fixes.
Use one primary category:
- `app_or_backend`: HAINDY reached the correct visible state, inputs were correct, and the product/backend returned the wrong outcome.
- `automation_execution`: HAINDY acted on the wrong visible target, used the wrong coordinates, or never achieved the intended visible interaction.
- `verification_or_reporting`: execution appears correct, but the verifier/reporting path marked it wrong.
- `environment_or_setup`: device, app reset, account state, backend selection, or test environment made the run invalid.
- `test_plan_or_expectation`: the plan's expected result does not match the product's intended or current behavior.
- `tool_call_session_runtime`: the session daemon, session state, command dispatch, timeout, or background-task lifecycle failed before a normal trace/report could be produced.

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
- If a tool-call session has no trace, do not invent a missing test-run root cause. Explain whether the flow was session-only (`explore`/`act`) or whether a background `test` failed before trace creation.
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
For session-only evidence, cite session IDs, action artifact filenames, screenshot filenames, and log paths instead of pretending a trace exists.
Do not overstate confidence.

## Example Trigger Phrases

- `The last test run failed. I need you to do forensics and tell me what happened.`
- `Analyze the latest failed HAINDY run from the data root.`
- `Do forensics on run 20260313T144056Z_447777a8 and suggest fixes.`
- `Explain the sequence of events and root cause for the last mobile_adb failure.`
- `The latest tool-call session got stuck. Figure out what happened.`
- `Analyze session 0f4d... from ~/.haindy/sessions and explain why explore stopped.`
