# Unify Run Screenshot Storage Under `data/`

## Status
Drafted on 2026-03-17.

## Summary

- Make `data/` the canonical root for machine-oriented run artifacts, including run-scoped screenshot evidence.
- Remove the hardcoded `debug_screenshots/` runtime path.
- Keep `reports/` as the home for human-facing outputs such as HTML reports and optional screen recordings.
- Make the configured screenshot paths in `src/config/settings.py` and `.env.example` truthful again.

## Problem

The current screenshot storage model is internally inconsistent.

1. Runtime settings already declare screenshot storage under `data/`:
   - `data/screenshots`
   - `data/screenshots/desktop`
   - `data/screenshots/mobile`
   - `data/model_logs/model_calls.jsonl`
   - `data/traces`
2. `DebugLogger` still hardcodes run-scoped screenshot storage to `debug_screenshots/<run_id>/`.
3. The CLI initializes `DebugLogger` for normal runs, so Test Runner and Computer Use screenshot persistence usually flows into `debug_screenshots/`, not into the configured `data/` paths.
4. Reports and step artifacts then prefer the `DebugLogger` path over configured screenshot settings.

This creates several concrete problems:

1. Screenshot storage bypasses the configuration surface.
2. The codebase advertises `data/` as the artifact root, but the default runtime behavior still writes a separate top-level tree.
3. Test and doc references have to special-case `debug_screenshots/`.
4. Cleanup and retention behavior become harder to reason about because screenshot evidence is split across multiple roots.

## Current Behavior Snapshot

- `src/config/settings.py`
  - `desktop_screenshot_dir = data/screenshots/desktop`
  - `mobile_screenshot_dir = data/screenshots/mobile`
  - `screenshots_dir = data/screenshots`
  - `model_log_path = data/model_logs/model_calls.jsonl`
- `src/monitoring/debug_logger.py`
  - hardcodes `debug_screenshots/<run_id>/`
- `src/main.py`
  - always initializes the global `DebugLogger` for a normal run
- `src/agents/test_runner_artifacts.py`
  - uses `DebugLogger.save_screenshot(...)` when the logger exists
- `src/agents/computer_use/support_mixin.py`
  - persists Computer Use turn screenshots through the debug logger
- `src/agents/test_runner.py`
  - reports `debug_logger.debug_dir` as the screenshot artifact directory when available

## Design Decision

Use `data/screenshots/runs/<run_id>/` as the canonical home for run-scoped screenshot evidence.

This is the recommended target because it:

1. Keeps screenshot evidence under the already-configured `data/` root.
2. Reuses the existing `screenshots_dir` concept instead of introducing another top-level storage family.
3. Preserves a clear distinction between:
   - machine/debug artifacts in `data/`
   - human-facing reports in `reports/`
4. Requires a smaller refactor than a full artifact-tree redesign.

## Proposed Target Layout

After the fix, the intended screenshot-related layout should be:

- `data/screenshots/runs/<run_id>/`
  - per-run screenshot evidence for Test Runner and Computer Use
- `data/screenshots/desktop/`
  - low-level desktop capture output when the desktop capture layer persists frames directly
- `data/screenshots/mobile/`
  - low-level mobile capture output when the mobile capture layer persists frames directly
- `data/model_logs/model_calls.jsonl`
  - append-only model call log
- `data/model_logs/screenshots/`
  - screenshots attached to model-call logging
- `data/traces/<run_id>.json`
  - per-run trace JSON
- `reports/<run_id>/`
  - reports, report-side action logs, bug payloads

## Non-Goals

1. Do not redesign the full `data/` layout in this pass.
2. Do not move HTML reports or optional screen recordings out of `reports/`.
3. Do not migrate existing on-disk historical artifacts automatically.
4. Do not introduce a compatibility symlink or fallback directory unless implementation proves it is necessary.
5. Do not rename every debug/reporting type just for terminology cleanup in the same change.

## Target Behavior

For a normal CLI run:

1. The run-scoped screenshot evidence directory is derived from settings and run id.
2. `DebugLogger` no longer creates or writes to `debug_screenshots/`.
3. Test Runner screenshots, Computer Use turn screenshots, and report artifact references all converge on the same run-scoped directory under `data/screenshots/runs/<run_id>/`.
4. Any fallback screenshot persistence path should still land under configured screenshot storage, not under a separate top-level directory.
5. Configuration, docs, and tests all describe the same runtime behavior.

## Implementation Plan

### 1. Introduce one canonical run-screenshot path resolver

Add a single helper for resolving run-scoped screenshot paths from settings.

Recommended shape:

- keep `screenshots_dir` as the configured screenshot root
- derive `screenshots_dir / "runs" / <run_id>` for run-scoped evidence

Possible homes:

- `src/runtime/artifact_paths.py`
- or a small helper added near `src/config/settings.py`

Requirements:

1. The helper must be easy to reuse from CLI initialization, Test Runner, and Computer Use.
2. It must not hardcode `debug_screenshots/`.
3. It should produce directories lazily and predictably.

### 2. Refactor `DebugLogger` to use configured paths

Update `src/monitoring/debug_logger.py` so that it no longer owns a hardcoded directory layout.

Recommended changes:

1. Pass the resolved run screenshot directory into `DebugLogger` at initialization time.
2. Keep `test_run_id` behavior intact.
3. Keep `ai_interactions.jsonl` colocated with run screenshots unless there is a strong reason to split it later.
4. Remove direct creation of `debug_screenshots/<run_id>/`.

This lets the class remain in place while removing the storage inconsistency.

### 3. Rewire runtime call sites to the canonical path

Update the places that currently assume `DebugLogger` owns screenshot storage:

- `src/main.py`
  - initialize the debug logger with the resolved run screenshot directory
- `src/agents/test_runner_artifacts.py`
  - continue to use the debug logger when present, but the logger should now point into `data/`
  - consider replacing the temporary-file fallback with a configured screenshot path fallback
- `src/agents/computer_use/support_mixin.py`
  - keep screenshot persistence through the logger, but now into the canonical run directory
- `src/agents/test_runner.py`
  - report the resolved canonical screenshot directory in report artifacts

### 4. Tighten the configuration contract

After the runtime is rewired, make the config/docs surface fully match reality.

Update together:

- `src/config/settings.py`
- `.env.example`
- `README.md`
- `docs/RUNBOOK.md` if it mentions artifact locations
- active architecture docs that currently mention `debug_screenshots/`

Important rule for this pass:

- the config should describe the actual default runtime path without exceptions

### 5. Update tests to reflect the new contract

Replace live test expectations that currently name `debug_screenshots/`.

Expected impact areas:

- `tests/test_test_runner_artifacts.py`
- `tests/test_test_runner_executor.py`
- `tests/test_test_runner_step_processor.py`
- `tests/test_main.py`
- any tests asserting report artifact metadata or screenshot paths

Keep the test focus on behavior, not on the old directory name.

### 6. Remove the obsolete top-level path from live code/docs

Once runtime, config, and tests are updated:

1. Remove live references to `debug_screenshots/` from code and active docs.
2. Keep historical mentions in `docs/plans/**/completed/` if desired, since those are historical records.
3. Ensure new runs no longer create a top-level `debug_screenshots/` directory.

## Code Areas

- `src/main.py`
- `src/monitoring/debug_logger.py`
- `src/agents/test_runner_artifacts.py`
- `src/agents/test_runner.py`
- `src/agents/computer_use/support_mixin.py`
- `src/config/settings.py`
- `.env.example`
- `README.md`
- `docs/RUNBOOK.md`
- active design docs under `docs/design/`
- tests that assert screenshot paths or artifact metadata

## Test Plan

Add or update tests to prove:

1. A normal run initializes screenshot storage under `data/screenshots/runs/<run_id>/`.
2. `DebugLogger` writes screenshots and `ai_interactions.jsonl` into the configured run directory.
3. Test Runner screenshot persistence uses the canonical run directory.
4. Computer Use turn screenshots use the same directory.
5. Report artifact metadata points to the canonical run screenshot directory.
6. No runtime path assembly still depends on the literal string `debug_screenshots`.

Standard validation before merging implementation:

- `.venv/bin/ruff check .`
- `.venv/bin/ruff format .`
- `.venv/bin/mypy src`
- `.venv/bin/pytest`

## Acceptance Criteria

1. New runs do not create `debug_screenshots/`.
2. Run-scoped screenshot evidence is written under `data/screenshots/runs/<run_id>/`.
3. The default runtime behavior matches `.env.example` and `src/config/settings.py`.
4. Reports and traces still work without changing their outward behavior.
5. The user can inspect one run and find screenshot evidence under `data/` without needing to know about a second top-level artifact root.

## Follow-Up Work

Possible later cleanup after this fix lands:

1. Decide whether `data/model_logs/screenshots/` should also become run-scoped.
2. Decide whether `DebugLogger` should eventually be renamed to better reflect its broader artifact role.
3. Consider a larger artifact-layout pass that groups more run-scoped machine artifacts together if that proves helpful.
