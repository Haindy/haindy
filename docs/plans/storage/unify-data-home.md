# Unify HAINDY Data Artifacts Under `~/.haindy/data`

## Summary

- Move default data-family artifact, cache, and log paths out of the invocation directory and into `HAINDY_HOME/data/projects/<project-id>/...`.
- Keep live tool-call session state under `HAINDY_HOME/sessions/<session-id>/...`.
- Do not automatically move or copy an existing `./data` directory; explicit legacy path overrides continue to work.

## Public Behavior

- `HAINDY_HOME` defaults to `~/.haindy`.
- The default data root becomes `HAINDY_HOME/data/projects/<project-id>`.
- `<project-id>` is deterministic from the resolved current working directory and formatted as `<sanitized-cwd-basename>-<sha256[:12]>`.
- `HAINDY_DATA_DIR` or `storage.data_dir` is an exact root override; project scoping is not added when it is set.
- Specific path overrides still win over `data_dir`, including `HAINDY_MODEL_LOG_PATH`, `HAINDY_DESKTOP_SCREENSHOT_DIR`, and `storage.planning_cache_path`.
- `reports_dir`, screen recordings, `debug_screenshots`, and `generated_test_plans` stay unchanged for this first pass.

## Implementation

- Add a small settings helper that computes the project-scoped default data root from `haindy_home` and `Path.cwd().resolve()`.
- Update unset data-family paths to derive from the effective data root: screenshot directories, model logs, planning/situational/task/execution replay caches, coordinate caches, and platform screenshot directories.
- Update `RunTraceWriter` so its default trace directory is `get_settings().data_dir / "traces"`.
- Ensure `Settings.create_directories()` creates all derived data directories, including macOS screenshot output.
- Preserve tool-call session paths exactly under `HAINDY_HOME/sessions/<session-id>/...`.
- Update docs and bundled HAINDY skill text so forensic paths point at the effective home data root rather than `./data`.

## Test Plan

- Add settings tests for default project-scoped paths using a temp `HAINDY_HOME` and temp cwd.
- Add settings tests proving `HAINDY_DATA_DIR` is an exact override and specific path env vars override derived defaults.
- Update existing config/runtime-environment tests that assert old `data/...` defaults.
- Add trace tests proving the default trace path follows `settings.data_dir / "traces"` while explicit `trace_dir` remains supported.
- Add a tool-call path regression test confirming session paths remain under `HAINDY_HOME/sessions`.
- Run `.venv/bin/ruff check .`, `.venv/bin/ruff format .`, `.venv/bin/mypy haindy`, and `.venv/bin/pytest`.

## Assumptions

- Existing explicit settings or env path overrides are user intent and are preserved.
- Project-scoped home data avoids cache collisions between unrelated working directories.
- Users with old copied env vars may keep writing to `./data` until those overrides are removed.
