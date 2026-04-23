# Update HAINDY Test Forensics for Home Data and Tool-Call Sessions

## Summary

- Work on branch `feat/haindy-test-forensics-tool-call`.
- Update the `haindy-test-forensics` skill and helper so they resolve current HAINDY artifact locations and support tool-call mode.
- Chosen scope: full session coverage for tool-call `test`, `explore`, and one-off session evidence.
- Use effective HAINDY settings only; do not silently fall back to old `./data` paths.

## Key Changes

- Update `.codex/skills/haindy-test-forensics/scripts/locate_run_artifacts.py` to load `settings.data_dir`, `settings.reports_dir`, and `settings.haindy_home`.
- Extend the helper CLI with `--session-id <id>` and `--latest-session`; keep `--run-id`, `--latest`, and `--latest-failed`.
- For batch and tool-call background `test`, resolve traces from `<data-root>/traces/<run_id>.json`, model logs from trace metadata or `settings.model_log_path`, and reports from `settings.reports_dir / run_id`.
- For tool-call sessions, summarize `~/.haindy/sessions/<session-id>/session.json`, `action_artifacts/*.json`, `screenshots/`, `logs/daemon.log`, and `logs/ai_interactions.jsonl`; correlate to a trace when `latest_background_run_id` is present.
- For `explore` or `act` sessions without a trace, return a valid session-focused summary instead of failing because no trace exists.
- Update `SKILL.md` and `agents/openai.yaml` so the skill description, evidence order, workflow, examples, and output guidance distinguish durable data artifacts from session-local tool-call artifacts.
- Keep legacy `./data` behavior explicit: mention that old artifacts require fixing/removing overrides, but do not auto-scan unconfigured legacy paths.

## Implementation Approach

- Verify the worktree before edits and avoid overwriting unrelated user changes.
- Write this plan under `docs/plans/` before code edits.
- Produce one JSON summary shape with optional fields for `run_id`, `session_id`, `mode`, `data_root`, `reports_dir`, `trace_path`, `model_log_path`, `session_dir`, `session_metadata_path`, `action_artifact_paths`, `session_screenshot_dir`, `daemon_log_path`, `ai_interactions_log_path`, failure summaries, and `rg` hints.
- Selection rules: `--session-id` is session-first; `--run-id` is trace-first and then scans sessions for matching metadata/artifacts; `--latest-session` picks newest session metadata directory; default remains latest failed trace.
- Self-review checkpoints: after each helper change, confirm the selected artifact root comes from settings; after each skill-doc change, confirm it tells the investigator which artifact is authoritative for batch test, tool-call test, and session-only explore/act.

## Test Plan

- Add pytest coverage for the helper script using temp `HAINDY_HOME`, `HAINDY_DATA_DIR`, and `HAINDY_REPORTS_DIR` values.
- Cover default data-root trace lookup, exact data-dir override lookup, reports-dir override lookup, `--session-id` summary, `--run-id` session correlation, `--latest-session`, and a session-only no-trace case.
- Cover the no-fallback policy by creating old `./data` artifacts while settings point elsewhere and asserting the helper does not choose them.
- Run `.venv/bin/ruff check .`, `.venv/bin/ruff format .`, `.venv/bin/mypy haindy`, and `.venv/bin/pytest`.

## Assumptions

- Tool-call `test` remains the only tool-call flow guaranteed to have a durable run trace.
- Session-only `explore` and `act` forensics should be best-effort from session metadata, screenshots, daemon logs, and AI interaction logs.
- The implementation should not add compatibility fallbacks for old `./data`; it should make the current configured locations clear and fail loudly when artifacts are absent.
