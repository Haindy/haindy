# Teal Backport Notes

Summary of teal backport changes for downstream contributors:
- Desktop driver added under `src/desktop/` with BrowserDriver compatibility and config-based driver switching.
- Computer Use provider abstraction with Gemini (Google) default and OpenAI fallback.
- Cache upgrades (coordinate, task plan, execution replay) stored under `data/` by default.
- Model-call logging, per-run trace artifacts, and screenshot retention added to reporting.
- Execution replay now keys by plan fingerprint and no longer relies on `can_be_replayed`; validation-only action sets are not replay-cached.
