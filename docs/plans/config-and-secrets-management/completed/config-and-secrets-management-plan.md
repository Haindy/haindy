# Implementation Plan: Configuration and Secrets Management Refactor

**Status:** Completed — implemented in PR #85, branch `feat/config-and-secrets-management`
**Date:** 2026-03-23

---

## Problem

HAINDY used a single flat `.env` file with ~120 `HAINDY_*` environment variables for all
configuration. This mixed secrets (API keys) with behavioural settings (timeouts, paths, visual
modes), making it impossible to commit project-level config, hard to share across machines, and
unfriendly to new users who had to manually populate a long file before the tool would run.

---

## Goals

1. Separate secrets (API keys) from settings (behaviour, paths, models)
2. Store settings in a structured `~/.haindy/settings.json` with hierarchical sections
3. Store API keys in the system keychain with an encrypted local file fallback
4. Support a project-level `.haindy.json` in the working directory for committable overrides
5. Provide interactive CLI commands for credential management (`--auth-set`, etc.)
6. Provide a one-shot migration command (`--config-migrate`) for existing `.env` users
7. Keep full backward compatibility — env vars remain supported at the highest priority

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Settings file format | JSON | No new dependency; consistent with project; users rarely hand-edit |
| Secrets storage | `keyring` + encrypted file fallback | Native OS keychain where available; AES-GCM file for headless servers |
| Project-level settings | `.haindy.json` in CWD | Committable (no secrets allowed), overrides user settings per project |
| Migration | `--config-migrate` command | One-shot import; original `.env` untouched |
| `Settings` model shape | Stays flat | Consumed in too many call sites to restructure; hierarchy lives only in the loader and JSON file |

---

## Priority Chain (lowest to highest)

1. Pydantic field defaults
2. `~/.haindy/settings.json` (user-level)
3. `.haindy.json` in CWD (project-level)
4. `HAINDY_*` env vars + `.env` file (CI/CD, always wins)

API keys additionally resolved as: system keychain → encrypted file fallback → env var override.

---

## New Files

| Path | Purpose |
|------|---------|
| `src/auth/credentials.py` | `get_api_key` / `set_api_key` / `delete_api_key` via `keyring` + `EncryptedJsonFileStore` fallback |
| `src/config/settings_file.py` | `load_settings_file`, `flatten_settings_dict`, `write_settings_file`, `flat_to_nested`; full `_JSON_TO_FIELD` mapping |
| `src/config/migrate.py` | `migrate_from_dotenv()` — reads `.env`, routes secrets to keychain, rest to `settings.json` |
| `src/cli/auth_commands.py` | `handle_auth_set/status/clear` — interactive credential management |
| `src/cli/config_commands.py` | `handle_config_show/migrate` — config inspection and migration |
| `tests/conftest.py` | Autouse fixture isolating `load_settings_file` and `get_api_key` so developer's local config never leaks into tests |
| `tests/test_credentials.py` | Keychain + file fallback tests |
| `tests/test_settings_file.py` | JSON schema loading, flattening, writing, field mapping validation |
| `tests/test_migrate.py` | Migration logic tests |
| `.agents/skills/haindy-setup/SKILL.md` | Setup skill for fresh installs |

---

## Modified Files

| Path | Change |
|------|--------|
| `src/auth/store.py` | `LocalEncryptedAuthStore` replaced by generic `EncryptedJsonFileStore(store_path, key_path)` with `get/set/delete/get_all` |
| `src/auth/paths.py` | Added `get_user_settings_path()`, `get_api_key_store_path()`, `get_api_key_store_key_path()` |
| `src/auth/manager.py` | Updated to use `EncryptedJsonFileStore`; OAuth credential read/write moved to `_get/_set_oauth_credentials` helpers |
| `src/config/settings.py` | `load_settings()` gains 4-layer priority loading; `_SECRET_FIELD_TO_PROVIDER` constant added |
| `src/main.py` | `--auth-set`, `--auth-clear`, `--auth-status`, `--config-show`, `--config-migrate` CLI flags added |
| `pyproject.toml` | `keyring>=25.0.0` added to dependencies |
| `CLAUDE.md` | Setup section updated; Shared Contracts section extended with rules for new settings/secret fields |
| `README.md` / `docs/RUNBOOK.md` | Config setup instructions updated |
| `.env.example` | Header comment added pointing to new CLI commands |
| `tests/test_auth.py` | Rewritten to test `EncryptedJsonFileStore` directly and use `_FakeStore` helper |

---

## Settings JSON Schema

Sections and their most important fields:

```json
{
  "openai":          { "model", "max_retries", "request_timeout_seconds" },
  "computer_use":    { "provider", "model", "google_model", "anthropic_model",
                       "vertex_project", "vertex_location", "safety_policy",
                       "openai_transport", "visual_mode", "keyframe_max_turns", ... },
  "desktop":         { "prefer_resolution", "keyboard_layout", "enable_resolution_switch",
                       "screenshot_dir", "display", "clipboard_timeout_seconds", ... },
  "mobile":          { "screenshot_dir", "coordinate_cache_path", "default_adb_serial", ... },
  "screen_recording":{ "enable", "output_dir", "framerate", "draw_cursor", "prefix" },
  "execution":       { "automation_backend", "max_test_steps", "step_timeout",
                       "actions_max_turns", "actions_allowed_domains", "scroll_*", ... },
  "logging":         { "level", "format", "file", "model_log_path", "max_screenshots" },
  "storage":         { "data_dir", "reports_dir", "screenshots_dir", "cache_dir",
                       "task_plan_cache_path", "planning_cache_path", ... },
  "cache":           { "enable_planning", "enable_situational", "enable_execution_replay" },
  "security":        { "rate_limit_enabled", "rate_limit_requests_per_minute", "sanitize_screenshots" },
  "dev":             { "debug_mode", "save_agent_conversations", "haindy_home" },
  "agent_models":    { "scope_triage", "test_planner", "test_runner", "situational_agent" }
}
```

API key fields (`openai_api_key`, `anthropic_api_key`, `vertex_api_key`) are explicitly rejected
if found in any settings file.

---

## Key Implementation Notes

**`_JSON_TO_FIELD` mapping** — `settings_file.py` uses a fully explicit dict mapping every
`"section.json_key"` to the flat `Settings` field name. Several mappings are non-obvious:
`computer_use.provider` → `cu_provider`, `execution.actions_max_turns` →
`actions_computer_tool_max_turns`, `logging.level` → `log_level`. The field integrity test
`test_all_json_keys_map_to_known_fields` guards this at test time.

**`EncryptedJsonFileStore` refactor** — `LocalEncryptedAuthStore` was purpose-built for OAuth
tokens. By generalising it to `EncryptedJsonFileStore(store_path, key_path)` with a simple
`get/set/delete` interface over an encrypted JSON dict, both the OAuth credential store and the
API key store share the same AES-GCM machinery without code duplication. The OAuth manager now
stores individual credential fields as top-level keys in `codex_oauth.enc`.

**Test isolation** — `tests/conftest.py` patches `load_settings_file` (returns `{}`) and
`get_api_key` (returns `None`) for every test automatically. Without this, the developer's local
`~/.haindy/settings.json` and keychain entries would bleed into test runs.

**`keyring` fallback** — `keyring.errors.NoKeyringError` (available since keyring 25.0.0) is the
reliable fallback trigger. Broad `except Exception` is used as a secondary catch since some
keyring backends raise other errors on headless systems rather than `NoKeyringError`.

---

## New CLI Commands

```
haindy --auth-set openai|google|anthropic   # interactive credential setup
haindy --auth-status                         # show configured providers
haindy --auth-clear <provider>               # remove credentials
haindy --config-show                         # effective config, secrets redacted
haindy --config-migrate [path]               # import .env (default: ./env)
```

---

## Test Results

- 559 tests passing, 1 skipped, 0 failures
- 88 new/rewritten tests across `test_auth.py`, `test_credentials.py`, `test_settings_file.py`, `test_migrate.py`
- `ruff check` clean
