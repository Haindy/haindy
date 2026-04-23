---
name: haindy-setup
description: Use when setting up a fresh HAINDY installation, configuring API credentials for the first time, migrating from a legacy .env file, or verifying the effective configuration.
metadata:
  short-description: Configure HAINDY credentials and settings
---

# Setting Up HAINDY Configuration

HAINDY separates credentials (API keys) from settings (behaviour, paths, models).
Credentials live in the system keychain with an encrypted local file as fallback.
Settings live in `~/.haindy/settings.json`.

## Step 1 — Check what is already configured

```bash
haindy auth status
```

Shows which providers have API keys stored and Codex OAuth status.

## Step 2 — Store credentials interactively

Run the relevant command for the provider you want to use. Each flow guides you
through what is needed and warns about any missing complementary credentials.

```bash
haindy auth login openai        # API key — covers both non-CU and CU calls
haindy auth login openai-codex  # OAuth — covers non-CU calls only
haindy auth login google        # Vertex API key — CU only; prompts for OpenAI too
haindy auth login anthropic     # Anthropic API key — CU only; prompts for OpenAI too
```

Re-run `haindy auth status` to confirm.

## Step 3 — Settings file

A skeleton `~/.haindy/settings.json` is created automatically on first run.
The most commonly changed fields:

```json
{
  "computer_use": {
    "provider": "openai",
    "model": "<model name for the chosen provider>"
  },
  "execution": {
    "automation_backend": "desktop"
  },
  "logging": {
    "level": "INFO"
  }
}
```

Provider models are stored per provider. Use `openai.model`, `openai-codex.model`,
`google.model`, `anthropic.model`, plus provider-specific
`<provider>.computer_use_model` entries for CU-capable providers.

Key sections:

| Section | Fields |
|---------|--------|
| `computer_use` | `provider` (openai\|google\|anthropic), `visual_mode`, `safety_policy` |
| `openai` / `openai-codex` / `google` / `anthropic` | `model`, and `computer_use_model` for CU-capable providers |
| `execution` | `automation_backend` (desktop\|mobile_adb\|mobile_ios), `max_test_steps` |
| `desktop` | `prefer_resolution`, `keyboard_layout` (auto\|us\|es) |
| `macos` | `keyboard_layout` (us\|es), `key_delay_ms`, `clipboard_timeout_seconds` |
| `ios` | `default_device_udid`, `idb_timeout_seconds` |
| `logging` | `level` (DEBUG\|INFO\|WARNING), `format` (json\|text) |
| `storage` | `data_dir`, `reports_dir`, `cache_dir` |
| `cache` | `enable_planning`, `enable_situational`, `enable_execution_replay` |

Storage note: if `storage.data_dir` is unset, data artifacts go under
`~/.haindy/data/projects/<project-id>/`, where the project id is derived from
the resolved current working directory. Setting `storage.data_dir` or
`HAINDY_DATA_DIR` uses that exact root. Old `.env` path overrides like
`HAINDY_DATA_DIR=data` keep writing to `./data` until removed.

## Step 4 — Verify the effective configuration

```bash
haindy config show
```

Prints the full resolved configuration with all API key values redacted.

## Migrating from a legacy .env file

```bash
haindy config migrate          # reads .env in the current directory
haindy config migrate /path/to/.env
```

Routes API keys to the keychain and writes all other settings to
`~/.haindy/settings.json`. The original `.env` is not modified.

## Clearing credentials

```bash
haindy auth clear openai
haindy auth clear anthropic
haindy auth clear google
haindy auth clear openai-codex
```

## Priority chain (lowest to highest)

1. Built-in defaults
2. `~/.haindy/settings.json`
3. `HAINDY_*` environment variables / `.env` file

Environment variables always win, so CI/CD pipelines that set `HAINDY_OPENAI_API_KEY`
(etc.) continue to work without any changes.

## Troubleshooting

**Keychain not available (headless server):** `haindy auth login` falls back silently to
an AES-GCM encrypted file at `~/.local/state/haindy/auth/api_keys.enc`. No action
needed.

**Settings file ignored:** Check for JSON syntax errors — `haindy config show`
will report a parse error if the file is malformed.

**Secrets rejected from settings file:** API key fields (`openai_api_key`,
`anthropic_api_key`, `vertex_api_key`) are not allowed in JSON settings files.
Use `haindy auth login <provider>` instead.
