---
name: haindy-setup
description: Use when setting up a fresh HAINDY installation, configuring API credentials for the first time, migrating from a legacy .env file, or verifying the effective configuration.
metadata:
  short-description: Configure HAINDY credentials and settings
---

# Setting Up HAINDY Configuration

HAINDY separates credentials (API keys) from settings (behaviour, paths, models).
Credentials live in the system keychain with an encrypted local file as fallback.
Settings live in `~/.haindy/settings.json` and optionally `.haindy.json` in the
working directory.

## Step 1 — Check what is already configured

```bash
haindy --auth-status
```

This shows which providers have API keys stored and, for Google, whether the
Vertex project and location are set.

## Step 2 — Store API keys interactively

Run the relevant commands for the providers you need. Each prompts for secrets
with hidden input (no shell history exposure).

```bash
haindy --auth-set openai      # prompts for OpenAI API key (sk-...)
haindy --auth-set anthropic   # prompts for Anthropic API key
haindy --auth-set google      # prompts for Vertex project ID, location, and API key
```

Re-run `--auth-status` to confirm.

## Step 3 — Create a settings file (optional but recommended)

Create `~/.haindy/settings.json` for persistent non-secret preferences:

```json
{
  "computer_use": {
    "provider": "google"
  },
  "execution": {
    "automation_backend": "desktop"
  },
  "logging": {
    "level": "INFO"
  }
}
```

Key sections and their most commonly changed fields:

| Section | Fields |
|---------|--------|
| `computer_use` | `provider` (openai\|google\|anthropic), `visual_mode`, `safety_policy` |
| `execution` | `automation_backend` (desktop\|mobile_adb), `max_test_steps` |
| `desktop` | `prefer_resolution`, `keyboard_layout` (us\|es) |
| `logging` | `level` (DEBUG\|INFO\|WARNING), `format` (json\|text) |
| `storage` | `data_dir`, `reports_dir`, `cache_dir` |
| `cache` | `enable_planning`, `enable_situational`, `enable_execution_replay` |

A `.haindy.json` in the working directory uses the same schema and overrides
`~/.haindy/settings.json` for that project only. It is safe to commit (no
secrets are allowed in either file).

## Step 4 — Verify the effective configuration

```bash
haindy --config-show
```

Prints the full resolved configuration with all API key values redacted.

## Migrating from a legacy .env file

If you have an existing `.env`, run:

```bash
haindy --config-migrate          # reads .env in the current directory
haindy --config-migrate /path/to/.env   # explicit path
```

This reads the `.env`, routes API keys to the keychain, and writes all other
settings to `~/.haindy/settings.json`. The original `.env` is not modified.
Run `--config-show` afterwards to verify, then delete the `.env` when satisfied.

## Clearing credentials

```bash
haindy --auth-clear openai
haindy --auth-clear anthropic
haindy --auth-clear google
```

## Priority chain (lowest to highest)

1. Built-in defaults
2. `~/.haindy/settings.json`
3. `.haindy.json` in CWD
4. `HAINDY_*` environment variables / `.env` file

Environment variables always win, so CI/CD pipelines that set `HAINDY_OPENAI_API_KEY`
(etc.) continue to work without any changes.

## Troubleshooting

**Keychain not available (headless server):** `--auth-set` falls back silently to
an AES-GCM encrypted file at `~/.local/state/haindy/auth/api_keys.enc`. No action
needed.

**Settings file ignored:** Check for JSON syntax errors — `haindy --config-show`
will report a parse error if the file is malformed.

**Secrets rejected from settings file:** API key fields (`openai_api_key`,
`anthropic_api_key`, `vertex_api_key`) are not allowed in JSON settings files.
Use `--auth-set` or env vars instead.
