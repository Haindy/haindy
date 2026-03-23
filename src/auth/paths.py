"""Filesystem paths for local authentication state."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_state_path


def get_auth_state_dir() -> Path:
    """Return the per-user directory used for HAINDY auth state."""
    return user_state_path("haindy", ensure_exists=False) / "auth"


def get_codex_oauth_store_path() -> Path:
    """Return the encrypted credential blob path."""
    return get_auth_state_dir() / "codex_oauth.enc"


def get_codex_oauth_key_path() -> Path:
    """Return the symmetric key path for encrypted credential storage."""
    return get_auth_state_dir() / "codex_oauth.key"


def get_user_settings_path() -> Path:
    """Return the user-level settings file path."""
    return Path("~/.haindy").expanduser() / "settings.json"


def get_api_key_store_path() -> Path:
    """Return the encrypted API key store path."""
    return get_auth_state_dir() / "api_keys.enc"


def get_api_key_store_key_path() -> Path:
    """Return the symmetric key path for the API key store."""
    return get_auth_state_dir() / "api_keys.key"
