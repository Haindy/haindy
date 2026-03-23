"""API key credential storage via system keychain with encrypted file fallback."""

from __future__ import annotations

import keyring
import keyring.errors

from .paths import get_api_key_store_key_path, get_api_key_store_path
from .store import EncryptedJsonFileStore

KEYRING_SERVICE = "haindy"

_PROVIDERS = ("openai", "anthropic", "vertex")

_PROVIDER_TO_ACCOUNT: dict[str, str] = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "vertex": "vertex_api_key",
}


def _file_store() -> EncryptedJsonFileStore:
    return EncryptedJsonFileStore(
        store_path=get_api_key_store_path(),
        key_path=get_api_key_store_key_path(),
    )


def get_api_key(provider: str) -> str | None:
    """Return the stored API key for *provider*, or None if not configured.

    Checks the system keychain first; falls back to the encrypted local file
    when no keyring backend is available (e.g. headless servers).
    """
    account = _PROVIDER_TO_ACCOUNT[provider]
    try:
        value = keyring.get_password(KEYRING_SERVICE, account)
        if value:
            return str(value)
    except (keyring.errors.NoKeyringError, Exception):
        pass
    return _file_store().get(account)


def set_api_key(provider: str, value: str) -> None:
    """Persist the API key for *provider*.

    Writes to the system keychain when available; falls back to the encrypted
    local file.
    """
    account = _PROVIDER_TO_ACCOUNT[provider]
    try:
        keyring.set_password(KEYRING_SERVICE, account, value)
        return
    except (keyring.errors.NoKeyringError, Exception):
        pass
    _file_store().set(account, value)


def delete_api_key(provider: str) -> None:
    """Remove the stored API key for *provider* from all storage locations."""
    account = _PROVIDER_TO_ACCOUNT[provider]
    try:
        keyring.delete_password(KEYRING_SERVICE, account)
    except Exception:
        pass
    _file_store().delete(account)


def list_configured_providers() -> dict[str, bool]:
    """Return a mapping of provider name to whether a key is configured."""
    return {provider: bool(get_api_key(provider)) for provider in _PROVIDERS}
