"""Tests for API key credential storage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import keyring.errors
import pytest

from src.auth.credentials import (
    KEYRING_SERVICE,
    delete_api_key,
    get_api_key,
    list_configured_providers,
    set_api_key,
)
from src.auth.store import EncryptedJsonFileStore


def _make_store(tmp_path: Path) -> EncryptedJsonFileStore:
    return EncryptedJsonFileStore(
        store_path=tmp_path / "api_keys.enc",
        key_path=tmp_path / "api_keys.key",
    )


class TestApiKeyViaFileFallback:
    """Tests for set/get/delete when keyring is unavailable."""

    def test_set_and_get_key(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with (
            patch("src.auth.credentials.keyring.set_password", side_effect=keyring.errors.NoKeyringError()),
            patch("src.auth.credentials.keyring.get_password", side_effect=keyring.errors.NoKeyringError()),
            patch("src.auth.credentials._file_store", return_value=store),
        ):
            set_api_key("openai", "sk-test-key")
            result = get_api_key("openai")

        assert result == "sk-test-key"

    def test_delete_key(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with (
            patch("src.auth.credentials.keyring.set_password", side_effect=keyring.errors.NoKeyringError()),
            patch("src.auth.credentials.keyring.get_password", side_effect=keyring.errors.NoKeyringError()),
            patch("src.auth.credentials.keyring.delete_password", side_effect=Exception()),
            patch("src.auth.credentials._file_store", return_value=store),
        ):
            set_api_key("anthropic", "ant-key")
            delete_api_key("anthropic")
            result = get_api_key("anthropic")

        assert result is None

    def test_get_missing_key_returns_none(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with (
            patch("src.auth.credentials.keyring.get_password", side_effect=keyring.errors.NoKeyringError()),
            patch("src.auth.credentials._file_store", return_value=store),
        ):
            result = get_api_key("vertex")

        assert result is None

    def test_multiple_providers_stored_independently(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with (
            patch("src.auth.credentials.keyring.set_password", side_effect=keyring.errors.NoKeyringError()),
            patch("src.auth.credentials.keyring.get_password", side_effect=keyring.errors.NoKeyringError()),
            patch("src.auth.credentials._file_store", return_value=store),
        ):
            set_api_key("openai", "openai-key")
            set_api_key("anthropic", "anthropic-key")
            openai_key = get_api_key("openai")
            anthropic_key = get_api_key("anthropic")

        assert openai_key == "openai-key"
        assert anthropic_key == "anthropic-key"


class TestApiKeyViaKeychain:
    """Tests for set/get/delete via system keychain."""

    def test_keychain_set_and_get(self) -> None:
        in_memory: dict[tuple[str, str], str] = {}

        def fake_set(service: str, account: str, value: str) -> None:
            in_memory[(service, account)] = value

        def fake_get(service: str, account: str) -> str | None:
            return in_memory.get((service, account))

        with (
            patch("src.auth.credentials.keyring.set_password", side_effect=fake_set),
            patch("src.auth.credentials.keyring.get_password", side_effect=fake_get),
        ):
            set_api_key("openai", "sk-from-keychain")
            result = get_api_key("openai")

        assert result == "sk-from-keychain"
        assert in_memory[(KEYRING_SERVICE, "openai_api_key")] == "sk-from-keychain"

    def test_keychain_preferred_over_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.set("openai_api_key", "from-file")

        with (
            patch("src.auth.credentials.keyring.get_password", return_value="from-keychain"),
            patch("src.auth.credentials._file_store", return_value=store),
        ):
            result = get_api_key("openai")

        assert result == "from-keychain"

    def test_delete_clears_both_keychain_and_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.set("openai_api_key", "from-file")

        deleted: list[tuple[str, str]] = []

        def fake_delete(service: str, account: str) -> None:
            deleted.append((service, account))

        with (
            patch("src.auth.credentials.keyring.delete_password", side_effect=fake_delete),
            patch("src.auth.credentials.keyring.get_password", return_value=None),
            patch("src.auth.credentials._file_store", return_value=store),
        ):
            delete_api_key("openai")
            result = get_api_key("openai")

        assert (KEYRING_SERVICE, "openai_api_key") in deleted
        assert result is None


class TestListConfiguredProviders:
    def test_all_unconfigured(self) -> None:
        with patch("src.auth.credentials.get_api_key", return_value=None):
            result = list_configured_providers()

        assert result == {"openai": False, "anthropic": False, "vertex": False}

    def test_some_configured(self) -> None:
        def fake_get(provider: str) -> str | None:
            return "key" if provider == "openai" else None

        with patch("src.auth.credentials.get_api_key", side_effect=fake_get):
            result = list_configured_providers()

        assert result == {"openai": True, "anthropic": False, "vertex": False}
