"""Shared test fixtures for the HAINDY test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from haindy.config.settings import get_settings
from haindy.config.settings_file import load_settings_file as _load_settings_file


@pytest.fixture(autouse=True)
def _isolate_config_layers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Prevent developer's local ~/.haindy/settings.json and keychain from leaking into tests.

    Without this fixture, tests could behave differently depending on the
    developer's local configuration. This fixture patches the two new
    priority layers so they always return empty/None in tests.

    Tests that specifically test the settings file or credentials layers
    (test_settings_file.py, test_credentials.py, test_migrate.py) patch
    these themselves as needed.
    """
    get_settings.cache_clear()
    developer_settings_path = Path("~/.haindy/settings.json").expanduser()

    def load_isolated_settings(path: Path) -> dict[str, object]:
        if path == developer_settings_path:
            return {}
        return _load_settings_file(path)

    monkeypatch.setattr(
        "haindy.config.settings_file.load_settings_file",
        load_isolated_settings,
    )
    monkeypatch.setattr(
        "haindy.auth.credentials.get_api_key",
        lambda _provider: None,
    )
    yield
    get_settings.cache_clear()
