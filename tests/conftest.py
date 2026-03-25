"""Shared test fixtures for the HAINDY test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_config_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent developer's local ~/.haindy/settings.json and keychain from leaking into tests.

    Without this fixture, tests could behave differently depending on the
    developer's local configuration. This fixture patches the two new
    priority layers so they always return empty/None in tests.

    Tests that specifically test the settings file or credentials layers
    (test_settings_file.py, test_credentials.py, test_migrate.py) patch
    these themselves as needed.
    """
    monkeypatch.setattr(
        "haindy.config.settings.load_settings_file",
        lambda _path: {},
        raising=False,
    )
    monkeypatch.setattr(
        "haindy.config.settings.get_api_key",
        lambda _provider: None,
        raising=False,
    )
