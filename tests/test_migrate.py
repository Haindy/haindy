"""Tests for .env migration logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.migrate import MigrationResult, migrate_from_dotenv


def _write_env(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestMigrateFromDotenv:
    def test_missing_dotenv_returns_warning(self, tmp_path: Path) -> None:
        result = migrate_from_dotenv(
            dotenv_path=tmp_path / "nonexistent.env",
            settings_out=tmp_path / "settings.json",
        )

        assert result.warnings
        assert "not found" in result.warnings[0].lower()

    def test_empty_dotenv_produces_no_output(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        settings_path = tmp_path / "settings.json"
        _write_env(env_path, "")

        result = migrate_from_dotenv(dotenv_path=env_path, settings_out=settings_path)

        assert result.settings_written == []
        assert result.secrets_stored == []
        assert result.secrets_skipped == []
        assert not settings_path.exists()

    def test_secrets_stored_for_non_empty_api_keys(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        settings_path = tmp_path / "settings.json"
        _write_env(
            env_path,
            "HAINDY_OPENAI_API_KEY=sk-test-key\nHAINDY_ANTHROPIC_API_KEY=ant-key\n",
        )

        stored_keys: dict[str, str] = {}

        def fake_set_api_key(provider: str, value: str) -> None:
            stored_keys[provider] = value

        with patch("src.config.migrate.set_api_key", side_effect=fake_set_api_key):
            result = migrate_from_dotenv(dotenv_path=env_path, settings_out=settings_path)

        assert "openai" in result.secrets_stored
        assert "anthropic" in result.secrets_stored
        assert stored_keys["openai"] == "sk-test-key"
        assert stored_keys["anthropic"] == "ant-key"

    def test_empty_api_key_skipped(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        _write_env(env_path, "HAINDY_OPENAI_API_KEY=\n")

        with patch("src.config.migrate.set_api_key") as mock_set:
            result = migrate_from_dotenv(
                dotenv_path=env_path, settings_out=tmp_path / "settings.json"
            )

        assert "openai" in result.secrets_skipped
        mock_set.assert_not_called()

    def test_settings_written_to_file(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        settings_path = tmp_path / "settings.json"
        _write_env(
            env_path,
            "HAINDY_DESKTOP_KEYBOARD_LAYOUT=es\nHAINDY_LOG_LEVEL=DEBUG\n",
        )

        with patch("src.config.migrate.set_api_key"):
            result = migrate_from_dotenv(dotenv_path=env_path, settings_out=settings_path)

        assert settings_path.exists()
        written = json.loads(settings_path.read_text(encoding="utf-8"))
        assert written["desktop"]["keyboard_layout"] == "es"
        assert written["logging"]["level"] == "DEBUG"
        assert "desktop_keyboard_layout" in result.settings_written
        assert "log_level" in result.settings_written

    def test_dry_run_does_not_write_files_or_store_keys(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        settings_path = tmp_path / "settings.json"
        _write_env(
            env_path,
            "HAINDY_OPENAI_API_KEY=sk-key\nHAINDY_LOG_LEVEL=DEBUG\n",
        )

        with patch("src.config.migrate.set_api_key") as mock_set:
            result = migrate_from_dotenv(
                dotenv_path=env_path, settings_out=settings_path, dry_run=True
            )

        assert result.dry_run is True
        assert not settings_path.exists()
        mock_set.assert_not_called()
        assert "openai" in result.secrets_stored

    def test_unrecognized_haindy_var_produces_warning(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        _write_env(env_path, "HAINDY_UNKNOWN_FUTURE_VAR=value\n")

        result = migrate_from_dotenv(
            dotenv_path=env_path, settings_out=tmp_path / "settings.json"
        )

        assert any("HAINDY_UNKNOWN_FUTURE_VAR" in w for w in result.warnings)

    def test_non_haindy_vars_ignored_without_warning(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        _write_env(env_path, "DISPLAY=:0\nHOME=/home/user\n")

        result = migrate_from_dotenv(
            dotenv_path=env_path, settings_out=tmp_path / "settings.json"
        )

        assert result.warnings == []

    def test_dotenv_not_deleted(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        _write_env(env_path, "HAINDY_LOG_LEVEL=INFO\n")

        migrate_from_dotenv(dotenv_path=env_path, settings_out=tmp_path / "settings.json")

        assert env_path.exists()

    def test_mixed_content_splits_correctly(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        settings_path = tmp_path / "settings.json"
        _write_env(
            env_path,
            "\n".join([
                "HAINDY_OPENAI_API_KEY=sk-key",
                "HAINDY_ANTHROPIC_API_KEY=",
                "HAINDY_LOG_LEVEL=WARNING",
                "HAINDY_CU_PROVIDER=anthropic",
                "HAINDY_MAX_TEST_STEPS=50",
            ]) + "\n",
        )

        stored: dict[str, str] = {}

        with patch("src.config.migrate.set_api_key", side_effect=lambda p, v: stored.update({p: v})):
            result = migrate_from_dotenv(dotenv_path=env_path, settings_out=settings_path)

        assert stored.get("openai") == "sk-key"
        assert "anthropic" in result.secrets_skipped
        written = json.loads(settings_path.read_text(encoding="utf-8"))
        assert written["logging"]["level"] == "WARNING"
        assert written["computer_use"]["provider"] == "anthropic"
        assert written["execution"]["max_test_steps"] == 50
