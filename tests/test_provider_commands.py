"""Unit tests for provider CLI commands."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch


def _make_configured_providers(*, openai=False, vertex=False, anthropic=False):
    return {"openai": openai, "vertex": vertex, "anthropic": anthropic}


def _make_codex_status(*, oauth_connected=False, oauth_expired=False):
    return SimpleNamespace(
        oauth_connected=oauth_connected,
        oauth_expired=oauth_expired,
    )


class TestHandleProviderList:
    def test_renders_table_without_errors_when_nothing_configured(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch(
                "haindy.auth.credentials.list_configured_providers",
                return_value=_make_configured_providers(),
            ),
            patch("haindy.auth.OpenAIAuthManager") as mock_mgr,
        ):
            mock_mgr.return_value.get_status.return_value = _make_codex_status()
            result = pcm.handle_provider_list()

        assert result == 0

    def test_renders_table_when_providers_configured(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_data = {
            "agent": {"provider": "anthropic"},
            "computer_use": {"provider": "google"},
        }
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings_data), encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch(
                "haindy.auth.credentials.list_configured_providers",
                return_value=_make_configured_providers(
                    openai=True, vertex=True, anthropic=True
                ),
            ),
            patch("haindy.auth.OpenAIAuthManager") as mock_mgr,
        ):
            mock_mgr.return_value.get_status.return_value = _make_codex_status(
                oauth_connected=True
            )
            result = pcm.handle_provider_list()

        assert result == 0


class TestHandleProviderSet:
    def test_set_openai_writes_both_agent_and_cu_provider_and_defaults(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch("haindy.cli.provider_commands.get_api_key", return_value="sk-test"),
        ):
            result = pcm.handle_provider_set("openai")

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data.get("agent", {}).get("provider") == "openai"
        assert data.get("computer_use", {}).get("provider") == "openai"
        assert data.get("openai", {}).get("model") == "gpt-5.4"
        assert data.get("openai", {}).get("computer_use_model") == "gpt-5.4"

    def test_set_anthropic_writes_both_agent_and_cu_provider_and_defaults(
        self, tmp_path
    ):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch(
                "haindy.cli.provider_commands.get_api_key", return_value="sk-ant-key"
            ),
        ):
            result = pcm.handle_provider_set("anthropic")

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data.get("agent", {}).get("provider") == "anthropic"
        assert data.get("computer_use", {}).get("provider") == "anthropic"
        assert data.get("anthropic", {}).get("model") == "claude-sonnet-4-6"
        assert (
            data.get("anthropic", {}).get("computer_use_model") == "claude-sonnet-4-6"
        )

    def test_set_openai_codex_only_writes_agent_provider(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with patch.object(pcm, "_SETTINGS_PATH", settings_path):
            result = pcm.handle_provider_set("openai-codex")

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data.get("agent", {}).get("provider") == "openai-codex"
        assert data.get("openai-codex", {}).get("model") == "gpt-5.4"
        # computer_use.provider should NOT be set
        assert "computer_use" not in data or "provider" not in data.get(
            "computer_use", {}
        )

    def test_set_preserves_existing_provider_model(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "google": {
                        "model": "gemini-custom-pro",
                        "computer_use_model": "gemini-custom-cu",
                    }
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch(
                "haindy.cli.provider_commands.get_api_key", return_value="vertex-key"
            ),
        ):
            result = pcm.handle_provider_set("google")

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data.get("google", {}).get("model") == "gemini-custom-pro"
        assert data.get("google", {}).get("computer_use_model") == "gemini-custom-cu"

    def test_set_anthropic_with_no_key_returns_1(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch("haindy.cli.provider_commands.get_api_key", return_value=None),
        ):
            result = pcm.handle_provider_set("anthropic")

        assert result == 1

    def test_set_unknown_provider_returns_1(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with patch.object(pcm, "_SETTINGS_PATH", settings_path):
            result = pcm.handle_provider_set("unknown-provider")

        assert result == 1


class TestHandleProviderSetComputerUse:
    def test_set_openai_cu_writes_only_computer_use_provider(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch("haindy.cli.provider_commands.get_api_key", return_value="sk-test"),
        ):
            result = pcm.handle_provider_set_computer_use("openai")

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data.get("computer_use", {}).get("provider") == "openai"
        assert data.get("openai", {}).get("computer_use_model") == "gpt-5.4"
        # agent.provider should NOT be written
        assert "agent" not in data or "provider" not in data.get("agent", {})

    def test_set_google_cu_writes_computer_use_provider(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch(
                "haindy.cli.provider_commands.get_api_key", return_value="vertex-key"
            ),
        ):
            result = pcm.handle_provider_set_computer_use("google")

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data.get("computer_use", {}).get("provider") == "google"
        assert (
            data.get("google", {}).get("computer_use_model") == "gemini-3-flash-preview"
        )

    def test_set_cu_without_key_returns_1(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with (
            patch.object(pcm, "_SETTINGS_PATH", settings_path),
            patch("haindy.cli.provider_commands.get_api_key", return_value=None),
        ):
            result = pcm.handle_provider_set_computer_use("google")

        assert result == 1

    def test_set_cu_unknown_provider_returns_1(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with patch.object(pcm, "_SETTINGS_PATH", settings_path):
            result = pcm.handle_provider_set_computer_use("openai-codex")

        assert result == 1


class TestHandleProviderSetModel:
    def test_set_non_cu_model_writes_provider_specific_model(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with patch.object(pcm, "_SETTINGS_PATH", settings_path):
            result = pcm.handle_provider_set_model("google", "gemini-3-flash-preview")

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data.get("google", {}).get("model") == "gemini-3-flash-preview"

    def test_set_cu_model_writes_provider_specific_cu_model(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with patch.object(pcm, "_SETTINGS_PATH", settings_path):
            result = pcm.handle_provider_set_model(
                "anthropic",
                "claude-sonnet-4-6",
                computer_use=True,
            )

        assert result == 0
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert (
            data.get("anthropic", {}).get("computer_use_model") == "claude-sonnet-4-6"
        )

    def test_set_openai_codex_cu_model_is_rejected(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with patch.object(pcm, "_SETTINGS_PATH", settings_path):
            result = pcm.handle_provider_set_model(
                "openai-codex",
                "gpt-5.4",
                computer_use=True,
            )

        assert result == 1

    def test_set_invalid_openai_model_is_rejected(self, tmp_path):
        import haindy.cli.provider_commands as pcm

        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        with patch.object(pcm, "_SETTINGS_PATH", settings_path):
            result = pcm.handle_provider_set_model("openai", "gpt-5.2")

        assert result == 1
