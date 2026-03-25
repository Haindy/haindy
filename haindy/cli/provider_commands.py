"""Handlers for haindy provider list|set|set-computer-use|set-model commands."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from haindy.auth.credentials import get_api_key
from haindy.config.settings import (
    DEFAULT_CU_PROVIDER_MODELS,
    DEFAULT_NON_CU_PROVIDER_MODELS,
    LEGACY_OPENAI_COMPUTER_USE_MODEL,
    SUPPORTED_OPENAI_COMPUTER_USE_MODEL,
    SUPPORTED_OPENAI_MODEL,
)
from haindy.config.settings_file import load_settings_file, write_settings_file

_CONSOLE = Console()

_SETTINGS_PATH = Path("~/.haindy/settings.json").expanduser()

_ALL_PROVIDERS = ("openai", "openai-codex", "google", "anthropic")
_CU_CAPABLE_PROVIDERS = ("openai", "google", "anthropic")
_AGENT_ONLY_PROVIDERS = ("openai-codex",)

_PROVIDER_KEY_FIELD: dict[str, str] = {
    "openai": "openai",
    "google": "vertex",
    "anthropic": "anthropic",
}
_PROVIDER_SECTION = {provider: provider for provider in _ALL_PROVIDERS}


def _validate_provider(provider: str, *, computer_use: bool = False) -> bool:
    valid_providers = _CU_CAPABLE_PROVIDERS if computer_use else _ALL_PROVIDERS
    if provider in valid_providers:
        return True

    scope = "CU provider" if computer_use else "provider"
    _CONSOLE.print(
        f"[red]Unknown {scope}: {provider}. "
        f"Choose from: {', '.join(valid_providers)}[/red]"
    )
    return False


def _validate_credentials(provider: str) -> bool:
    if provider in _AGENT_ONLY_PROVIDERS:
        return True

    keychain_key = _PROVIDER_KEY_FIELD.get(provider, provider)
    has_key = bool(get_api_key(keychain_key))
    if has_key:
        return True

    _CONSOLE.print(
        f"[red]No credentials configured for {provider!r}. "
        f"Run: haindy auth login {provider}[/red]"
    )
    return False


def _validate_model_name(
    provider: str, model: str, *, computer_use: bool = False
) -> str | None:
    normalized_model = str(model or "").strip()
    if not normalized_model:
        _CONSOLE.print("[red]Model name cannot be empty.[/red]")
        return None

    if computer_use and provider == "openai-codex":
        _CONSOLE.print(
            "[red]Provider 'openai-codex' does not support computer use and cannot have a CU model.[/red]"
        )
        return None

    if provider in {"openai", "openai-codex"} and not computer_use:
        if normalized_model != SUPPORTED_OPENAI_MODEL:
            _CONSOLE.print(
                f"[red]Unsupported OpenAI model {normalized_model!r}. "
                f"Supported model is {SUPPORTED_OPENAI_MODEL!r}.[/red]"
            )
            return None

    if provider == "openai" and computer_use:
        if normalized_model == LEGACY_OPENAI_COMPUTER_USE_MODEL:
            _CONSOLE.print(
                "[red]OpenAI computer-use model 'computer-use-preview' is no longer supported. "
                f"Use {SUPPORTED_OPENAI_COMPUTER_USE_MODEL!r}.[/red]"
            )
            return None
        if normalized_model != SUPPORTED_OPENAI_COMPUTER_USE_MODEL:
            _CONSOLE.print(
                f"[red]Unsupported OpenAI computer-use model {normalized_model!r}. "
                f"Supported model is {SUPPORTED_OPENAI_COMPUTER_USE_MODEL!r}.[/red]"
            )
            return None

    return normalized_model


def _model_field_name(*, computer_use: bool) -> str:
    return "computer_use_model" if computer_use else "model"


def _default_model_for_provider(provider: str, *, computer_use: bool = False) -> str:
    defaults = (
        DEFAULT_CU_PROVIDER_MODELS if computer_use else DEFAULT_NON_CU_PROVIDER_MODELS
    )
    return defaults[provider]


def _build_model_default_payload(
    settings_data: dict[str, object],
    provider: str,
    *,
    computer_use: bool = False,
) -> dict[str, dict[str, str]]:
    section_name = _PROVIDER_SECTION[provider]
    section_data = settings_data.get(section_name, {})
    if not isinstance(section_data, dict):
        section_data = {}

    field_name = _model_field_name(computer_use=computer_use)
    if field_name in section_data:
        return {}

    return {
        section_name: {
            field_name: _default_model_for_provider(provider, computer_use=computer_use)
        }
    }


def handle_provider_list() -> int:
    """Print a table showing available providers and their configuration status."""
    from haindy.auth import OpenAIAuthManager
    from haindy.auth.credentials import list_configured_providers

    configured = list_configured_providers()
    auth_manager = OpenAIAuthManager()
    codex_status = auth_manager.get_status()

    settings_data = load_settings_file(_SETTINGS_PATH)
    agent_provider = settings_data.get("agent", {}).get("provider", "openai")
    cu_provider = settings_data.get("computer_use", {}).get("provider", "google")

    table = Table(title="Provider Configuration", show_header=True)
    table.add_column("Provider", style="bold")
    table.add_column("Credentials")
    table.add_column("Agent (non-CU)")
    table.add_column("Computer Use")

    for provider in _ALL_PROVIDERS:
        if provider == "openai-codex":
            if codex_status.oauth_connected and not codex_status.oauth_expired:
                cred_mark = "[green]connected[/green]"
            else:
                cred_mark = "[dim]not connected[/dim]"
        else:
            keychain_key = _PROVIDER_KEY_FIELD.get(provider, provider)
            cred_mark = (
                "[green]configured[/green]"
                if configured.get(keychain_key)
                else "[dim]not configured[/dim]"
            )

        agent_mark = "[green]active[/green]" if agent_provider == provider else ""
        cu_mark = "[green]active[/green]" if cu_provider == provider else ""

        if provider in _AGENT_ONLY_PROVIDERS:
            cu_mark = "[dim]n/a[/dim]"

        table.add_row(provider, cred_mark, agent_mark, cu_mark)

    _CONSOLE.print(table)
    return 0


def handle_provider_set(provider: str) -> int:
    """Set the active provider for agent (non-CU) calls.

    For providers that also support computer-use, also updates computer_use.provider.
    For agent-only providers (openai-codex), only sets agent.provider and warns about CU.
    """
    if not _validate_provider(provider):
        return 1

    if not _validate_credentials(provider):
        return 1

    settings_data = load_settings_file(_SETTINGS_PATH)
    write_settings_file(
        _SETTINGS_PATH,
        {
            "agent": {"provider": provider},
            **_build_model_default_payload(settings_data, provider, computer_use=False),
        },
    )
    _CONSOLE.print(f"[green]Agent provider set to {provider!r}.[/green]")

    if provider in _AGENT_ONLY_PROVIDERS:
        _CONSOLE.print(
            f"[yellow]Note: {provider!r} does not support computer use. "
            "Computer-use provider is unchanged.[/yellow]"
        )
    else:
        write_settings_file(
            _SETTINGS_PATH,
            {
                "computer_use": {"provider": provider},
                **_build_model_default_payload(
                    settings_data, provider, computer_use=True
                ),
            },
        )
        _CONSOLE.print(
            f"[green]Computer-use provider also set to {provider!r}.[/green]"
        )

    return 0


def handle_provider_set_computer_use(provider: str) -> int:
    """Set the active provider for computer-use calls only."""
    if not _validate_provider(provider, computer_use=True):
        return 1

    if not _validate_credentials(provider):
        return 1

    settings_data = load_settings_file(_SETTINGS_PATH)
    write_settings_file(
        _SETTINGS_PATH,
        {
            "computer_use": {"provider": provider},
            **_build_model_default_payload(settings_data, provider, computer_use=True),
        },
    )
    _CONSOLE.print(f"[green]Computer-use provider set to {provider!r}.[/green]")
    return 0


def handle_provider_set_model(
    provider: str,
    model: str,
    *,
    computer_use: bool = False,
) -> int:
    """Set the configured model for one provider."""
    if not _validate_provider(provider, computer_use=computer_use):
        return 1

    normalized_model = _validate_model_name(
        provider,
        model,
        computer_use=computer_use,
    )
    if normalized_model is None:
        return 1

    section_name = _PROVIDER_SECTION[provider]
    field_name = _model_field_name(computer_use=computer_use)
    write_settings_file(
        _SETTINGS_PATH,
        {section_name: {field_name: normalized_model}},
    )

    scope_label = "computer-use" if computer_use else "non-CU"
    _CONSOLE.print(
        f"[green]Configured {scope_label} model for {provider!r} as {normalized_model!r}.[/green]"
    )
    return 0
