"""Handlers for haindy provider list|set|set-computer-use commands."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from haindy.auth.credentials import get_api_key
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
    if provider not in _ALL_PROVIDERS:
        _CONSOLE.print(
            f"[red]Unknown provider: {provider}. "
            f"Choose from: {', '.join(_ALL_PROVIDERS)}[/red]"
        )
        return 1

    # Validate credentials exist
    if provider not in _AGENT_ONLY_PROVIDERS:
        keychain_key = _PROVIDER_KEY_FIELD.get(provider, provider)
        has_key = bool(get_api_key(keychain_key))
        if not has_key:
            _CONSOLE.print(
                f"[red]No credentials configured for {provider!r}. "
                f"Run: haindy auth login {provider}[/red]"
            )
            return 1

    write_settings_file(_SETTINGS_PATH, {"agent": {"provider": provider}})
    _CONSOLE.print(f"[green]Agent provider set to {provider!r}.[/green]")

    if provider in _AGENT_ONLY_PROVIDERS:
        _CONSOLE.print(
            f"[yellow]Note: {provider!r} does not support computer use. "
            "Computer-use provider is unchanged.[/yellow]"
        )
    else:
        write_settings_file(_SETTINGS_PATH, {"computer_use": {"provider": provider}})
        _CONSOLE.print(f"[green]Computer-use provider also set to {provider!r}.[/green]")

    return 0


def handle_provider_set_computer_use(provider: str) -> int:
    """Set the active provider for computer-use calls only."""
    if provider not in _CU_CAPABLE_PROVIDERS:
        _CONSOLE.print(
            f"[red]Unknown CU provider: {provider}. "
            f"Choose from: {', '.join(_CU_CAPABLE_PROVIDERS)}[/red]"
        )
        return 1

    keychain_key = _PROVIDER_KEY_FIELD.get(provider, provider)
    has_key = bool(get_api_key(keychain_key))
    if not has_key:
        _CONSOLE.print(
            f"[red]No credentials configured for {provider!r}. "
            f"Run: haindy auth login {provider}[/red]"
        )
        return 1

    write_settings_file(_SETTINGS_PATH, {"computer_use": {"provider": provider}})
    _CONSOLE.print(f"[green]Computer-use provider set to {provider!r}.[/green]")
    return 0
