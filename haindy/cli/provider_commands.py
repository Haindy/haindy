"""Provider selection commands for haindy provider <list|set|set-computer-use>."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from haindy.auth import OpenAIAuthManager
from haindy.auth.credentials import list_configured_providers
from haindy.config.settings import get_settings
from haindy.config.settings_file import write_settings_file

_CONSOLE = Console()
_VALID_PROVIDERS = ("openai", "anthropic", "google", "openai-codex")
_API_PROVIDERS = ("openai", "anthropic", "google")
_SETTINGS_PATH = Path("~/.haindy/settings.json")


async def handle_provider_list() -> int:
    """List available providers and show which are active for each use."""
    configured = list_configured_providers()
    codex_status = OpenAIAuthManager().get_status()
    settings = get_settings()
    current_agent = str(settings.agent_provider or "openai").strip().lower()
    current_cu = str(settings.cu_provider or "openai").strip().lower()

    codex_active = bool(codex_status.oauth_connected and not codex_status.oauth_expired)

    has_openai = bool(configured.get("openai")) or codex_active
    has_anthropic = bool(configured.get("anthropic"))
    has_google = bool(configured.get("vertex"))

    any_configured = has_openai or has_anthropic or has_google

    if not any_configured:
        _CONSOLE.print(
            "[yellow]No providers configured. Run: haindy auth login <provider>[/yellow]"
        )
        _CONSOLE.print("  Available: openai, anthropic, google, openai-codex")
        return 0

    table = Table(title="Provider Configuration", show_header=True)
    table.add_column("Provider", style="bold")
    table.add_column("Credentials")
    table.add_column("Non-CU (planning)")
    table.add_column("Computer Use")

    if has_openai:
        cred_label = "[green]configured[/green]"
        ncu_label = "[green]active[/green]" if current_agent == "openai" else ""
        cu_label = "[green]active[/green]" if current_cu == "openai" else ""
        table.add_row("openai", cred_label, ncu_label, cu_label)

    if codex_active and not configured.get("openai"):
        # Show codex separately when no direct openai key
        cred_label = "[green]oauth[/green]"
        ncu_label = "[green]active[/green]" if current_agent == "openai-codex" else ""
        cu_label = "[dim]not supported[/dim]"
        table.add_row("openai-codex", cred_label, ncu_label, cu_label)

    if has_anthropic:
        cred_label = "[green]configured[/green]"
        ncu_label = "[green]active[/green]" if current_agent == "anthropic" else ""
        cu_label = "[green]active[/green]" if current_cu == "anthropic" else ""
        table.add_row("anthropic", cred_label, ncu_label, cu_label)

    if has_google:
        cred_label = "[green]configured[/green]"
        ncu_label = "[green]active[/green]" if current_agent == "google" else ""
        cu_label = "[green]active[/green]" if current_cu == "google" else ""
        table.add_row("google", cred_label, ncu_label, cu_label)

    _CONSOLE.print(table)
    _CONSOLE.print(
        f"\n[dim]Non-CU provider: {current_agent}  |  CU provider: {current_cu}[/dim]"
    )
    _CONSOLE.print(
        "[dim]Change with: haindy provider set <provider>  "
        "or  haindy provider set-computer-use <provider>[/dim]"
    )
    return 0


async def handle_provider_set(provider: str) -> int:
    """Set the active provider for all calls (CU and non-CU)."""
    if provider not in _VALID_PROVIDERS:
        _CONSOLE.print(
            f"[red]Unknown provider: {provider}. "
            f"Choose from: {', '.join(_VALID_PROVIDERS)}[/red]"
        )
        return 1

    configured = list_configured_providers()
    codex_status = OpenAIAuthManager().get_status()
    codex_active = bool(codex_status.oauth_connected and not codex_status.oauth_expired)

    has_creds = _provider_has_credentials(provider, configured, codex_active)
    if not has_creds:
        _CONSOLE.print(f"[red]No credentials configured for {provider}.[/red]")
        _CONSOLE.print(f"Run: haindy auth login {provider}")
        return 1

    settings_path = _SETTINGS_PATH.expanduser()

    if provider == "openai-codex":
        _CONSOLE.print(
            "[yellow]openai-codex cannot be used for computer use. "
            "Only the non-CU (planning) provider will be updated.[/yellow]"
        )
        write_settings_file(settings_path, {"agent": {"provider": "openai-codex"}})
        _CONSOLE.print("[green]Non-CU provider set to openai-codex.[/green]")
        _CONSOLE.print(
            "[dim]Note: you still need a CU provider. "
            "Run: haindy provider set-computer-use <provider>[/dim]"
        )
    else:
        write_settings_file(
            settings_path,
            {
                "agent": {"provider": provider},
                "computer_use": {"provider": provider},
            },
        )
        _CONSOLE.print(f"[green]Provider set to {provider!r} for all calls.[/green]")

    return 0


async def handle_provider_set_computer_use(provider: str) -> int:
    """Set the active provider for computer use only."""
    if provider not in _API_PROVIDERS:
        _CONSOLE.print(
            f"[red]Invalid CU provider: {provider}. "
            f"Choose from: {', '.join(_API_PROVIDERS)}[/red]"
        )
        return 1

    configured = list_configured_providers()
    codex_status = OpenAIAuthManager().get_status()
    codex_active = bool(codex_status.oauth_connected and not codex_status.oauth_expired)

    has_creds = _provider_has_credentials(provider, configured, codex_active)
    if not has_creds:
        _CONSOLE.print(f"[red]No credentials configured for {provider}.[/red]")
        _CONSOLE.print(f"Run: haindy auth login {provider}")
        return 1

    settings_path = _SETTINGS_PATH.expanduser()
    write_settings_file(settings_path, {"computer_use": {"provider": provider}})
    _CONSOLE.print(f"[green]CU provider set to {provider!r}.[/green]")
    return 0


def _provider_has_credentials(
    provider: str, configured: dict[str, bool], codex_active: bool
) -> bool:
    """Return True if credentials are available for the given provider."""
    if provider == "openai":
        return bool(configured.get("openai")) or codex_active
    if provider == "openai-codex":
        return codex_active
    if provider == "anthropic":
        return bool(configured.get("anthropic"))
    if provider == "google":
        return bool(configured.get("vertex"))
    return False
