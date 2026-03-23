"""Interactive handlers for haindy --auth-set/--auth-status/--auth-clear."""

from __future__ import annotations

import getpass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.auth.credentials import delete_api_key, list_configured_providers, set_api_key
from src.config.settings_file import load_settings_file, write_settings_file

_CONSOLE = Console()

_KNOWN_PROVIDERS = ("openai", "google", "anthropic")


async def handle_auth_set(provider: str) -> int:
    """Interactively prompt for and store credentials for *provider*."""
    if provider not in _KNOWN_PROVIDERS:
        _CONSOLE.print(f"[red]Unknown provider: {provider}. Choose from: {', '.join(_KNOWN_PROVIDERS)}[/red]")
        return 1

    if provider == "openai":
        return await _set_openai()
    if provider == "google":
        return await _set_google()
    if provider == "anthropic":
        return await _set_anthropic()

    return 1  # unreachable


async def handle_auth_status() -> int:
    """Print a table showing which providers have credentials configured."""
    user_settings = load_settings_file(Path("~/.haindy/settings.json").expanduser())
    computer_use = user_settings.get("computer_use", {})

    configured = list_configured_providers()

    table = Table(title="Provider Credential Status", show_header=True)
    table.add_column("Provider", style="bold")
    table.add_column("API Key")
    table.add_column("Notes")

    openai_mark = "[green]configured[/green]" if configured.get("openai") else "[dim]not configured[/dim]"
    table.add_row("openai", openai_mark, "")

    google_key_mark = "[green]configured[/green]" if configured.get("vertex") else "[dim]not configured[/dim]"
    project = computer_use.get("vertex_project") or ""
    location = computer_use.get("vertex_location") or ""
    google_notes = ""
    if project:
        google_notes = f"project={project}"
        if location:
            google_notes += f", location={location}"
    table.add_row("google", google_key_mark, google_notes)

    anthropic_mark = "[green]configured[/green]" if configured.get("anthropic") else "[dim]not configured[/dim]"
    table.add_row("anthropic", anthropic_mark, "")

    _CONSOLE.print(table)
    return 0


async def handle_auth_clear(provider: str) -> int:
    """Remove stored credentials for *provider*."""
    if provider not in _KNOWN_PROVIDERS:
        _CONSOLE.print(f"[red]Unknown provider: {provider}. Choose from: {', '.join(_KNOWN_PROVIDERS)}[/red]")
        return 1

    keychain_provider = "vertex" if provider == "google" else provider
    delete_api_key(keychain_provider)
    _CONSOLE.print(f"[green]Credentials cleared for {provider}.[/green]")
    return 0


async def _set_openai() -> int:
    _CONSOLE.print("Enter your OpenAI API key (input hidden):")
    key = getpass.getpass(prompt="")
    if not key.strip():
        _CONSOLE.print("[red]No key entered. Aborted.[/red]")
        return 1
    set_api_key("openai", key.strip())
    _CONSOLE.print("[green]OpenAI API key stored.[/green]")
    return 0


async def _set_anthropic() -> int:
    _CONSOLE.print("Enter your Anthropic API key (input hidden):")
    key = getpass.getpass(prompt="")
    if not key.strip():
        _CONSOLE.print("[red]No key entered. Aborted.[/red]")
        return 1
    set_api_key("anthropic", key.strip())
    _CONSOLE.print("[green]Anthropic API key stored.[/green]")
    return 0


async def _set_google() -> int:
    _CONSOLE.print("Google Vertex AI setup")
    _CONSOLE.print("")

    project = input("Vertex project ID: ").strip()
    if not project:
        _CONSOLE.print("[red]Project ID is required. Aborted.[/red]")
        return 1

    location_input = input("Vertex location [us-central1]: ").strip()
    location = location_input if location_input else "us-central1"

    _CONSOLE.print("Enter your Vertex API key (input hidden):")
    key = getpass.getpass(prompt="")
    if not key.strip():
        _CONSOLE.print("[red]No API key entered. Aborted.[/red]")
        return 1

    set_api_key("vertex", key.strip())

    settings_path = Path("~/.haindy/settings.json").expanduser()
    write_settings_file(
        settings_path,
        {"computer_use": {"vertex_project": project, "vertex_location": location}},
    )
    _CONSOLE.print(f"[green]Google Vertex credentials stored (project={project}, location={location}).[/green]")
    return 0
