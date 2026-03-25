"""Interactive handlers for haindy auth login|status|clear."""

from __future__ import annotations

import getpass
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from haindy.auth import (
    CODEX_OAUTH_REDIRECT_URI,
    CodexOAuthClient,
    OAuthCallbackCapture,
    OpenAIAuthManager,
)
from haindy.auth.credentials import (
    delete_api_key,
    list_configured_providers,
    set_api_key,
)
from haindy.config.settings_file import load_settings_file, write_settings_file

_CONSOLE = Console()

_API_PROVIDERS = ("openai", "google", "anthropic")
_ALL_PROVIDERS = (*_API_PROVIDERS, "openai-codex")


async def handle_auth_login(provider: str) -> int:
    """Interactively prompt for and store credentials for *provider*."""
    if provider not in _ALL_PROVIDERS:
        _CONSOLE.print(
            f"[red]Unknown provider: {provider}. Choose from: {', '.join(_ALL_PROVIDERS)}[/red]"
        )
        return 1

    if provider == "openai":
        rc = await _set_openai()
        if rc == 0:
            _apply_provider("openai")
        return rc

    if provider == "openai-codex":
        rc = await _login_with_codex_oauth()
        if rc == 0:
            await _prompt_cu_setup_after_codex()
        return rc

    if provider == "google":
        rc = await _set_google()
        if rc == 0:
            _apply_provider("google")
            _print_provider_hint("google")
        return rc

    if provider == "anthropic":
        rc = await _set_anthropic()
        if rc == 0:
            _apply_provider("anthropic")
            _print_provider_hint("anthropic")
        return rc

    return 1  # unreachable


async def handle_auth_status() -> int:
    """Print a table showing which providers have credentials configured."""
    user_settings = load_settings_file(Path("~/.haindy/settings.json").expanduser())
    computer_use = user_settings.get("computer_use", {})

    configured = list_configured_providers()
    auth_manager = OpenAIAuthManager()
    codex_status = auth_manager.get_status()

    table = Table(title="Provider Credential Status", show_header=True)
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Notes")

    openai_mark = (
        "[green]configured[/green]"
        if configured.get("openai")
        else "[dim]not configured[/dim]"
    )
    table.add_row("openai", openai_mark, "")

    google_key_mark = (
        "[green]configured[/green]"
        if configured.get("vertex")
        else "[dim]not configured[/dim]"
    )
    project = computer_use.get("vertex_project") or ""
    location = computer_use.get("vertex_location") or ""
    google_notes = ""
    if project:
        google_notes = f"project={project}"
        if location:
            google_notes += f", location={location}"
    table.add_row("google", google_key_mark, google_notes)

    anthropic_mark = (
        "[green]configured[/green]"
        if configured.get("anthropic")
        else "[dim]not configured[/dim]"
    )
    table.add_row("anthropic", anthropic_mark, "")

    if codex_status.oauth_connected:
        codex_notes = codex_status.oauth_account_label or ""
        expires = _format_optional_timestamp(codex_status.oauth_expires_at)
        if expires != "n/a":
            if codex_notes:
                codex_notes += " "
            codex_notes += f"(expires {expires})"
        if codex_status.oauth_expired:
            codex_notes += " [EXPIRED]"
        table.add_row("openai-codex", "[green]connected[/green]", codex_notes)
    else:
        table.add_row("openai-codex", "[dim]not connected[/dim]", "")

    _CONSOLE.print(table)
    return 0


async def handle_auth_clear(provider: str) -> int:
    """Remove stored credentials for *provider*."""
    if provider not in _ALL_PROVIDERS:
        _CONSOLE.print(
            f"[red]Unknown provider: {provider}. Choose from: {', '.join(_ALL_PROVIDERS)}[/red]"
        )
        return 1

    if provider == "openai-codex":
        OpenAIAuthManager().clear_oauth_credentials()
        _CONSOLE.print("[green]Codex OAuth session cleared.[/green]")
        return 0

    keychain_provider = "vertex" if provider == "google" else provider
    delete_api_key(keychain_provider)
    _CONSOLE.print(f"[green]Credentials cleared for {provider}.[/green]")
    return 0


# --- provider setup flows ---


async def _prompt_cu_setup_after_codex() -> None:
    """After Codex login, prompt the user to set up a CU provider if none has a key."""
    configured = list_configured_providers()
    has_cu_key = (
        configured.get("openai")
        or configured.get("vertex")
        or configured.get("anthropic")
    )
    if has_cu_key:
        return

    _CONSOLE.print("")
    _CONSOLE.print(
        "[yellow]Codex OAuth covers non-CU calls (planning, analysis).[/yellow]"
    )
    _CONSOLE.print(
        "[yellow]Computer use still requires an API key. Set one up now?[/yellow]"
    )
    _CONSOLE.print("  [1] OpenAI API key  (also covers non-CU)")
    _CONSOLE.print("  [2] Google Vertex API key")
    _CONSOLE.print("  [3] Anthropic API key")
    _CONSOLE.print("  [4] Skip — add it later")
    choice = input("Choice (1/2/3/4): ").strip()

    if choice == "1":
        rc = await _set_openai()
        if rc == 0:
            _apply_provider("openai")
    elif choice == "2":
        rc = await _set_google()
        if rc == 0:
            _apply_provider("google")
    elif choice == "3":
        rc = await _set_anthropic()
        if rc == 0:
            _apply_provider("anthropic")


def _apply_provider(provider: str) -> None:
    """Write both agent.provider and computer_use.provider to settings."""
    settings_path = Path("~/.haindy/settings.json").expanduser()
    write_settings_file(
        settings_path,
        {
            "agent": {"provider": provider},
            "computer_use": {"provider": provider},
        },
    )
    _CONSOLE.print(
        f"[dim]Provider set to {provider!r} in ~/.haindy/settings.json[/dim]"
    )


def _print_provider_hint(provider: str) -> None:
    """Print a hint about using haindy provider set after auth login."""
    _CONSOLE.print(
        f"[dim]Run 'haindy provider set {provider}' "
        "to use this provider for all calls.[/dim]"
    )


# --- credential input helpers ---


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

    project = input("Vertex project ID (optional, press Enter to skip): ").strip()
    location_input = input(
        "Vertex location [us-central1] (optional, press Enter to skip): "
    ).strip()
    location = location_input if location_input else "us-central1"

    _CONSOLE.print("Enter your Vertex API key (input hidden):")
    key = getpass.getpass(prompt="")
    if not key.strip():
        _CONSOLE.print("[red]No API key entered. Aborted.[/red]")
        return 1

    set_api_key("vertex", key.strip())

    if project:
        settings_path = Path("~/.haindy/settings.json").expanduser()
        write_settings_file(
            settings_path,
            {"computer_use": {"vertex_project": project, "vertex_location": location}},
        )
        _CONSOLE.print(
            f"[green]Google Vertex credentials stored (project={project}, location={location}).[/green]"
        )
    else:
        _CONSOLE.print("[green]Google Vertex API key stored.[/green]")

    return 0


async def _login_with_codex_oauth() -> int:
    """Run the interactive browser-based Codex OAuth login flow."""
    auth_manager = OpenAIAuthManager()
    oauth_client = CodexOAuthClient()
    pkce = oauth_client.generate_pkce()
    authorize_url = oauth_client.build_authorize_url(pkce.state, pkce.code_challenge)
    callback_capture = OAuthCallbackCapture(CODEX_OAUTH_REDIRECT_URI)
    redirect_url: str | None = None
    callback_listening = False

    try:
        await callback_capture.start()
        callback_listening = True
    except OSError as exc:
        _CONSOLE.print(
            f"[yellow]Callback listener unavailable ({exc}). Falling back to manual redirect capture.[/yellow]"
        )

    _CONSOLE.print("\n[bold cyan]OpenAI Codex OAuth Login[/bold cyan]")
    _CONSOLE.print("[dim]Opening the authorization URL in your browser...[/dim]")
    opened = webbrowser.open(authorize_url)
    if not opened:
        _CONSOLE.print("[yellow]Browser open failed. Open this URL manually:[/yellow]")
        _CONSOLE.print(authorize_url)

    try:
        if callback_listening:
            _CONSOLE.print("[dim]Waiting for the localhost callback...[/dim]")
            redirect_url = await callback_capture.wait_for_redirect(timeout_seconds=120)
            if redirect_url is None:
                _CONSOLE.print(
                    "[yellow]No callback received within 120 seconds. Falling back to manual redirect capture.[/yellow]"
                )
    finally:
        await callback_capture.close()

    if not redirect_url:
        _CONSOLE.print(
            "[dim]After authorizing, paste the final redirect URL below.[/dim]"
        )
        redirect_url = _CONSOLE.input("Redirect URL: ").strip()
        if not redirect_url:
            _CONSOLE.print("[red]No redirect URL provided.[/red]")
            return 1

    try:
        code, state = oauth_client.parse_redirect_url(redirect_url)
        if state != pkce.state:
            raise ValueError("OAuth callback state mismatch.")
        token = await oauth_client.exchange_authorization_code(code, pkce.code_verifier)
        credentials = auth_manager.save_oauth_token_bundle(token)
    except Exception as exc:
        _CONSOLE.print(f"[red]Codex OAuth login failed: {exc}[/red]")
        return 1

    _CONSOLE.print("[green]Codex OAuth login successful.[/green]")
    if credentials.account_label:
        _CONSOLE.print(f"[dim]Account: {credentials.account_label}[/dim]")
    _CONSOLE.print(
        f"[dim]Expires: {_format_optional_timestamp(credentials.expires_at)}[/dim]"
    )
    return 0


def _format_optional_timestamp(value: datetime | None) -> str:
    if value is None:
        return "n/a"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
