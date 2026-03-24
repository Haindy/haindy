"""Handlers for haindy --config-show and --config-migrate."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

from haindy.config.migrate import MigrationResult, migrate_from_dotenv
from haindy.config.settings import get_settings

_CONSOLE = Console()

_SECRET_FIELDS = frozenset({"openai_api_key", "anthropic_api_key", "vertex_api_key"})


async def handle_config_show() -> int:
    """Print the effective configuration with secret fields redacted."""
    settings = get_settings()
    raw = settings.model_dump()

    for field in _SECRET_FIELDS:
        if raw.get(field):
            raw[field] = "[REDACTED]"

    formatted = json.dumps(raw, indent=2, default=str)
    _CONSOLE.print(Syntax(formatted, "json", theme="monokai"))
    return 0


async def handle_config_migrate(dotenv_path: Path, dry_run: bool = False) -> int:
    """Migrate an existing .env file to settings.json and keychain."""
    if dry_run:
        _CONSOLE.print(
            f"[bold]Dry run:[/bold] reading {dotenv_path} (no changes will be made)"
        )
    else:
        _CONSOLE.print(f"Migrating {dotenv_path} ...")

    result = migrate_from_dotenv(dotenv_path=dotenv_path, dry_run=dry_run)

    _print_migration_result(result, dotenv_path)
    return 0


def _print_migration_result(result: MigrationResult, dotenv_path: Path) -> None:
    if result.warnings:
        _CONSOLE.print("\n[yellow]Warnings:[/yellow]")
        for w in result.warnings:
            _CONSOLE.print(f"  {w}")

    if (
        not result.settings_written
        and not result.secrets_stored
        and not result.secrets_skipped
    ):
        _CONSOLE.print("[dim]Nothing to migrate.[/dim]")
        return

    if result.secrets_stored:
        label = "Would store" if result.dry_run else "Stored"
        _CONSOLE.print(
            f"\n[green]{label} API keys for:[/green] {', '.join(result.secrets_stored)}"
        )

    if result.secrets_skipped:
        _CONSOLE.print(
            f"[dim]Skipped (empty) API keys for:[/dim] {', '.join(result.secrets_skipped)}"
        )

    if result.settings_written:
        label = "Would write" if result.dry_run else "Wrote"
        _CONSOLE.print(
            f"[green]{label} {len(result.settings_written)} settings to ~/.haindy/settings.json[/green]"
        )

    if not result.dry_run:
        _CONSOLE.print(
            f"\n[dim]The original {dotenv_path} was not modified. "
            "You can delete it once you have verified the new configuration.[/dim]"
        )
