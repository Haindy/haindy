"""Interactive first-time setup wizard for haindy."""

from __future__ import annotations

import asyncio
import importlib.resources
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

from src.cli.doctor import run_doctor

_console = Console()

_SETUP_MARKER = Path.home() / ".haindy" / "setup_complete"

# AI CLIs and the target paths for haindy skill files
AI_CLIS: dict[str, tuple[str, str]] = {
    "claude": (
        "~/.claude/skills/haindy-setup/",
        "~/.claude/skills/haindy/",
    ),
    "codex": (
        "~/.agents/skills/haindy-setup/",
        "~/.agents/skills/haindy/",
    ),
    "opencode": (
        "~/.config/opencode/skills/haindy-setup/",
        "~/.config/opencode/skills/haindy/",
    ),
}


def _install_skills(cli_name: str, setup_target: str, main_target: str) -> None:
    """Copy bundled SKILL.md files to the target paths for the given CLI."""
    targets = {
        "haindy-setup": Path(setup_target).expanduser(),
        "haindy": Path(main_target).expanduser(),
    }
    try:
        skills_pkg = importlib.resources.files("src.skills")
        for skill_name, target_dir in targets.items():
            target_dir.mkdir(parents=True, exist_ok=True)
            skill_file = skills_pkg / skill_name / "SKILL.md"
            dest = target_dir / "SKILL.md"
            dest.write_bytes(skill_file.read_bytes())
        _console.print(f"Skills installed. Open {cli_name} and run: /haindy-setup")
    except Exception as exc:
        _console.print(
            f"[yellow]Could not install skills for {cli_name}: {exc}[/yellow]"
        )


def run_setup_wizard(non_interactive: bool = False) -> int:
    """Run the interactive (or non-interactive) first-time setup wizard.

    Returns 0 on success, 1 if setup is incomplete.
    """
    try:
        return _wizard(non_interactive=non_interactive)
    except KeyboardInterrupt:
        _console.print("\nSetup interrupted. Run 'haindy setup' to resume.")
        return 0


def _wizard(non_interactive: bool) -> int:
    _console.rule("[bold cyan]Haindy Setup Wizard[/bold cyan]")
    _console.print(
        "\nThis wizard will help you configure Haindy. You will need:\n"
        "  - AI provider credentials (OpenAI, Anthropic, or Google)\n"
        "  - OS-level permissions for desktop automation\n"
        "  - (optional) Mobile testing tools (Android ADB or iOS idb)\n"
    )

    _console.rule("Step 2: Environment Detection")
    platform_label = {
        "darwin": "macOS",
        "linux": "Linux",
        "win32": "Windows",
    }.get(sys.platform, sys.platform)
    _console.print(f"Detected OS: [bold]{platform_label}[/bold]")

    settings_path = Path("~/.haindy/settings.json").expanduser()
    if settings_path.exists():
        _console.print(f"Settings file: [green]found[/green] ({settings_path})")
    else:
        _console.print(f"Settings file: [dim]not found[/dim] ({settings_path})")

    _console.rule("Step 3: AI CLI Skill Installation")
    for cli_binary, (setup_target, main_target) in AI_CLIS.items():
        if shutil.which(cli_binary) is None:
            continue
        should_install = True
        if not non_interactive:
            should_install = Confirm.ask(
                f"We found [bold]{cli_binary}[/bold] on your system.\n"
                f"Install the Haindy skills to {cli_binary}?",
                default=True,
            )
        if should_install:
            _install_skills(cli_binary, setup_target, main_target)

    _console.rule("Step 4: Dependency Check")
    check_android = False
    check_ios = False
    if not non_interactive:
        check_android = Confirm.ask("Check Android/ADB dependencies?", default=False)
        if sys.platform == "darwin":
            check_ios = Confirm.ask("Check iOS/idb dependencies?", default=False)

    doctor_code = run_doctor(include_android=check_android, include_ios=check_ios)
    if not non_interactive and doctor_code != 0:
        _console.print(
            "\n[yellow]Some dependencies are missing. "
            "Review the table above for installation instructions.[/yellow]"
        )

    _console.rule("Step 5: Credential Setup")
    try:
        from src.auth.credentials import get_api_key
        from src.cli.auth_commands import handle_auth_command
    except Exception as exc:
        _console.print(
            f"[red]Could not load credentials module ({exc}). Skipping credential setup.[/red]"
        )
    else:
        for provider in ("openai", "anthropic", "google"):
            keychain_provider = "vertex" if provider == "google" else provider
            try:
                has_key = bool(get_api_key(keychain_provider))
            except Exception:
                has_key = False

            if has_key:
                _console.print(f"[green]{provider} credentials: configured[/green]")
                continue

            _console.print(f"[yellow]{provider} credentials: not configured[/yellow]")
            if non_interactive:
                _console.print(f"  run: haindy --auth login {provider}")
                continue

            if Confirm.ask(f"Set up {provider} credentials now?", default=True):
                asyncio.run(handle_auth_command(["login", provider]))

    _console.rule("Step 7: Final Check")
    final_code = run_doctor()

    if final_code == 0:
        _SETUP_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _SETUP_MARKER.touch()
        _console.print(
            "\n[green]Setup complete![/green] Run:\n\n"
            "  haindy --plan <requirements_file> --context <context_file>\n"
        )
        return 0

    _console.print(
        "\n[yellow]Setup incomplete. Fix the issues above and run: haindy setup[/yellow]"
    )
    return 1
