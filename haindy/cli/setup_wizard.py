"""Interactive first-time setup wizard for haindy."""

from __future__ import annotations

import asyncio
import importlib.resources
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

from haindy.cli.doctor import run_doctor
from haindy.config.settings_file import write_settings_file

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
        skills_pkg = importlib.resources.files("haindy.skills")
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

    _console.rule("Step 1: Environment Detection")
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

    _console.rule("Step 2: AI CLI Skill Installation")
    installed_clis: list[str] = []
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
            installed_clis.append(cli_binary)

    _console.rule("Step 3: Dependency Check")
    doctor_code = run_doctor()
    if not non_interactive and doctor_code != 0:
        _console.print(
            "\n[yellow]Some dependencies are missing. "
            "Review the table above for installation instructions.[/yellow]"
        )

    _console.rule("Step 4: Credential Setup")
    try:
        from haindy.auth.credentials import get_api_key
        from haindy.cli.auth_commands import handle_auth_login
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
                _console.print(f"  run: haindy auth login {provider}")
                continue

            if Confirm.ask(f"Set up {provider} credentials now?", default=True):
                asyncio.run(handle_auth_login(provider))

    _console.rule("Step 5: Provider Selection")
    try:
        from haindy.auth import OpenAIAuthManager
        from haindy.auth.credentials import list_configured_providers

        configured = list_configured_providers()
        codex_status = OpenAIAuthManager().get_status()
        available: list[str] = []
        if configured.get("openai") or (
            codex_status.oauth_connected and not codex_status.oauth_expired
        ):
            available.append("openai")
        if configured.get("anthropic"):
            available.append("anthropic")
        if configured.get("vertex"):
            available.append("google")

        _sp = Path("~/.haindy/settings.json").expanduser()
        if available:
            if len(available) == 1:
                chosen = available[0]
                _console.print(
                    f"Using [bold]{chosen}[/bold] for all AI calls "
                    "(only configured provider)."
                )
                write_settings_file(
                    _sp,
                    {
                        "agent": {"provider": chosen},
                        "computer_use": {"provider": chosen},
                    },
                )
            elif not non_interactive:
                _console.print("Which provider should handle planning and analysis?")
                for i, p in enumerate(available, 1):
                    _console.print(f"  [{i}] {p}")
                _console.print("  [s] Skip -- keep current setting")
                choice_ncu = input(f"Choice (1-{len(available)}/s): ").strip()

                _console.print("Which provider should handle computer use?")
                for i, p in enumerate(available, 1):
                    _console.print(f"  [{i}] {p}")
                _console.print("  [s] Skip -- keep current setting")
                choice_cu = input(f"Choice (1-{len(available)}/s): ").strip()

                ncu_prov = (
                    available[int(choice_ncu) - 1]
                    if choice_ncu.isdigit() and 1 <= int(choice_ncu) <= len(available)
                    else None
                )
                cu_prov = (
                    available[int(choice_cu) - 1]
                    if choice_cu.isdigit() and 1 <= int(choice_cu) <= len(available)
                    else None
                )
                patch: dict[str, object] = {}
                if ncu_prov:
                    patch["agent"] = {"provider": ncu_prov}
                    _console.print(f"[dim]Non-CU provider set to {ncu_prov!r}.[/dim]")
                if cu_prov:
                    patch["computer_use"] = {"provider": cu_prov}
                    _console.print(f"[dim]CU provider set to {cu_prov!r}.[/dim]")
                if patch:
                    write_settings_file(_sp, patch)

                _console.print("")
                _console.print(
                    "[dim]Change later with: haindy provider set <provider>[/dim]"
                )
                _console.print(
                    "[dim]                   haindy provider set-computer-use <provider>[/dim]"
                )
        else:
            _console.print(
                "[dim]No providers configured yet. "
                "Run: haindy auth login <provider>[/dim]"
            )
    except Exception as exc:
        _console.print(f"[yellow]Provider selection skipped ({exc}).[/yellow]")

    _console.rule("Step 6: Final Check")
    final_code = run_doctor()

    if final_code == 0:
        _SETUP_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _SETUP_MARKER.touch()

        if installed_clis:
            agents = " or ".join(installed_clis)
            _console.print(
                f"\n[green]Setup complete.[/green] Open {agents} and use the "
                f"[bold]haindy[/bold] skill to run your first task. For example:\n\n"
                f"  Use the haindy skill to open a desktop session, get a screenshot "
                f"and tell me what you see.\n\n"
                f"[dim]Note: you may need to restart any open agent sessions for the "
                f"skill to load.[/dim]\n"
            )
        else:
            try:
                skill_path = str(
                    importlib.resources.files("haindy.skills") / "haindy" / "SKILL.md"
                )
            except Exception:
                skill_path = "haindy/skills/haindy/SKILL.md"
            _console.print(
                "\n[green]Setup complete.[/green] Open your favourite coding agent "
                "and give it this prompt:\n\n"
                "  Run `haindy session new --desktop` to start a desktop session, "
                "then run `haindy screenshot --session <SESSION_ID>` and tell me "
                "what you see on screen.\n\n"
                f"[dim]Tip: if you have claude, codex, or opencode installed, "
                f"re-run [bold]haindy setup[/bold] to install the haindy skill and "
                f"skip the manual prompting. For other coding agents, install the "
                f"skill manually from {skill_path}[/dim]\n"
            )
        return 0

    _console.print(
        "\n[yellow]Setup incomplete. Fix the issues above and run: haindy setup[/yellow]"
    )
    return 1
