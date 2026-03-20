"""HAINDY CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents import ScopeTriageAgent, SituationalAgent, TestPlannerAgent
from src.auth import (
    CODEX_OAUTH_REDIRECT_URI,
    CodexOAuthClient,
    OAuthCallbackCapture,
    OpenAIAuthManager,
)
from src.config.settings import Settings, get_settings
from src.core.types import ScopeTriageResult, TestPlan, TestState
from src.desktop.controller import DesktopController
from src.desktop.screen_recorder import ScreenRecorder, ScreenRecorderError
from src.error_handling import ScopeTriageBlockedError
from src.mobile.controller import MobileController
from src.monitoring.debug_logger import initialize_debug_logger
from src.monitoring.logger import get_logger, get_run_id, setup_logging
from src.monitoring.reporter import TestReporter
from src.orchestration.coordinator import WorkflowCoordinator
from src.orchestration.scope_pipeline import run_scope_triage_and_plan
from src.orchestration.state_manager import StateManager
from src.runtime.agent_factory import AgentFactory
from src.runtime.environment import normalize_automation_backend
from src.runtime.execution_context_builder import build_execution_context_bundle
from src.security.rate_limiter import RateLimiter
from src.security.sanitizer import DataSanitizer
from src.tool_call_mode.cli import (
    is_tool_call_command,
    run_tool_call_cli,
    run_tool_call_daemon_cli,
)
from src.tool_call_mode.launcher import public_cli_program_name

console = Console()
logger = get_logger("main")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    cli_name = public_cli_program_name()
    parser = argparse.ArgumentParser(
        description="HAINDY - Autonomous AI Testing Agent v0.1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Full desktop-first execution (required)
  {cli_name} --plan requirements.md --context execution_context.txt

  # Login with OpenAI Codex OAuth
  {cli_name} --codex-auth login

  # Berserk mode
  {cli_name} --berserk --plan requirements.md --context execution_context.txt

  # Test your active OpenAI auth configuration
  {cli_name} --test-api

Fallback:
  python -m src.main --plan requirements.md --context execution_context.txt
        """,
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-p",
        "--plan",
        type=Path,
        help="Path to plain-text test requirements/plan file",
    )
    input_group.add_argument(
        "--test-api",
        action="store_true",
        help="Test the active non-CU OpenAI auth configuration",
    )
    input_group.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )
    input_group.add_argument(
        "--codex-auth",
        choices=["login", "logout", "status"],
        help="Manage OpenAI Codex OAuth credentials for non-CU OpenAI requests",
    )

    parser.add_argument(
        "--context",
        type=Path,
        help="Path to plain-text execution context file (required for execution)",
    )
    parser.add_argument(
        "--mobile",
        action="store_true",
        help="Use mobile ADB backend (hard override for this run)",
    )
    parser.add_argument(
        "--berserk",
        action="store_true",
        help="Berserk mode - aggressive autonomous operation without confirmations",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logging output (JSON)",
    )

    record_group = parser.add_mutually_exclusive_group()
    record_group.add_argument(
        "--record",
        action="store_true",
        dest="record",
        help="Force-enable desktop screen recording for this run",
    )
    record_group.add_argument(
        "--no-record",
        action="store_false",
        dest="record",
        help="Force-disable desktop screen recording for this run",
    )
    parser.set_defaults(record=None)

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory for test results (default: reports/)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "html", "markdown"],
        default="html",
        help="Report format (default: html)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Test execution timeout in seconds (default: 7200)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of test steps (default: 50)",
    )

    return parser


def _create_planning_agents(
    settings: Settings,
) -> tuple[ScopeTriageAgent, TestPlannerAgent, SituationalAgent]:
    """Instantiate planning + setup agents from current settings."""
    planning_agents = AgentFactory(settings).create_planning_agents()
    return (
        planning_agents.scope_triage,
        planning_agents.test_planner,
        planning_agents.situational_agent,
    )


async def run_test(
    requirements: str,
    context_text: str,
    output_dir: Path | None = None,
    report_format: str = "html",
    timeout: int = 7200,
    max_steps: int = 50,
    berserk: bool = False,
    record_override: bool | None = None,
    automation_backend: str = "desktop",
) -> int:
    """Run a test with mandatory requirements and context inputs."""
    automation_backend = normalize_automation_backend(automation_backend)
    automation_controller: DesktopController | MobileController | None = None
    coordinator: WorkflowCoordinator | None = None
    screen_recorder: ScreenRecorder | None = None
    recording_artifact_path: Path | None = None
    settings = get_settings()

    try:
        _ensure_openai_cu_api_key_for_runtime(settings)

        test_run_id = get_run_id()
        if test_run_id == "unknown":
            test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        debug_logger = initialize_debug_logger(test_run_id)
        console.print(f"[dim]Debug logging initialized for run: {test_run_id}[/dim]")

        console.print(
            Panel.fit(
                "[bold cyan]HAINDY - Autonomous Testing Agent[/bold cyan]",
                border_style="cyan",
            )
        )
        if berserk:
            console.print("\n[bold red]BERSERK MODE ACTIVATED[/bold red]")

        triage_agent, planner, situational_agent = _create_planning_agents(settings)

        console.print("\n[yellow]Validating execution context...[/yellow]")
        assessment = await situational_agent.assess_context(
            requirements=requirements,
            context_text=context_text,
        )
        if (
            automation_backend == "mobile_adb"
            and assessment.target_type != "mobile_adb"
        ):
            raise ScopeTriageBlockedError(
                triage_result=ScopeTriageResult(
                    in_scope="",
                    explicit_exclusions=[],
                    ambiguous_points=assessment.notes,
                    blocking_questions=[
                        "--mobile requires mobile_adb context (provide Android adb setup details)."
                    ],
                )
            )
        if (
            assessment.target_type == "mobile_adb"
            and automation_backend != "mobile_adb"
        ):
            raise ScopeTriageBlockedError(
                triage_result=ScopeTriageResult(
                    in_scope="",
                    explicit_exclusions=[],
                    ambiguous_points=assessment.notes,
                    blocking_questions=[
                        "Context requires mobile_adb execution. Re-run with --mobile."
                    ],
                )
            )
        if not assessment.sufficient:
            raise ScopeTriageBlockedError(
                triage_result=ScopeTriageResult(
                    in_scope="",
                    explicit_exclusions=[],
                    ambiguous_points=assessment.notes,
                    blocking_questions=assessment.as_blocking_questions(),
                )
            )

        context_bundle = build_execution_context_bundle(
            context_text=context_text,
            assessment=assessment,
            automation_backend=automation_backend,
        )

        console.print("[yellow]Generating test plan...[/yellow]")
        test_plan, triage_result = await run_scope_triage_and_plan(
            requirements=requirements,
            planner=planner,
            triage_agent=triage_agent,
            context=context_bundle.planning_context,
            cache_key_context=context_bundle.planning_cache_key_context,
        )

        console.print(f"[cyan]Initializing {automation_backend} runtime...[/cyan]")
        automation_controller, coordinator = await _create_coordinator_stack(
            max_steps=max_steps,
            backend=automation_backend,
            agent_factory=AgentFactory(settings),
        )
        action_agent = coordinator.get_action_agent()

        should_record = bool(settings.enable_screen_recording)
        if record_override is not None:
            should_record = bool(record_override)
        if should_record and automation_backend == "desktop":
            screen_recorder = ScreenRecorder(
                output_dir=settings.screen_recording_output_dir,
                framerate=settings.screen_recording_framerate,
                draw_cursor=settings.screen_recording_draw_cursor,
                filename_prefix=settings.screen_recording_prefix,
            )
            try:
                recording_artifact_path = screen_recorder.start()
                console.print(
                    f"[dim]Screen recording started:[/dim] {recording_artifact_path}"
                )
            except ScreenRecorderError as exc:
                logger.warning(
                    "Unable to start screen recording",
                    exc_info=True,
                    extra={"error": str(exc)},
                )
                screen_recorder = None
        elif should_record:
            logger.info(
                "Screen recording requested but disabled for non-desktop backend",
                extra={"automation_backend": automation_backend},
            )

        console.print(
            "[cyan]Preparing entrypoint state with Situational Agent...[/cyan]"
        )
        await situational_agent.prepare_entrypoint(
            automation_controller.driver,
            assessment,
            action_agent=action_agent,
        )

        console.print("[cyan]Running test...[/cyan]")
        test_state = await _run_with_timeout(
            coordinator=coordinator,
            requirements=requirements,
            precomputed_plan=test_plan,
            triage_result=triage_result,
            timeout=timeout,
            context=context_bundle.test_context,
            initial_url=context_bundle.initial_url,
        )

        if screen_recorder:
            try:
                stopped_path = screen_recorder.stop()
                if stopped_path:
                    recording_artifact_path = stopped_path
            except ScreenRecorderError:
                logger.warning("Unable to stop screen recording cleanly", exc_info=True)
            finally:
                screen_recorder = None

        raw_status = getattr(
            test_state, "status", getattr(test_state, "test_status", "unknown")
        )
        status_value = raw_status.value if hasattr(raw_status, "value") else raw_status
        success_statuses = {"passed", "completed"}
        status_color = "green" if status_value in success_statuses else "red"

        console.print("\n[bold]Test Execution Summary:[/bold]")
        console.print(f"Status: [{status_color}]{status_value}[/{status_color}]")

        report = getattr(test_state, "test_report", None)
        if report and recording_artifact_path:
            report.artifacts["screen_recording_path"] = str(recording_artifact_path)

        final_triage = triage_result
        if coordinator:
            stored_triage = coordinator.get_scope_triage_result(
                test_state.test_plan.plan_id
            )
            if stored_triage:
                final_triage = stored_triage
        _print_scope_triage_followups(final_triage)

        if output_dir is None:
            output_dir = debug_logger.reports_dir
        output_dir.mkdir(exist_ok=True)

        action_storage = None
        if (
            coordinator
            and hasattr(coordinator, "_agents")
            and "test_runner" in coordinator._agents
        ):
            runner = coordinator._agents["test_runner"]
            if hasattr(runner, "get_action_storage"):
                action_storage = runner.get_action_storage()

        reporter = TestReporter()
        report_path, actions_path = await reporter.generate_report(
            test_state=test_state,
            output_dir=output_dir,
            format=report_format,
            action_storage=action_storage,
        )
        console.print(f"[green]Report saved to:[/green] {report_path}")
        if actions_path:
            console.print(f"[green]Actions saved to:[/green] {actions_path}")

        return 0 if status_value in success_statuses else 1

    except ScopeTriageBlockedError as scope_error:
        logger.warning("Context/scope gate blocked planning", exc_info=True)
        _print_scope_blockers(scope_error)
        return 1
    except asyncio.TimeoutError:
        console.print(
            f"\n[red]Error: Test execution timed out after {timeout} seconds[/red]"
        )
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Test execution interrupted by user[/yellow]")
        return 130
    except Exception as exc:
        console.print(f"\n[red]Error during test execution: {exc}[/red]")
        logger.exception("Test execution failed")
        return 1
    finally:
        if screen_recorder:
            try:
                screen_recorder.stop()
            except ScreenRecorderError:
                logger.warning(
                    "Screen recording stop failed during cleanup", exc_info=True
                )
        if coordinator:
            try:
                await coordinator.cleanup()
            except Exception:
                logger.warning("Coordinator cleanup failed", exc_info=True)
        if automation_controller:
            try:
                await automation_controller.stop()
            except Exception:
                logger.warning("Automation controller stop failed", exc_info=True)


async def _create_coordinator_stack(
    max_steps: int,
    backend: str = "desktop",
    agent_factory: AgentFactory | None = None,
) -> tuple[DesktopController | MobileController, WorkflowCoordinator]:
    """Build and initialize the automation backend/coordinator stack."""
    normalized_backend = normalize_automation_backend(backend)
    if normalized_backend == "mobile_adb":
        automation_controller: DesktopController | MobileController = MobileController()
    else:
        automation_controller = DesktopController()
    try:
        await automation_controller.start()
    except Exception:
        with contextlib.suppress(Exception):
            await automation_controller.stop()
        raise

    coordinator: WorkflowCoordinator | None = None
    try:
        coordinator = WorkflowCoordinator(
            state_manager=StateManager(),
            automation_driver=automation_controller.driver,
            agent_factory=agent_factory,
            max_steps=max_steps,
        )
        await coordinator.initialize()
        return automation_controller, coordinator
    except Exception:
        if coordinator is not None:
            with contextlib.suppress(Exception):
                await coordinator.cleanup()
        with contextlib.suppress(Exception):
            await automation_controller.stop()
        raise


async def _run_with_timeout(
    coordinator: WorkflowCoordinator,
    requirements: str,
    precomputed_plan: TestPlan,
    triage_result: ScopeTriageResult,
    timeout: int,
    context: dict[str, Any] | None = None,
    initial_url: str | None = None,
) -> TestState:
    """Execute the coordinator run with a timeout guard."""
    return await asyncio.wait_for(
        coordinator.execute_test_from_requirements(
            requirements=requirements,
            initial_url=initial_url,
            context=context,
            precomputed_plan=precomputed_plan,
            triage_result=triage_result,
        ),
        timeout=timeout,
    )


def _print_scope_triage_followups(triage_result: ScopeTriageResult | None) -> None:
    """Display exclusions and ambiguities identified during gating/planning."""
    if triage_result is None:
        return

    exclusions = [
        item.strip() for item in triage_result.explicit_exclusions if item.strip()
    ]
    ambiguities = [
        item.strip() for item in triage_result.ambiguous_points if item.strip()
    ]

    if exclusions:
        console.print("\n[bold cyan]Explicit Exclusions[/bold cyan]")
        for item in exclusions:
            console.print(f"- {item}")

    if ambiguities:
        console.print("\n[bold yellow]Open Ambiguities[/bold yellow]")
        for item in ambiguities:
            console.print(f"- {item}")


def _print_scope_blockers(error: ScopeTriageBlockedError) -> None:
    """Print context/scope blockers."""
    console.print("\n[bold red]Execution blocked before planning[/bold red]")

    blocking = error.blocking_questions or []
    if not blocking and getattr(error, "triage_result", None):
        blocking = getattr(error.triage_result, "blocking_questions", []) or []

    if blocking:
        for item in blocking:
            console.print(f"- {item}")
    else:
        console.print("- Missing required execution context details.")

    additional = error.ambiguous_points or []
    if additional:
        console.print("\n[bold yellow]Additional notes[/bold yellow]")
        for item in additional:
            console.print(f"- {item}")


def _render_plan_table(test_plan: TestPlan) -> None:
    """Render generated plan summary."""
    if not getattr(test_plan, "steps", None):
        console.print("[yellow]No steps were generated for this plan.[/yellow]")
        return

    table = Table(title="Test Plan", show_lines=True)
    table.add_column("Step", style="cyan", width=6)
    table.add_column("Action", style="green")
    table.add_column("Target", style="yellow")
    table.add_column("Expected Result", style="white")

    for i, step in enumerate(test_plan.steps, 1):
        instruction = getattr(step, "action_instruction", None)
        action_label = None
        target_label = None
        expected_label = None

        if instruction is not None:
            action_type = getattr(instruction, "action_type", None)
            action_label = getattr(action_type, "value", None) or action_type
            target_label = getattr(instruction, "target", None)
            expected_label = getattr(instruction, "expected_outcome", None)

        table.add_row(
            str(i),
            action_label or getattr(step, "action", ""),
            target_label or "-",
            expected_label
            or getattr(step, "expected_result", None)
            or getattr(step, "description", ""),
        )

    console.print(table)


async def test_api_connection() -> int:
    """Test OpenAI API connection."""
    console.print("\n[bold cyan]Testing OpenAI API Connection[/bold cyan]")
    try:
        from src.models.openai_client import OpenAIClient

        auth_status = OpenAIAuthManager().get_status()
        client = OpenAIClient()
        response = await client.call(
            messages=[
                {
                    "role": "user",
                    "content": "Say 'API test successful' and nothing else.",
                }
            ],
        )
        if "API test successful" in response["content"]:
            console.print("[green]✓ OpenAI API connection successful![/green]")
            console.print(f"[dim]Model: {response['model']}[/dim]")
            console.print(
                f"[dim]Auth mode: {auth_status.active_mode.replace('_', ' ')}[/dim]"
            )
            return 0
        console.print("[red]✗ Unexpected API response[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]✗ API test failed: {exc}[/red]")
        return 1


def show_version() -> int:
    """Show version information."""
    console.print("\n[bold cyan]HAINDY - Autonomous AI Testing Agent[/bold cyan]")
    console.print("Version: [green]0.1.0[/green]")
    console.print("Python: [dim]3.10+[/dim]")
    return 0


async def read_plan_file(file_path: Path) -> str:
    """Read requirements text from a plan file."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        sys.exit(1)
    return file_path.read_text().strip()


async def read_context_file(file_path: Path) -> str:
    """Read plain-text context used for entrypoint setup viability."""
    if not file_path.exists():
        console.print(f"[red]Error: Context file not found: {file_path}[/red]")
        sys.exit(1)
    content = file_path.read_text().strip()
    if not content:
        console.print(f"[red]Error: Context file is empty: {file_path}[/red]")
        sys.exit(1)
    return content


def _ensure_openai_cu_api_key_for_runtime(settings: Settings) -> None:
    """Reject OpenAI computer-use runs when no API key is configured."""
    provider = str(getattr(settings, "cu_provider", "")).strip().lower()
    api_key = str(getattr(settings, "openai_api_key", "") or "").strip()
    if provider == "openai" and not api_key:
        raise ValueError(
            "OpenAI computer-use requires HAINDY_OPENAI_API_KEY. Codex OAuth only "
            "applies to non-CU OpenAI requests."
        )


def _format_optional_timestamp(value: datetime | None) -> str:
    """Format an optional timestamp for CLI output."""
    if value is None:
        return "n/a"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _open_browser(url: str) -> bool:
    """Open a URL in the user's browser."""
    return webbrowser.open(url)


async def _handle_codex_auth_command(command: str) -> int:
    """Execute a Codex OAuth CLI command."""
    auth_manager = OpenAIAuthManager()

    if command == "status":
        status = auth_manager.get_status()
        console.print("\n[bold cyan]OpenAI Auth Status[/bold cyan]")
        console.print(f"Active mode: [green]{status.active_mode}[/green]")
        console.print(
            f"Codex OAuth: {'connected' if status.oauth_connected else 'not connected'}"
        )
        if status.oauth_account_label:
            console.print(f"Account: [dim]{status.oauth_account_label}[/dim]")
        if status.oauth_connected:
            console.print(
                f"Expires: [dim]{_format_optional_timestamp(status.oauth_expires_at)}[/dim]"
            )
            if status.oauth_expired:
                console.print("[yellow]Stored OAuth session is expired.[/yellow]")
        console.print(
            f"API key configured: {'yes' if status.api_key_available else 'no'}"
        )
        return 0

    if command == "logout":
        auth_manager.clear_oauth_credentials()
        status = auth_manager.get_status()
        console.print("[green]Codex OAuth session cleared.[/green]")
        console.print(f"[dim]Active mode: {status.active_mode}[/dim]")
        return 0

    return await _login_with_codex_oauth(auth_manager)


async def _login_with_codex_oauth(auth_manager: OpenAIAuthManager) -> int:
    """Run the interactive browser-based Codex OAuth login flow."""
    oauth_client = CodexOAuthClient()
    pkce = oauth_client.generate_pkce()
    authorize_url = oauth_client.build_authorize_url(
        pkce.state,
        pkce.code_challenge,
    )
    callback_capture = OAuthCallbackCapture(CODEX_OAUTH_REDIRECT_URI)
    redirect_url: str | None = None
    callback_listening = False

    try:
        await callback_capture.start()
        callback_listening = True
    except OSError as exc:
        console.print(
            f"[yellow]Callback listener unavailable ({exc}). Falling back to manual redirect capture.[/yellow]"
        )

    console.print("\n[bold cyan]OpenAI Codex OAuth Login[/bold cyan]")
    console.print("[dim]Opening the authorization URL in your browser...[/dim]")
    opened = _open_browser(authorize_url)
    if not opened:
        console.print("[yellow]Browser open failed. Open this URL manually:[/yellow]")
        console.print(authorize_url)

    try:
        if callback_listening:
            console.print("[dim]Waiting for the localhost callback...[/dim]")
            redirect_url = await callback_capture.wait_for_redirect(timeout_seconds=120)
            if redirect_url is None:
                console.print(
                    "[yellow]No callback received within 120 seconds. Falling back to manual redirect capture.[/yellow]"
                )
    finally:
        await callback_capture.close()

    if not redirect_url:
        console.print(
            "[dim]After authorizing, paste the final redirect URL below.[/dim]"
        )
        redirect_url = console.input("Redirect URL: ").strip()
        if not redirect_url:
            console.print("[red]No redirect URL provided.[/red]")
            return 1

    try:
        code, state = oauth_client.parse_redirect_url(redirect_url)
        if state != pkce.state:
            raise ValueError("OAuth callback state mismatch.")
        token = await oauth_client.exchange_authorization_code(code, pkce.code_verifier)
        credentials = auth_manager.save_oauth_token_bundle(token)
    except Exception as exc:
        console.print(f"[red]Codex OAuth login failed: {exc}[/red]")
        return 1

    console.print("[green]Codex OAuth login successful.[/green]")
    if credentials.account_label:
        console.print(f"[dim]Account: {credentials.account_label}[/dim]")
    console.print(
        f"[dim]Expires: {_format_optional_timestamp(credentials.expires_at)}[/dim]"
    )
    return 0


async def async_main(args: list[str] | None = None) -> int:
    """Async main entrypoint."""
    argv = list(args) if args is not None else sys.argv[1:]
    if is_tool_call_command(argv):
        if argv and argv[0] == "__tool_call_daemon":
            return await run_tool_call_daemon_cli(argv)
        return await run_tool_call_cli(argv)

    parser = create_parser()
    parsed_args = parser.parse_args(argv)

    if parsed_args.version:
        return show_version()
    if parsed_args.test_api:
        return await test_api_connection()

    settings = get_settings()
    if parsed_args.debug:
        settings.debug_mode = True
        settings.log_level = "DEBUG"
    settings.log_format = "json" if parsed_args.verbose else "text"

    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file,
    )

    # Initialize security components for side effects / parity with existing flow
    RateLimiter()
    DataSanitizer()

    if parsed_args.codex_auth:
        return await _handle_codex_auth_command(parsed_args.codex_auth)

    if not parsed_args.plan:
        parser.print_help()
        return 1
    if not parsed_args.context:
        console.print(
            "[red]Error: --context is required when running with --plan[/red]"
        )
        return 1

    requirements = await read_plan_file(parsed_args.plan)
    context_text = await read_context_file(parsed_args.context)
    automation_backend = (
        "mobile_adb"
        if parsed_args.mobile
        else normalize_automation_backend(
            getattr(settings, "automation_backend", "desktop")
        )
    )

    return await run_test(
        requirements=requirements,
        context_text=context_text,
        output_dir=parsed_args.output,
        report_format=parsed_args.format,
        timeout=parsed_args.timeout,
        max_steps=parsed_args.max_steps,
        berserk=parsed_args.berserk,
        record_override=parsed_args.record,
        automation_backend=automation_backend,
    )


def main(args: list[str] | None = None) -> int:
    """Main entry point for HAINDY."""
    try:
        return asyncio.run(async_main(args))
    except Exception as exc:
        console.print(f"[red]Fatal error: {exc}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
