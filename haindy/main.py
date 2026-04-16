"""HAINDY CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from haindy.agents import ScopeTriageAgent, SituationalAgent, TestPlannerAgent
from haindy.auth import OpenAIAuthManager
from haindy.cli.auth_commands import (
    handle_auth_clear,
    handle_auth_login,
    handle_auth_status,
)
from haindy.cli.config_commands import handle_config_migrate, handle_config_show
from haindy.cli.provider_commands import (
    handle_provider_list,
    handle_provider_set,
    handle_provider_set_computer_use,
    handle_provider_set_model,
)
from haindy.config.settings import Settings, get_settings
from haindy.config.settings_file import ensure_settings_skeleton
from haindy.core.types import ScopeTriageResult, TestPlan, TestState
from haindy.desktop.controller import DesktopController
from haindy.desktop.screen_recorder import ScreenRecorder, ScreenRecorderError
from haindy.error_handling import ScopeTriageBlockedError
from haindy.feedback import build_issue_url, console_feedback_hint
from haindy.mobile.controller import MobileController
from haindy.mobile.ios_controller import IOSController
from haindy.monitoring.debug_logger import initialize_debug_logger
from haindy.monitoring.logger import get_logger, get_run_id, setup_logging
from haindy.monitoring.reporter import TestReporter
from haindy.orchestration.coordinator import WorkflowCoordinator
from haindy.orchestration.scope_pipeline import run_scope_triage_and_plan
from haindy.orchestration.state_manager import StateManager
from haindy.runtime.agent_factory import AgentFactory
from haindy.runtime.environment import normalize_automation_backend
from haindy.runtime.execution_context_builder import build_execution_context_bundle
from haindy.security.rate_limiter import RateLimiter
from haindy.security.sanitizer import DataSanitizer
from haindy.tool_call_mode.cli import (
    ToolCallUsageError,
    create_tool_call_parser,
    dispatch_tool_call_args,
    is_tool_call_command,
    run_tool_call_cli,
    run_tool_call_daemon_cli,
)

console = Console()
logger = get_logger("main")


def create_parser() -> argparse.ArgumentParser:
    """Create the shared public command line parser."""
    return create_tool_call_parser()


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
    automation_controller: (
        DesktopController | MobileController | IOSController | None
    ) = None
    coordinator: WorkflowCoordinator | None = None
    screen_recorder: ScreenRecorder | None = None
    recording_artifact_path: Path | None = None
    settings = get_settings()

    try:
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
        if (
            automation_backend == "mobile_ios"
            and assessment.target_type != "mobile_ios"
        ):
            raise ScopeTriageBlockedError(
                triage_result=ScopeTriageResult(
                    in_scope="",
                    explicit_exclusions=[],
                    ambiguous_points=assessment.notes,
                    blocking_questions=[
                        "--ios requires mobile_ios context (provide iOS idb setup details)."
                    ],
                )
            )
        if (
            assessment.target_type == "mobile_ios"
            and automation_backend != "mobile_ios"
        ):
            raise ScopeTriageBlockedError(
                triage_result=ScopeTriageResult(
                    in_scope="",
                    explicit_exclusions=[],
                    ambiguous_points=assessment.notes,
                    blocking_questions=[
                        "Context requires mobile_ios execution. Re-run with --ios."
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

        _print_feedback_hint(
            command="run",
            run_id=test_run_id,
            exit_reason=str(status_value),
            error=None
            if status_value in success_statuses
            else f"Status: {status_value}",
        )

        return 0 if status_value in success_statuses else 1

    except ScopeTriageBlockedError as scope_error:
        logger.warning("Context/scope gate blocked planning", exc_info=True)
        _print_scope_blockers(scope_error)
        _print_feedback_hint(
            command="run",
            run_id=get_run_id(),
            exit_reason="scope_blocked",
            error=str(scope_error) or "Scope triage blocked planning.",
        )
        return 1
    except asyncio.TimeoutError:
        console.print(
            f"\n[red]Error: Test execution timed out after {timeout} seconds[/red]"
        )
        _print_feedback_hint(
            command="run",
            run_id=get_run_id(),
            exit_reason="timeout",
            error=f"Test execution timed out after {timeout} seconds.",
        )
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Test execution interrupted by user[/yellow]")
        return 130
    except Exception as exc:
        console.print(f"\n[red]Error during test execution: {exc}[/red]")
        logger.exception("Test execution failed")
        _print_feedback_hint(
            command="run",
            run_id=get_run_id(),
            exit_reason=type(exc).__name__,
            error=str(exc),
        )
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
) -> tuple[DesktopController | MobileController | IOSController, WorkflowCoordinator]:
    """Build and initialize the automation backend/coordinator stack."""
    normalized_backend = normalize_automation_backend(backend)
    if normalized_backend == "mobile_adb":
        automation_controller: DesktopController | MobileController | IOSController = (
            MobileController()
        )
    elif normalized_backend == "mobile_ios":
        automation_controller = IOSController()
    elif sys.platform == "darwin":
        from haindy.macos.controller import MacOSController

        automation_controller = MacOSController()  # type: ignore[assignment]
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


def _print_feedback_hint(
    *,
    command: str,
    run_id: str | None,
    exit_reason: str | None,
    error: str | None,
) -> None:
    """Print a pre-filled GitHub issue URL unless the user has opted out."""

    url = build_issue_url(
        command=command,
        run_id=run_id,
        exit_reason=exit_reason,
        error=error,
    )
    if url:
        console.print(console_feedback_hint(url))


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
        from haindy.models.openai_client import OpenAIClient

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
    console.print("Version: [green]0.2.0[/green]")
    console.print("Python: [dim]3.11+[/dim]")
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


def _validate_auth_for_run(settings: Settings) -> list[str]:
    """Return a list of actionable error messages for missing auth. Empty = ready to run."""
    issues: list[str] = []

    auth_manager = OpenAIAuthManager()
    codex_status = auth_manager.get_status()

    agent_provider = str(getattr(settings, "agent_provider", "openai")).strip().lower()
    _agent_key_field = {
        "google": "vertex_api_key",
        "anthropic": "anthropic_api_key",
    }.get(agent_provider, "openai_api_key")

    if agent_provider in ("openai", "openai-codex"):
        has_agent_key = bool(settings.openai_api_key) or (
            codex_status.oauth_connected and not codex_status.oauth_expired
        )
    else:
        has_agent_key = bool(getattr(settings, _agent_key_field, ""))

    if not has_agent_key:
        if agent_provider in ("google", "anthropic"):
            issues.append(
                f"No API key for agent provider '{agent_provider}'. "
                f"Run: haindy auth login {agent_provider}"
            )
        else:
            issues.append(
                "No OpenAI credentials for non-CU calls (planning, analysis). "
                "Run: haindy auth login openai  or  haindy auth login openai-codex"
            )

    provider = str(settings.cu_provider).strip().lower()
    _cu_key_field = {"google": "vertex_api_key", "anthropic": "anthropic_api_key"}.get(
        provider, "openai_api_key"
    )
    has_cu_key = bool(getattr(settings, _cu_key_field, ""))
    if not has_cu_key:
        cu_login_arg = provider if provider in ("google", "anthropic") else "openai"
        issues.append(
            f"No API key for computer-use provider '{provider}'. "
            f"Run: haindy auth login {cu_login_arg}"
        )

    return issues


def _any_auth_configured() -> bool:
    """Return True if any credentials (API keys or Codex OAuth) are stored."""
    from haindy.auth.credentials import list_configured_providers

    if any(list_configured_providers().values()):
        return True
    return bool(OpenAIAuthManager().get_status().oauth_connected)


_SETUP_MARKER = Path.home() / ".haindy" / "setup_complete"


def _is_setup_complete() -> bool:
    return _SETUP_MARKER.exists()


async def async_main(args: list[str] | None = None) -> int:
    """Async main entrypoint."""
    argv = list(args) if args is not None else sys.argv[1:]
    if argv and argv[0] == "__tool_call_daemon":
        return await run_tool_call_daemon_cli(argv)

    ensure_settings_skeleton(Path("~/.haindy/settings.json").expanduser())

    parser = create_parser()
    if not argv:
        if not _any_auth_configured():
            console.print(
                "[yellow]No credentials configured. "
                "Run: haindy auth login openai (or google / anthropic / openai-codex)[/yellow]"
            )
            console.print("")
        parser.print_help()
        return 1

    try:
        parsed_args = parser.parse_args(argv)
    except ToolCallUsageError as exc:
        if is_tool_call_command(argv):
            return await run_tool_call_cli(argv)
        console.print(f"[red]Error: {exc}[/red]")
        console.print("")
        parser.print_help()
        return 2

    if getattr(parsed_args, "tool_command", None):
        envelope, exit_code = await dispatch_tool_call_args(parsed_args)
        print(envelope.model_dump_json())
        return exit_code

    command = parsed_args.command

    if command == "version":
        return show_version()

    if command == "setup":
        from haindy.cli.setup_wizard import run_setup_wizard

        return run_setup_wizard(non_interactive=parsed_args.non_interactive)

    if command == "doctor":
        from haindy.cli.doctor import run_doctor

        return run_doctor()

    if command == "test-api":
        return await test_api_connection()

    if command == "auth":
        auth_command = getattr(parsed_args, "auth_command", None)
        if not auth_command:
            console.print(
                "[red]Usage: haindy auth <login|status|clear> [provider][/red]"
            )
            return 1
        if auth_command == "status":
            return await handle_auth_status()
        if auth_command == "login":
            return await handle_auth_login(parsed_args.provider)
        if auth_command == "clear":
            return await handle_auth_clear(parsed_args.provider)

    if command == "config":
        config_command = getattr(parsed_args, "config_command", None)
        if not config_command:
            console.print("[red]Usage: haindy config <show|migrate> [path][/red]")
            return 1
        if config_command == "show":
            return await handle_config_show()
        if config_command == "migrate":
            return await handle_config_migrate(Path(parsed_args.dotenv_path))

    if command == "provider":
        provider_command = getattr(parsed_args, "provider_command", None)
        if not provider_command:
            console.print(
                "[red]Usage: haindy provider <list|set|set-computer-use|set-model> ...[/red]"
            )
            return 1
        if provider_command == "list":
            return handle_provider_list()
        if provider_command == "set":
            return handle_provider_set(parsed_args.provider)
        if provider_command == "set-computer-use":
            return handle_provider_set_computer_use(parsed_args.provider)
        if provider_command == "set-model":
            return handle_provider_set_model(
                parsed_args.provider,
                parsed_args.model,
                computer_use=parsed_args.computer_use,
            )

    if command == "run":
        if not _is_setup_complete():
            console.print(
                "Haindy is not set up yet. Run:\n\n"
                "  haindy setup\n\n"
                "Or, if you have Claude Code, Codex, or OpenCode installed, "
                "install the setup skill:\n\n"
                "  haindy setup --install-skill\n\n"
                "Then run /haindy-setup inside your AI coding tool."
            )
            sys.exit(1)

        if not parsed_args.plan:
            console.print("[red]Error: --plan is required[/red]")
            return 1
        if not parsed_args.context:
            console.print(
                "[red]Error: --context is required when running with --plan[/red]"
            )
            return 1

        settings = get_settings()
        if parsed_args.debug:
            settings.debug_mode = True
            settings.log_level = "DEBUG"
        setup_logging(
            log_level=settings.log_level,
            log_format=settings.log_format,
            log_file=settings.log_file,
        )
        RateLimiter()
        DataSanitizer()

        auth_issues = _validate_auth_for_run(settings)
        if auth_issues:
            for issue in auth_issues:
                console.print(f"[red]{issue}[/red]")
            return 1

        requirements = await read_plan_file(parsed_args.plan)
        context_text = await read_context_file(parsed_args.context)
        automation_backend = (
            "mobile_ios"
            if parsed_args.ios
            else (
                "mobile_adb"
                if parsed_args.mobile
                else normalize_automation_backend(
                    getattr(settings, "automation_backend", "desktop")
                )
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

    parser.print_help()
    return 1


def main(args: list[str] | None = None) -> int:
    """Main entry point for HAINDY."""
    try:
        return asyncio.run(async_main(args))
    except Exception as exc:
        console.print(f"[red]Fatal error: {exc}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
