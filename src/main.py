"""HAINDY desktop-first CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.scope_triage import ScopeTriageAgent
from src.agents.situational_agent import SituationalAgent
from src.agents.test_planner import TestPlannerAgent
from src.config.settings import get_settings
from src.core.types import ScopeTriageResult, TestPlan, TestState
from src.desktop.controller import DesktopController
from src.desktop.screen_recorder import ScreenRecorder, ScreenRecorderError
from src.error_handling import ScopeTriageBlockedError
from src.monitoring.debug_logger import initialize_debug_logger
from src.monitoring.logger import get_logger, get_run_id, setup_logging
from src.monitoring.reporter import TestReporter
from src.orchestration.communication import MessageBus
from src.orchestration.coordinator import WorkflowCoordinator
from src.orchestration.scope_pipeline import run_scope_triage_and_plan
from src.orchestration.state_manager import StateManager
from src.security.rate_limiter import RateLimiter
from src.security.sanitizer import DataSanitizer

console = Console()
logger = get_logger("main")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="HAINDY - Autonomous AI Testing Agent v0.1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full desktop-first execution (required)
  python -m src.main --plan requirements.md --context execution_context.txt

  # Berserk mode
  python -m src.main --berserk --plan requirements.md --context execution_context.txt

  # Test your OpenAI API configuration
  python -m src.main --test-api
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
        help="Test OpenAI API key configuration",
    )
    input_group.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    parser.add_argument(
        "--context",
        type=Path,
        help="Path to plain-text execution context file (required for execution)",
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
    settings,
) -> Tuple[ScopeTriageAgent, TestPlannerAgent, SituationalAgent]:
    """Instantiate planning + setup agents from current settings."""
    triage_cfg = settings.get_agent_model_config("scope_triage")
    planner_cfg = settings.get_agent_model_config("test_planner")
    situational_cfg = settings.get_agent_model_config("situational_agent")

    triage_agent = ScopeTriageAgent(
        name="ScopeTriage",
        model=triage_cfg.model,
        temperature=triage_cfg.temperature,
        reasoning_level=triage_cfg.reasoning_level,
        modalities=triage_cfg.modalities,
    )
    planner = TestPlannerAgent(
        name="TestPlanner",
        model=planner_cfg.model,
        temperature=planner_cfg.temperature,
        reasoning_level=planner_cfg.reasoning_level,
        modalities=planner_cfg.modalities,
    )
    situational_agent = SituationalAgent(
        name="SituationalAgent",
        model=situational_cfg.model,
        temperature=situational_cfg.temperature,
        reasoning_level=situational_cfg.reasoning_level,
        modalities=situational_cfg.modalities,
    )
    return triage_agent, planner, situational_agent


async def run_test(
    requirements: str,
    context_text: str,
    output_dir: Optional[Path] = None,
    report_format: str = "html",
    timeout: int = 7200,
    max_steps: int = 50,
    berserk: bool = False,
    record_override: Optional[bool] = None,
) -> int:
    """Run a test with mandatory requirements and context inputs."""
    desktop_controller: Optional[DesktopController] = None
    coordinator: Optional[WorkflowCoordinator] = None
    screen_recorder: Optional[ScreenRecorder] = None
    recording_artifact_path: Optional[Path] = None
    settings = get_settings()

    try:
        test_run_id = get_run_id()
        if test_run_id == "unknown":
            test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        debug_logger = initialize_debug_logger(test_run_id)
        console.print(f"[dim]Debug logging initialized for run: {test_run_id}[/dim]")

        console.print(
            Panel.fit(
                "[bold cyan]HAINDY - Desktop-First Autonomous Testing Agent[/bold cyan]",
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
        if not assessment.sufficient:
            raise ScopeTriageBlockedError(
                triage_result=ScopeTriageResult(
                    in_scope="",
                    explicit_exclusions=[],
                    ambiguous_points=assessment.notes,
                    blocking_questions=assessment.as_blocking_questions(),
                )
            )

        planning_context = {
            "execution_context": context_text,
            "target_type": assessment.target_type,
            "web_url": assessment.setup.web_url,
            "app_name": assessment.setup.app_name,
            "launch_command": assessment.setup.launch_command,
            "maximize": str(assessment.setup.maximize),
        }

        console.print("[yellow]Generating test plan...[/yellow]")
        test_plan, triage_result = await run_scope_triage_and_plan(
            requirements=requirements,
            planner=planner,
            triage_agent=triage_agent,
            context=planning_context,
        )

        console.print("[cyan]Initializing desktop runtime...[/cyan]")
        desktop_controller, coordinator = await _create_coordinator_stack(max_steps=max_steps)

        should_record = bool(settings.enable_screen_recording)
        if record_override is not None:
            should_record = bool(record_override)
        if should_record:
            screen_recorder = ScreenRecorder(
                output_dir=settings.screen_recording_output_dir,
                framerate=settings.screen_recording_framerate,
                draw_cursor=settings.screen_recording_draw_cursor,
                filename_prefix=settings.screen_recording_prefix,
            )
            try:
                recording_artifact_path = screen_recorder.start()
                console.print(f"[dim]Screen recording started:[/dim] {recording_artifact_path}")
            except ScreenRecorderError as exc:
                logger.warning("Unable to start screen recording", exc_info=True, extra={"error": str(exc)})
                screen_recorder = None

        console.print("[cyan]Preparing entrypoint state with Situational Agent...[/cyan]")
        await situational_agent.prepare_entrypoint(desktop_controller.driver, assessment)

        test_context = {
            "execution_context": context_text,
            "target_type": assessment.target_type,
            "entry_setup": {
                "web_url": assessment.setup.web_url,
                "app_name": assessment.setup.app_name,
                "launch_command": assessment.setup.launch_command,
                "maximize": assessment.setup.maximize,
            },
            "setup_notes": assessment.notes,
        }

        console.print("[cyan]Running test...[/cyan]")
        test_state = await _run_with_timeout(
            coordinator=coordinator,
            requirements=requirements,
            precomputed_plan=test_plan,
            triage_result=triage_result,
            timeout=timeout,
            context=test_context,
            initial_url=assessment.setup.web_url or None,
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

        raw_status = getattr(test_state, "status", getattr(test_state, "test_status", "unknown"))
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
            stored_triage = coordinator.get_scope_triage_result(test_state.test_plan.plan_id)
            if stored_triage:
                final_triage = stored_triage
        _print_scope_triage_followups(final_triage)

        if output_dir is None:
            output_dir = debug_logger.reports_dir
        output_dir.mkdir(exist_ok=True)

        action_storage = None
        if coordinator and hasattr(coordinator, "_agents") and "test_runner" in coordinator._agents:
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
        console.print(f"\n[red]Error: Test execution timed out after {timeout} seconds[/red]")
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
                logger.warning("Screen recording stop failed during cleanup", exc_info=True)
        if coordinator:
            await coordinator.cleanup()
        if desktop_controller:
            await desktop_controller.stop()


async def _create_coordinator_stack(
    max_steps: int,
) -> Tuple[DesktopController, WorkflowCoordinator]:
    """Build and initialize the desktop/coordinator stack."""
    desktop_controller = DesktopController()
    await desktop_controller.start()

    coordinator = WorkflowCoordinator(
        message_bus=MessageBus(),
        state_manager=StateManager(),
        automation_driver=desktop_controller.driver,
        max_steps=max_steps,
    )
    await coordinator.initialize()
    return desktop_controller, coordinator


async def _run_with_timeout(
    coordinator: WorkflowCoordinator,
    requirements: str,
    precomputed_plan: TestPlan,
    triage_result: ScopeTriageResult,
    timeout: int,
    context: Optional[Dict[str, Any]] = None,
    initial_url: Optional[str] = None,
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


def _print_scope_triage_followups(triage_result: Optional[ScopeTriageResult]) -> None:
    """Display exclusions and ambiguities identified during gating/planning."""
    if triage_result is None:
        return

    exclusions = [item.strip() for item in triage_result.explicit_exclusions if item.strip()]
    ambiguities = [item.strip() for item in triage_result.ambiguous_points if item.strip()]

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

        client = OpenAIClient()
        response = await client.call(
            messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}],
        )
        if "API test successful" in response["content"]:
            console.print("[green]✓ OpenAI API connection successful![/green]")
            console.print(f"[dim]Model: {response['model']}[/dim]")
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


async def async_main(args: Optional[list[str]] = None) -> int:
    """Async main entrypoint."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

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

    if not parsed_args.plan:
        parser.print_help()
        return 1
    if not parsed_args.context:
        console.print("[red]Error: --context is required when running with --plan[/red]")
        return 1

    requirements = await read_plan_file(parsed_args.plan)
    context_text = await read_context_file(parsed_args.context)

    return await run_test(
        requirements=requirements,
        context_text=context_text,
        output_dir=parsed_args.output,
        report_format=parsed_args.format,
        timeout=parsed_args.timeout,
        max_steps=parsed_args.max_steps,
        berserk=parsed_args.berserk,
        record_override=parsed_args.record,
    )


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for HAINDY."""
    try:
        return asyncio.run(async_main(args))
    except Exception as exc:
        console.print(f"[red]Fatal error: {exc}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
