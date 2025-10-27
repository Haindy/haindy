"""
HAINDY - Autonomous AI Testing Agent
Main entry point for the application.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from src.browser.controller import BrowserController
from src.agents.scope_triage import ScopeTriageAgent
from src.agents.test_planner import TestPlannerAgent
from src.config.settings import get_settings
from src.error_handling import ScopeTriageBlockedError
from src.models.openai_client import ResponseStreamObserver
from src.monitoring.logger import get_logger, setup_logging
from src.monitoring.reporter import TestReporter
from src.monitoring.debug_logger import initialize_debug_logger
from src.orchestration.communication import MessageBus
from src.orchestration.coordinator import WorkflowCoordinator
from src.orchestration.state_manager import StateManager
from src.core.types import ScopeTriageResult, TestPlan, TestState
from src.orchestration.scope_pipeline import run_scope_triage_and_plan
from src.security.rate_limiter import RateLimiter
from src.security.sanitizer import DataSanitizer

console = Console()
logger = get_logger("main")


class PlanGenerationStreamObserver(ResponseStreamObserver):
    """Update the CLI while the Responses API streams tokens."""

    def __init__(self, progress: Progress, task_id: int) -> None:
        self.progress = progress
        self.task_id = task_id
        self.output_tokens = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self._had_usage_delta = False
        self._final_usage: Optional[Dict[str, int]] = None
        self._saw_error = False
        self.estimated_tokens = 0
        self._last_delta_tokens = 0
        self._last_delta_chars = 0
        self._has_stream_activity = False

    def on_stream_start(self) -> None:
        self._safe_update("[cyan]GPT-5 thinking…[/cyan]")

    def on_text_delta(self, delta: str) -> None:
        # Text deltas are not displayed to avoid noisy console output.
        return

    def on_token_progress(
        self, total_tokens: int, delta_tokens: int, delta_chars: int
    ) -> None:
        self.estimated_tokens = max(self.estimated_tokens, total_tokens)
        self._last_delta_tokens = delta_tokens
        self._last_delta_chars = delta_chars
        self._has_stream_activity = True
        descriptor = self._format_progress_label()
        self._safe_update(descriptor)

    def on_usage_delta(self, delta: Dict[str, int]) -> None:
        self._had_usage_delta = True
        self._has_stream_activity = True
        self.output_tokens += int(delta.get("output_tokens", 0))
        self.total_tokens = max(self.total_tokens, int(delta.get("total_tokens", 0)))
        self.prompt_tokens = max(self.prompt_tokens, int(delta.get("input_tokens", 0)))
        self._safe_update(
            f"[cyan]GPT-5 planning… {self.output_tokens} output tokens[/cyan]"
        )

    def on_usage_total(self, totals: Dict[str, int]) -> None:
        self._final_usage = {
            "input_tokens": int(totals.get("input_tokens", 0)),
            "output_tokens": int(totals.get("output_tokens", 0)),
            "total_tokens": int(totals.get("total_tokens", 0)),
        }
        self.output_tokens = max(self.output_tokens, self._final_usage["output_tokens"])
        self.total_tokens = max(self.total_tokens, self._final_usage["total_tokens"])
        self.prompt_tokens = max(self.prompt_tokens, self._final_usage["input_tokens"])
        self.estimated_tokens = max(self.estimated_tokens, self._final_usage["output_tokens"])

    def on_error(self, error: object) -> None:
        self._saw_error = True
        self._safe_update(
            "[yellow]Streaming interrupted, retrying with fallback...[/yellow]"
        )

    def on_stream_end(self) -> None:
        if (
            not self._saw_error
            and not self._had_usage_delta
            and self.estimated_tokens == 0
            and not self._has_stream_activity
        ):
            self._safe_update("[cyan]Awaiting final response…[/cyan]")

    def mark_complete(self) -> None:
        if self._final_usage:
            message = (
                "[green]Test plan generated"
                f" ({self._final_usage['output_tokens']} output tokens)[/green]"
            )
        elif self._had_usage_delta or self.output_tokens:
            message = (
                "[green]Test plan generated"
                f" ({self.output_tokens} output tokens)[/green]"
            )
        elif self.estimated_tokens:
            message = (
                "[green]Test plan generated"
                f" (~{self.estimated_tokens} output tokens)[/green]"
            )
        else:
            label = (
                "fallback (usage metrics unavailable)"
                if self._saw_error
                else "usage metrics unavailable"
            )
            message = (
                "[green]Test plan generated[/green] "
                f"[yellow]({label})[/yellow]"
            )

        self._safe_update(message)

    def has_usage_data(self) -> bool:
        return (
            bool(self._final_usage and any(self._final_usage.values()))
            or self._had_usage_delta
            or self.estimated_tokens > 0
        )

    def get_usage_totals(self) -> Optional[Dict[str, int]]:
        if self._final_usage:
            return dict(self._final_usage)

        if self._had_usage_delta or self.output_tokens:
            return {
                "input_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens or self.output_tokens,
            }

        if self.estimated_tokens:
            return {
                "input_tokens": 0,
                "output_tokens": self.estimated_tokens,
                "total_tokens": self.estimated_tokens,
            }

        return None

    def _safe_update(self, description: str) -> None:
        try:
            self.progress.update(self.task_id, description=description)
        except Exception:  # pragma: no cover - defensive UI update guard
            return

    def _format_progress_label(self) -> str:
        if self._last_delta_tokens and self._last_delta_chars:
            return (
                "[cyan]GPT-5 planning… "
                f"{self.estimated_tokens} tokens (+{self._last_delta_tokens} | "
                f"{self._last_delta_chars} chars)[/cyan]"
            )
        return f"[cyan]GPT-5 planning… {self.estimated_tokens} tokens[/cyan]"
def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="HAINDY - Autonomous AI Testing Agent v0.1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive requirements mode
  python -m src.main --requirements
  
  # Test from a document file (PRD, design doc, etc.)
  python -m src.main --plan requirements.md
  
  # Run existing test scenario
  python -m src.main --json-test-plan test_scenarios/login_test.json
  
  # Berserk mode - full autonomous operation
  python -m src.main --berserk --plan requirements.pdf
  
  # Test your OpenAI API configuration
  python -m src.main --test-api
        """,
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-r", "--requirements",
        action="store_true",
        help="Interactive mode - enter test requirements via prompt",
    )
    input_group.add_argument(
        "-p", "--plan",
        type=Path,
        help="Path to document file with test requirements (any format)",
    )
    input_group.add_argument(
        "-j", "--json-test-plan",
        type=Path,
        help="Path to existing test scenario JSON file",
    )
    
    # Utility commands
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
    
    # Execution options
    parser.add_argument(
        "-u", "--url",
        help="Initial URL to navigate to",
    )
    parser.add_argument(
        "--berserk",
        action="store_true",
        help="Berserk mode - aggressive autonomous operation without confirmations",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Generate test plan without executing",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
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
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for test results (default: reports/)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "html", "markdown"],
        default="html",
        help="Report format (default: html)",
    )
    
    # Advanced options
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


def load_scenario(scenario_path: Path) -> dict:
    """Load test scenario from JSON file."""
    try:
        with open(scenario_path, "r") as f:
            scenario = json.load(f)
        
        # Validate required fields
        required_fields = ["name", "requirements", "url"]
        missing_fields = [f for f in required_fields if f not in scenario]
        if missing_fields:
            raise ValueError(f"Scenario missing required fields: {missing_fields}")
        
        return scenario
    except FileNotFoundError:
        console.print(f"[red]Error: Scenario file not found: {scenario_path}[/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in scenario file: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading scenario: {e}[/red]")
        sys.exit(1)


async def run_test(
    requirements: str,
    url: str,
    plan_only: bool = False,
    headless: bool = True,
    output_dir: Optional[Path] = None,
    report_format: str = "html",
    timeout: int = 7200,
    max_steps: int = 50,
    berserk: bool = False,
) -> int:
    """
    Run a test with the given requirements.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Initialize components
    browser_controller = None
    coordinator = None
    settings = get_settings()
    use_progress = settings.log_format == "json"
    
    try:
        # Generate test run ID
        test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize debug logger
        debug_logger = initialize_debug_logger(test_run_id)
        console.print(f"[dim]Debug logging initialized for run: {test_run_id}[/dim]")
        
        # Show startup banner
        console.print(Panel.fit(
            "[bold cyan]HAINDY - Autonomous AI Testing Agent[/bold cyan]\n"
            "Orchestrating multi-agent test execution",
            border_style="cyan",
        ))
        
        if berserk:
            console.print("\n[bold red]BERSERK MODE ACTIVATED[/bold red]")
            console.print("[yellow]Running in fully autonomous mode - no confirmations![/yellow]\n")

        if plan_only:
            console.print("\n[dim]Skipping browser startup (--plan-only mode). Only the Test Planner will run.[/dim]")
            console.print("\n[yellow]Generating test plan...[/yellow]")

            test_plan, triage_result = await _generate_plan_only_plan(requirements)

            _render_plan_table(test_plan)
            _print_plan_storage_locations(test_plan)
            _print_scope_triage_followups(triage_result)

            return 0

        # Initialize core components
        if use_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                init_task = progress.add_task("[cyan]Initializing components...", total=None)
                browser_controller, coordinator = await _create_coordinator_stack(
                    headless=headless,
                    max_steps=max_steps,
                )
                progress.update(init_task, completed=True)
        else:
            console.print("[cyan]Initializing components...[/cyan]")
            browser_controller, coordinator = await _create_coordinator_stack(
                headless=headless,
                max_steps=max_steps,
            )
            console.print("[green]Components initialized.[/green]")
        
        # Navigate to initial URL
        console.print(f"\n[cyan]Navigating to:[/cyan] {url}")
        await browser_controller.navigate(url)
        
        # Execute test
        if use_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                exec_task = progress.add_task("[cyan]Running test...", total=None)
                
                test_state = await _run_with_timeout(
                    coordinator=coordinator,
                    requirements=requirements,
                    url=url,
                    timeout=timeout,
                )
                
                progress.update(exec_task, completed=True)
        else:
            console.print("[cyan]Running test...[/cyan]")
            test_state = await _run_with_timeout(
                coordinator=coordinator,
                requirements=requirements,
                url=url,
                timeout=timeout,
            )
            console.print("[green]Test run completed.[/green]")
        
        # Display results summary
        console.print("\n[bold]Test Execution Summary:[/bold]")
        
        raw_status = getattr(test_state, "status", getattr(test_state, "test_status", "unknown"))
        status_value = raw_status.value if hasattr(raw_status, "value") else raw_status
        success_statuses = {"passed", "completed"}
        status_color = "green" if status_value in success_statuses else "red"
        console.print(f"Status: [{status_color}]{status_value}[/{status_color}]")
        
        # Display test execution metrics from test report
        report = getattr(test_state, "test_report", None)
        if report:
            test_cases = getattr(report, "test_cases", [])
            total_test_cases = len(test_cases)
            completed_test_cases = len(
                [
                    tc
                    for tc in test_cases
                    if (
                        getattr(tc, "status", None).value
                        if hasattr(getattr(tc, "status", None), "value")
                        else getattr(tc, "status", None)
                    )
                    in success_statuses
                ]
            )

            total_steps = sum(getattr(tc, "steps_total", 0) for tc in test_cases)
            completed_steps = sum(getattr(tc, "steps_completed", 0) for tc in test_cases)
            failed_steps = sum(getattr(tc, "steps_failed", 0) for tc in test_cases)
            skipped_steps = max(total_steps - completed_steps - failed_steps, 0)

            console.print(f"Test Cases: [cyan]{completed_test_cases}/{total_test_cases}[/cyan]")
            console.print(f"Total Steps: {total_steps}")
            console.print(f"Completed Steps: [green]{completed_steps}[/green]")
            console.print(f"Failed Steps: [red]{failed_steps}[/red]")
            console.print(f"Skipped Steps: [yellow]{skipped_steps}[/yellow]")

            if getattr(report, "summary", None):
                console.print(f"Success Rate: [cyan]{report.summary.success_rate:.1f}%[/cyan]")
        else:
            console.print("[yellow]No test report available[/yellow]")

        triage_result = None
        if coordinator:
            triage_result = coordinator.get_scope_triage_result(test_state.test_plan.plan_id)
        _print_scope_triage_followups(triage_result)
        
        # Generate report
        if output_dir is None:
            # Use the debug logger's reports directory for organized output
            output_dir = debug_logger.reports_dir
        output_dir.mkdir(exist_ok=True)
        
        console.print(f"\n[cyan]Generating {report_format} report...[/cyan]")
        
        # Get action storage from test runner if available
        action_storage = None
        if coordinator and hasattr(coordinator, '_agents') and 'test_runner' in coordinator._agents:
            test_runner = coordinator._agents['test_runner']
            if hasattr(test_runner, 'get_action_storage'):
                action_storage = test_runner.get_action_storage()
        
        reporter = TestReporter()
        report_path, actions_path = await reporter.generate_report(
            test_state=test_state,
            output_dir=output_dir,
            format=report_format,
            action_storage=action_storage
        )
        
        console.print(f"[green]Report saved to:[/green] {report_path}")
        
        # Print actions file path if it was generated
        if actions_path:
            console.print(f"[green]Actions saved to:[/green] {actions_path}")
        
        # Show debug summary
        debug_summary = debug_logger.get_debug_summary()
        console.print(f"\n[bold]Debug Information:[/bold]")
        console.print(f"Test Run ID: [cyan]{debug_summary['test_run_id']}[/cyan]")
        console.print(f"Debug Directory: [cyan]{debug_summary['debug_directory']}[/cyan]")
        console.print(f"AI Interactions Logged: [green]{debug_summary['ai_interactions']}[/green]")
        console.print(f"Screenshots Saved: [green]{debug_summary['screenshots_saved']}[/green]")
        
        # Return appropriate exit code
        return 0 if status_value in success_statuses else 1
        
    except ScopeTriageBlockedError as scope_error:
        logger.warning("Scope triage blocked planning", exc_info=True)
        _print_scope_blockers(scope_error)
        return 1
    except asyncio.TimeoutError:
        console.print(f"\n[red]Error: Test execution timed out after {timeout} seconds[/red]")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Test execution interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error during test execution: {e}[/red]")
        logger.exception("Test execution failed")
        return 1
    finally:
        # Cleanup
        if coordinator:
            await coordinator.cleanup()
        if browser_controller:
            await browser_controller.stop()


async def _create_coordinator_stack(
    headless: bool,
    max_steps: int,
) -> Tuple[BrowserController, WorkflowCoordinator]:
    """Build and initialize the browser/controller/coordinator stack."""
    browser_controller = BrowserController(headless=headless)
    await browser_controller.start()

    message_bus = MessageBus()
    state_manager = StateManager()

    coordinator = WorkflowCoordinator(
        message_bus=message_bus,
        state_manager=state_manager,
        browser_driver=browser_controller.driver,
        max_steps=max_steps,
    )

    await coordinator.initialize()
    return browser_controller, coordinator


async def _run_with_timeout(
    coordinator: WorkflowCoordinator,
    requirements: str,
    url: str,
    timeout: int,
) -> TestState:
    """Execute the coordinator run with a timeout guard."""
    return await asyncio.wait_for(
        coordinator.execute_test_from_requirements(
            requirements=requirements,
            initial_url=url,
        ),
        timeout=timeout,
    )


async def _generate_plan_only_plan(requirements: str) -> Tuple[TestPlan, ScopeTriageResult]:
    """Create a test plan using the two-pass scope triage pipeline."""
    settings = get_settings()
    triage_cfg = settings.get_agent_model_config("scope_triage")
    planner_cfg = settings.get_agent_model_config("test_planner")

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

    use_progress = settings.log_format == "json"

    if use_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Waiting for GPT-5 to craft the test plan...[/cyan]",
                total=None,
            )
            stream_observer = PlanGenerationStreamObserver(progress, task)
            test_plan, triage_result = await run_scope_triage_and_plan(
                requirements=requirements,
                planner=planner,
                triage_agent=triage_agent,
                planner_kwargs={"stream_observer": stream_observer},
            )
            stream_observer.mark_complete()
            if hasattr(progress, "stop_task"):
                progress.stop_task(task)

        usage_totals = stream_observer.get_usage_totals()
        _print_plan_stats(test_plan, usage_totals, stream_observer.has_usage_data())
    else:
        console.print("[cyan]Generating test plan…[/cyan]")
        test_plan, triage_result = await run_scope_triage_and_plan(
            requirements=requirements,
            planner=planner,
            triage_agent=triage_agent,
        )
        console.print("[green]Test plan ready.[/green]")
        _print_plan_stats(test_plan, None, False)

    return test_plan, triage_result


def _print_plan_stats(
    test_plan,
    usage_totals: Optional[Dict[str, int]],
    usage_available: bool,
) -> None:
    """Render a concise stats block for the generated plan."""
    num_cases = len(getattr(test_plan, "test_cases", []))
    total_steps = sum(len(case.steps) for case in getattr(test_plan, "test_cases", []))
    duration_seconds = getattr(test_plan, "estimated_duration_seconds", None)

    console.print("\n[bold cyan]Test Plan Stats[/bold cyan]")
    console.print(f"Test cases: [green]{num_cases}[/green]")
    console.print(f"Total steps: [green]{total_steps}[/green]")

    if duration_seconds is not None:
        minutes = duration_seconds / 60
        console.print(
            "Estimated duration: "
            f"[green]{duration_seconds}s[/green] (~{minutes:.1f} min)"
        )
    else:
        console.print("Estimated duration: [yellow]not provided[/yellow]")

    if usage_totals and (usage_totals.get("output_tokens") or usage_totals.get("total_tokens")):
        console.print(
            "LLM tokens: "
            f"[green]{usage_totals.get('output_tokens', 0)} output[/green] / "
            f"[green]{usage_totals.get('total_tokens', 0)} total[/green]"
        )
    elif usage_available:
        console.print("LLM tokens: [yellow]usage metrics unavailable[/yellow]")
    else:
        console.print("LLM tokens: [yellow]not reported[/yellow]")


def _print_scope_triage_followups(triage_result: Optional[ScopeTriageResult]) -> None:
    """Display scope exclusions and ambiguities identified during triage."""
    if triage_result is None:
        return

    exclusions = [item.strip() for item in triage_result.explicit_exclusions if item.strip()]
    ambiguities = [item.strip() for item in triage_result.ambiguous_points if item.strip()]

    if not exclusions and not ambiguities:
        return

    if exclusions:
        console.print("\n[bold cyan]Explicit Exclusions[/bold cyan]")
        for item in exclusions:
            console.print(f"- {item}")

    if ambiguities:
        console.print("\n[bold yellow]Scope Ambiguities[/bold yellow]")
        console.print(
            "[yellow]Share these with the requirements author before executing the plan.[/yellow]"
        )
        for item in ambiguities:
            console.print(f"- {item}")


def _print_scope_blockers(error: ScopeTriageBlockedError) -> None:
    """Print blocking questions discovered during scope triage."""
    console.print("\n[bold red]Scope triage blocked test planning[/bold red]")

    blocking = error.blocking_questions or []
    if not blocking and getattr(error, "triage_result", None):
        blocking = getattr(error.triage_result, "blocking_questions", []) or []

    if blocking:
        for item in blocking:
            console.print(f"- {item}")
    else:
        console.print("- Unresolved scope contradictions detected.")

    additional = error.ambiguous_points or []
    if not additional and getattr(error, "triage_result", None):
        additional = getattr(error.triage_result, "ambiguous_points", []) or []

    if additional:
        console.print("\n[bold yellow]Other ambiguities for follow-up[/bold yellow]")
        for item in additional:
            console.print(f"- {item}")

    console.print(
        "\n[yellow]Please update the requirements document to resolve these questions, "
        "then rerun the planner.[/yellow]"
    )


def _render_plan_table(test_plan) -> None:
    """Render the generated test plan as a table in the console."""
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


def _print_plan_storage_locations(test_plan) -> None:
    """Print the directory and filenames where the plan was saved."""
    plan_dir = Path("generated_test_plans") / str(test_plan.plan_id)
    json_path = plan_dir / f"test_plan_{test_plan.plan_id}.json"
    md_path = plan_dir / f"test_plan_{test_plan.plan_id}.md"

    console.print(
        f"\n[green]Plan saved to:[/green] {plan_dir.resolve()}",
        soft_wrap=True,
    )

    files_output = []
    if json_path.exists():
        files_output.append(f" - {json_path.name}")
    else:
        files_output.append(f" - {json_path.name} [yellow](not found)[/yellow]")

    if md_path.exists():
        files_output.append(f" - {md_path.name}")
    else:
        files_output.append(f" - {md_path.name} [yellow](not found)[/yellow]")

    for line in files_output:
        console.print(line, soft_wrap=True)


def get_interactive_requirements() -> tuple[str, str]:
    """Get test requirements interactively from user."""
    console.print("\n[bold cyan]HAINDY Interactive Mode[/bold cyan]")
    console.print("Enter your test requirements. You can paste multi-line text.")
    console.print("[dim]Press Enter twice on an empty line when done:[/dim]\n")
    
    lines = []
    empty_count = 0
    
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
    
    requirements = "\n".join(lines).strip()
    
    if not requirements:
        console.print("[red]Error: No requirements provided[/red]")
        sys.exit(1)
    
    # Get URL
    url = Prompt.ask("\n[cyan]Enter the starting URL for testing[/cyan]")
    
    return requirements, url


async def test_api_connection() -> int:
    """Test OpenAI API connection."""
    console.print("\n[bold cyan]Testing OpenAI API Connection[/bold cyan]")
    
    try:
        from src.models.openai_client import OpenAIClient
        
        console.print("[cyan]Testing API key...[/cyan]")
        client = OpenAIClient()
        # Test with a simple prompt
        response = await client.call(
            messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}],
        )
        
        if "API test successful" in response["content"]:
            console.print("[green]✓ OpenAI API connection successful![/green]")
            console.print(f"[dim]Model: {response['model']}[/dim]")
            console.print(f"[dim]Usage: {response['usage']['total_tokens']} tokens[/dim]")
            return 0
        else:
            console.print("[red]✗ Unexpected API response[/red]")
            return 1
            
    except Exception as e:
        console.print(f"[red]✗ API test failed: {e}[/red]")
        console.print("\n[yellow]Please check:[/yellow]")
        console.print("1. Your OPENAI_API_KEY environment variable is set")
        console.print("2. Your API key has sufficient credits")
        console.print("3. Your account is enabled for the GPT-4.1 or GPT-5 models")
        return 1


def show_version() -> int:
    """Show version information."""
    console.print("\n[bold cyan]HAINDY - Autonomous AI Testing Agent[/bold cyan]")
    console.print("Version: [green]0.1.0[/green]")
    console.print("Python: [dim]3.10+[/dim]")
    console.print("License: [dim]MIT[/dim]")
    return 0


async def read_plan_file(file_path: Path) -> tuple[str, str]:
    """Read requirements from a plan file and extract URL."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        sys.exit(1)
    
    console.print(f"\n[cyan]Reading requirements from:[/cyan] {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            file_contents = f.read().strip()
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        sys.exit(1)
    
    # Extract URL from the document
    import re
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?\'")\]}]'
    urls = re.findall(url_pattern, file_contents)
    
    requirements_text = file_contents

    if urls:
        # Use the first URL found
        url = urls[0]
        console.print(f"[dim]Found URL in document: {url}[/dim]")
        requirements_text = requirements_text.replace(url, "", 1).strip()
    else:
        # Prompt for URL
        url = Prompt.ask("\n[cyan]Enter the application URL[/cyan]")
        requirements_text = requirements_text.strip()
    
    return requirements_text, url


async def async_main(args: Optional[list[str]] = None) -> int:
    """Async main entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Handle utility commands first
    if parsed_args.version:
        return show_version()
    
    if parsed_args.test_api:
        return await test_api_connection()
    
    # Initialize configuration
    settings = get_settings()
    
    # Override settings with command line arguments
    if parsed_args.debug:
        settings.debug_mode = True
        settings.log_level = "DEBUG"
    
    if parsed_args.verbose:
        settings.log_format = "json"
    else:
        settings.log_format = "text"
    
    if parsed_args.headless is not None:
        settings.browser_headless = parsed_args.headless
    
    # Set up logging
    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file,
    )
    
    # Initialize security components
    rate_limiter = RateLimiter()
    sanitizer = DataSanitizer()
    
    # Handle different input modes
    if parsed_args.requirements:
        # Interactive mode
        requirements, url = get_interactive_requirements()
    elif parsed_args.plan:
        # Read requirements from plan file
        requirements, url = await read_plan_file(parsed_args.plan)
    elif parsed_args.json_test_plan:
        # Load from JSON scenario file
        scenario = load_scenario(parsed_args.json_test_plan)
        requirements = scenario["requirements"]
        url = scenario["url"]
    else:
        # No input provided
        parser.print_help()
        return 1
    
    # Run test
    return await run_test(
        requirements=requirements,
        url=url,
        plan_only=parsed_args.plan_only,
        headless=settings.browser_headless,
        output_dir=parsed_args.output,
        report_format=parsed_args.format,
        timeout=parsed_args.timeout,
        max_steps=parsed_args.max_steps,
        berserk=parsed_args.berserk,
    )


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for HAINDY.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        return asyncio.run(async_main(args))
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
