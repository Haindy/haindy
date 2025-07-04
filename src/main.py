"""
HAINDY - Autonomous AI Testing Agent
Main entry point for the application.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.browser.controller import BrowserController
from src.config.settings import get_settings
from src.monitoring.logger import get_logger, setup_logging
from src.monitoring.reporter import TestReporter
from src.orchestration.communication import MessageBus
from src.orchestration.coordinator import WorkflowCoordinator
from src.orchestration.state_manager import StateManager
from src.security.rate_limiter import RateLimiter
from src.security.sanitizer import DataSanitizer

console = Console()
logger = get_logger("main")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="HAINDY - Autonomous AI Testing Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a test from requirements text
  python -m src.main --requirements "Test the login flow"
  
  # Run a test from a scenario file
  python -m src.main --scenario test_scenarios/login_test.json
  
  # Run with custom configuration
  python -m src.main --scenario login_test.json --headless --debug
  
  # Generate test plan without execution
  python -m src.main --requirements "Test checkout" --plan-only
        """,
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-r", "--requirements",
        help="Test requirements as text",
    )
    input_group.add_argument(
        "-s", "--scenario",
        type=Path,
        help="Path to test scenario JSON file",
    )
    
    # Execution options
    parser.add_argument(
        "-u", "--url",
        help="Initial URL to navigate to (required with --requirements)",
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
        default=300,
        help="Test execution timeout in seconds (default: 300)",
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
    timeout: int = 300,
    max_steps: int = 50,
) -> int:
    """
    Run a test with the given requirements.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Initialize components
    browser_controller = None
    coordinator = None
    
    try:
        # Show startup banner
        console.print(Panel.fit(
            "[bold cyan]HAINDY - Autonomous AI Testing Agent[/bold cyan]\n"
            "Orchestrating multi-agent test execution",
            border_style="cyan",
        ))
        
        # Initialize core components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            init_task = progress.add_task("[cyan]Initializing components...", total=None)
            
            # Create browser controller
            browser_controller = BrowserController(headless=headless)
            await browser_controller.start()
            
            # Create orchestration components
            message_bus = MessageBus()
            state_manager = StateManager()
            
            # Create coordinator
            coordinator = WorkflowCoordinator(
                message_bus=message_bus,
                state_manager=state_manager,
                browser_driver=browser_controller.driver,
                max_steps=max_steps,
            )
            
            # Initialize agents
            await coordinator.initialize()
            
            progress.update(init_task, completed=True)
        
        # Navigate to initial URL
        console.print(f"\n[cyan]Navigating to:[/cyan] {url}")
        await browser_controller.navigate(url)
        
        if plan_only:
            # Generate test plan only
            console.print("\n[yellow]Generating test plan...[/yellow]")
            test_plan = await coordinator.generate_test_plan(requirements)
            
            # Display test plan
            table = Table(title="Test Plan", show_lines=True)
            table.add_column("Step", style="cyan", width=6)
            table.add_column("Action", style="green")
            table.add_column("Target", style="yellow")
            table.add_column("Expected Result", style="white")
            
            for i, step in enumerate(test_plan.steps, 1):
                table.add_row(
                    str(i),
                    step.action_instruction.action_type,
                    step.action_instruction.target or "-",
                    step.description,
                )
            
            console.print(table)
            return 0
        
        # Execute test
        console.print(f"\n[green]Executing test:[/green] {requirements[:100]}...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            exec_task = progress.add_task("[cyan]Running test...", total=None)
            
            # Execute test with timeout
            test_state = await asyncio.wait_for(
                coordinator.execute_test_from_requirements(
                    requirements=requirements,
                    initial_url=url,
                ),
                timeout=timeout,
            )
            
            progress.update(exec_task, completed=True)
        
        # Display results summary
        console.print("\n[bold]Test Execution Summary:[/bold]")
        
        status_color = "green" if test_state.status == "completed" else "red"
        console.print(f"Status: [{status_color}]{test_state.status}[/{status_color}]")
        console.print(f"Total Steps: {len(test_state.step_results)}")
        
        passed_steps = sum(1 for r in test_state.step_results.values() if r.success)
        console.print(f"Passed Steps: [green]{passed_steps}[/green]")
        console.print(f"Failed Steps: [red]{len(test_state.step_results) - passed_steps}[/red]")
        
        # Generate report
        if output_dir is None:
            output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        console.print(f"\n[cyan]Generating {report_format} report...[/cyan]")
        
        reporter = TestReporter()
        report_path = await reporter.generate_report(
            test_state=test_state,
            output_dir=output_dir,
            format=report_format,
        )
        
        console.print(f"[green]Report saved to:[/green] {report_path}")
        
        # Return appropriate exit code
        return 0 if test_state.status == "completed" else 1
        
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


async def async_main(args: Optional[list[str]] = None) -> int:
    """Async main entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Initialize configuration
    settings = get_settings()
    
    # Override settings with command line arguments
    if parsed_args.debug:
        settings.debug_mode = True
        settings.log_level = "DEBUG"
    
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
    
    # Determine test requirements and URL
    if parsed_args.requirements:
        requirements = parsed_args.requirements
        url = parsed_args.url
        if not url:
            console.print("[red]Error: --url is required when using --requirements[/red]")
            return 1
    else:
        # Load from scenario file
        scenario = load_scenario(parsed_args.scenario)
        requirements = scenario["requirements"]
        url = scenario["url"]
    
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