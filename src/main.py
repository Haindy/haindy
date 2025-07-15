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
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from src.browser.controller import BrowserController
from src.config.settings import get_settings
from src.monitoring.logger import get_logger, setup_logging
from src.monitoring.reporter import TestReporter
from src.monitoring.debug_logger import initialize_debug_logger
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
        default=1200,
        help="Test execution timeout in seconds (default: 1200)",
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
    timeout: int = 1200,
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
        
        status_color = "green" if getattr(test_state, 'status', getattr(test_state, 'test_status', 'unknown')) == "completed" else "red"
        status_text = getattr(test_state, 'status', getattr(test_state, 'test_status', 'unknown'))
        console.print(f"Status: [{status_color}]{status_text}[/{status_color}]")
        
        # Display test execution metrics from test report
        if test_state.test_report:
            report = test_state.test_report
            
            # Calculate totals from test cases
            total_test_cases = len(report.test_cases)
            completed_test_cases = len([tc for tc in report.test_cases if tc.status == "completed"])
            
            # Calculate step totals
            total_steps = sum(tc.steps_total for tc in report.test_cases)
            completed_steps = sum(tc.steps_completed for tc in report.test_cases)
            failed_steps = sum(tc.steps_failed for tc in report.test_cases)
            skipped_steps = total_steps - completed_steps - failed_steps
            
            console.print(f"Test Cases: [cyan]{completed_test_cases}/{total_test_cases}[/cyan]")
            console.print(f"Total Steps: {total_steps}")
            console.print(f"Completed Steps: [green]{completed_steps}[/green]")
            console.print(f"Failed Steps: [red]{failed_steps}[/red]")
            console.print(f"Skipped Steps: [yellow]{skipped_steps}[/yellow]")
            
            # Show summary if available
            if report.summary:
                console.print(f"Success Rate: [cyan]{report.summary.success_rate:.1f}%[/cyan]")
        else:
            console.print("[yellow]No test report available[/yellow]")
        
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
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Testing API key...", total=None)
            
            client = OpenAIClient()
            # Test with a simple prompt
            response = await client.call(
                messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}],
            )
            
            progress.update(task, completed=True)
        
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
        console.print("3. You have access to the gpt-4o-mini model")
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
    
    if urls:
        # Use the first URL found
        url = urls[0]
        console.print(f"[dim]Found URL in document: {url}[/dim]")
    else:
        # Prompt for URL
        url = Prompt.ask("\n[cyan]Enter the application URL[/cyan]")
    
    return file_contents, url


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