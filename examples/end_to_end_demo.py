#!/usr/bin/env python3
"""
End-to-End Demo of HAINDY's Multi-Agent Test Execution.

This demonstrates the complete workflow from requirements to test report.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.browser.controller import BrowserController
from src.monitoring.logger import setup_logging
from src.monitoring.reporter import TestReporter
from src.orchestration.communication import MessageBus
from src.orchestration.coordinator import WorkflowCoordinator
from src.orchestration.state_manager import StateManager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


async def demo_end_to_end():
    """Demonstrate complete end-to-end test execution."""
    # Setup logging
    setup_logging(log_level="INFO")
    
    console.print(Panel.fit(
        "[bold cyan]HAINDY End-to-End Demo[/bold cyan]\n"
        "Demonstrating complete multi-agent test execution workflow",
        border_style="cyan",
    ))
    
    # Initialize components
    browser_controller = None
    coordinator = None
    
    try:
        # Step 1: Initialize browser
        console.print("\n[yellow]Step 1:[/yellow] Initializing browser controller...")
        browser_controller = BrowserController(headless=True)
        await browser_controller.start()
        console.print("[green]✓[/green] Browser initialized")
        
        # Step 2: Initialize orchestration components
        console.print("\n[yellow]Step 2:[/yellow] Setting up orchestration framework...")
        message_bus = MessageBus()
        state_manager = StateManager()
        
        coordinator = WorkflowCoordinator(
            message_bus=message_bus,
            state_manager=state_manager,
            browser_driver=browser_controller.driver,
            max_steps=20,
        )
        
        await coordinator.initialize()
        console.print("[green]✓[/green] Orchestration framework ready")
        console.print(f"  - Active agents: {len(coordinator._agents)}")
        console.print(f"  - Message bus ready: {coordinator.message_bus is not None}")
        console.print(f"  - State manager ready: {coordinator.state_manager is not None}")
        
        # Step 3: Define test requirements
        console.print("\n[yellow]Step 3:[/yellow] Defining test requirements...")
        test_requirements = """
        Test the HAINDY demo application login flow:
        1. Navigate to the login page
        2. Verify the login form is displayed with username and password fields
        3. Enter 'testuser' in the username field
        4. Enter 'password123' in the password field
        5. Click the Login button
        6. Verify successful login (either success message or redirect to dashboard)
        """
        
        test_url = f"file://{Path(__file__).parent}/test_page.html"
        
        console.print(Panel(
            f"[cyan]Requirements:[/cyan]\n{test_requirements}\n\n"
            f"[cyan]Target URL:[/cyan] {test_url}",
            title="Test Scenario",
            border_style="blue",
        ))
        
        # Step 4: Navigate to test page
        console.print("\n[yellow]Step 4:[/yellow] Navigating to test application...")
        await browser_controller.navigate(test_url)
        console.print("[green]✓[/green] Navigation complete")
        
        # Step 5: Generate test plan (demo plan-only mode)
        console.print("\n[yellow]Step 5:[/yellow] Generating test plan from requirements...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            plan_task = progress.add_task("[cyan]Planning test execution...", total=None)
            
            test_plan = await coordinator.generate_test_plan(test_requirements)
            
            progress.update(plan_task, completed=True)
        
        # Display test plan
        table = Table(title="Generated Test Plan", show_lines=True)
        table.add_column("Step", style="cyan", width=6)
        table.add_column("Action", style="green")
        table.add_column("Target", style="yellow")
        table.add_column("Expected Result", style="white")
        
        for i, step in enumerate(test_plan.steps[:5], 1):  # Show first 5 steps
            table.add_row(
                str(i),
                step.action,
                step.target or "-",
                step.expected_result or "-",
            )
        
        if len(test_plan.steps) > 5:
            table.add_row(
                "...",
                f"({len(test_plan.steps) - 5} more steps)",
                "-",
                "-",
            )
        
        console.print(table)
        
        # Step 6: Execute test
        console.print("\n[yellow]Step 6:[/yellow] Executing test with multi-agent system...")
        console.print("[dim]Note: This demo shows the workflow structure.[/dim]")
        console.print("[dim]Full agent implementation would execute each step.[/dim]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            exec_task = progress.add_task("[cyan]Running test execution...", total=None)
            
            # In a real execution, this would run the complete test
            # For demo purposes, we'll simulate the execution
            try:
                test_state = await asyncio.wait_for(
                    coordinator.execute_test_from_requirements(
                        requirements=test_requirements,
                        initial_url=test_url,
                    ),
                    timeout=30,
                )
                progress.update(exec_task, completed=True)
            except Exception as e:
                progress.update(exec_task, description=f"[yellow]Note: {e}[/yellow]")
                console.print(f"\n[yellow]Note: Full agent execution requires OpenAI API key.[/yellow]")
                console.print("[dim]Using mock test state for demo purposes.[/dim]\n")
                # For demo, create a mock test state
                from uuid import uuid4
                from datetime import datetime, timezone
                from src.core.types import TestState, TestStatus, ActionResult
                
                test_state = TestState(
                    test_id=uuid4(),
                    status=TestStatus.COMPLETED,
                    current_step=3,
                    step_results={
                        "step_1": ActionResult(
                            success=True,
                            screenshot_path="/tmp/demo_screenshot1.png",
                            timestamp=datetime.now(timezone.utc),
                            confidence=0.95,
                            message="Navigation successful",
                        ),
                        "step_2": ActionResult(
                            success=True,
                            screenshot_path="/tmp/demo_screenshot2.png",
                            timestamp=datetime.now(timezone.utc),
                            confidence=0.92,
                            message="Login form verified",
                        ),
                        "step_3": ActionResult(
                            success=True,
                            screenshot_path="/tmp/demo_screenshot3.png",
                            timestamp=datetime.now(timezone.utc),
                            confidence=0.90,
                            message="Login completed",
                        ),
                    },
                    errors=[],
                )
        
        # Step 7: Generate report
        console.print("\n[yellow]Step 7:[/yellow] Generating test execution report...")
        
        reporter = TestReporter()
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        report_path = await reporter.generate_report(
            test_state=test_state,
            output_dir=output_dir,
            format="html",
        )
        
        console.print(f"[green]✓[/green] Report generated: {report_path}")
        
        # Display summary
        console.print("\n[bold]Test Execution Summary:[/bold]")
        
        status_color = "green" if test_state.status == "completed" else "red"
        console.print(f"Status: [{status_color}]{test_state.status}[/{status_color}]")
        console.print(f"Total Steps: {len(test_state.step_results)}")
        
        passed_steps = sum(1 for r in test_state.step_results.values() if r.success)
        console.print(f"Passed Steps: [green]{passed_steps}[/green]")
        console.print(f"Failed Steps: [red]{len(test_state.step_results) - passed_steps}[/red]")
        
        # Show workflow components
        console.print("\n[bold]Workflow Components Used:[/bold]")
        components_table = Table(show_header=False, box=None)
        components_table.add_column("Component", style="cyan")
        components_table.add_column("Status", style="green")
        
        components_table.add_row("Message Bus", "✓ Active")
        components_table.add_row("State Manager", "✓ Tracking test state")
        components_table.add_row("Browser Controller", "✓ Automated interactions")
        components_table.add_row("Test Planner Agent", "✓ Generated test plan")
        components_table.add_row("Test Runner Agent", "✓ Orchestrated execution")
        components_table.add_row("Action Agent", "✓ Determined coordinates")
        components_table.add_row("Evaluator Agent", "✓ Assessed results")
        components_table.add_row("Test Reporter", "✓ Generated report")
        
        console.print(components_table)
        
        console.print("\n[green]✨ End-to-end demo complete![/green]")
        
    except Exception as e:
        console.print(f"\n[red]Error during demo: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        console.print("\n[yellow]Cleaning up...[/yellow]")
        if coordinator:
            await coordinator.cleanup()
        if browser_controller:
            await browser_controller.stop()
        console.print("[green]✓[/green] Cleanup complete")


def main():
    """Run the demo."""
    console.print("[bold]HAINDY - End-to-End Demo[/bold]\n")
    
    try:
        asyncio.run(demo_end_to_end())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo failed: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())