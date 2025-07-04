#!/usr/bin/env python3
"""
Demonstration of the Test Runner Agent.

This script shows how the Test Runner Agent orchestrates test execution
by coordinating between Test Planner, Action, and Evaluator agents.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.test_runner import (
    ExecutionMode,
    TestRunnerAgent,
    ActionResult,
    TestPlan,
    TestState,
    TestStep,
)
from src.core.types import EvaluationResult
from src.monitoring.logger import setup_logging


class MockBrowserDriver:
    """Mock browser driver for demo."""
    
    def __init__(self):
        self.current_page = "home"
        self.logged_in = False
    
    async def navigate(self, url: str) -> None:
        print(f"   Browser: Navigating to {url}")
        if "login" in url:
            self.current_page = "login"
        await asyncio.sleep(0.5)
    
    async def wait_for_load(self) -> None:
        print("   Browser: Waiting for page load...")
        await asyncio.sleep(0.3)
    
    async def wait_for_idle(self) -> None:
        await asyncio.sleep(0.1)
    
    async def take_screenshot(self) -> bytes:
        return f"screenshot_{self.current_page}".encode()
    
    async def click(self, x: float, y: float) -> None:
        print(f"   Browser: Clicking at ({x:.0f}, {y:.0f})")
        if self.current_page == "login" and y > 500:
            self.logged_in = True
            self.current_page = "dashboard"
        await asyncio.sleep(0.2)
    
    async def type_text(self, text: str) -> None:
        print(f"   Browser: Typing '{text}'")
        await asyncio.sleep(0.2)


class MockActionAgent:
    """Mock action agent for demo."""
    
    async def determine_action(self, screenshot: bytes, instruction: str) -> ActionResult:
        print(f"   Action Agent: Analyzing - '{instruction}'")
        
        # Simulate different actions based on instruction
        if "username" in instruction.lower():
            return ActionResult(
                action_type="type",
                grid_cell="M15",
                offset_x=0.5,
                offset_y=0.3,
                confidence=0.95,
                requires_refinement=False
            )
        elif "password" in instruction.lower():
            return ActionResult(
                action_type="type",
                grid_cell="M20",
                offset_x=0.5,
                offset_y=0.4,
                confidence=0.93,
                requires_refinement=False
            )
        else:
            return ActionResult(
                action_type="click",
                grid_cell="M25",
                offset_x=0.5,
                offset_y=0.6,
                confidence=0.92,
                requires_refinement=False
            )


class MockEvaluatorAgent:
    """Mock evaluator agent for demo."""
    
    def __init__(self):
        self.step_count = 0
    
    async def evaluate_result(
        self, screenshot: bytes, expected_outcome: str, step_id=None
    ) -> EvaluationResult:
        self.step_count += 1
        
        print(f"   Evaluator Agent: Checking - '{expected_outcome}'")
        
        # Simulate evaluation results
        if "dashboard" in expected_outcome.lower() and b"dashboard" in screenshot:
            success = True
            actual = "Dashboard loaded successfully with user info"
        elif "error" in expected_outcome.lower():
            success = False
            actual = "Error message displayed"
        else:
            success = True
            actual = f"Action completed as expected"
        
        return EvaluationResult(
            step_id=step_id or uuid4(),
            success=success,
            confidence=0.90 + (0.05 if success else 0),
            expected_outcome=expected_outcome,
            actual_outcome=actual,
            deviations=[] if success else ["Unexpected result"],
            suggestions=[] if success else ["Check element visibility"]
        )


async def demonstrate_test_runner():
    """Run Test Runner Agent demonstration."""
    print("HAINDY Test Runner Agent Demonstration")
    print("=" * 50)
    
    # Create mock components
    browser = MockBrowserDriver()
    action_agent = MockActionAgent()
    evaluator_agent = MockEvaluatorAgent()
    
    # Initialize Test Runner
    print("\n1. Initializing Test Runner Agent...")
    runner = TestRunnerAgent(
        browser_driver=browser,
        action_agent=action_agent,
        evaluator_agent=evaluator_agent
    )
    
    # Mock AI responses
    async def mock_ai_analysis(messages, **kwargs):
        return {
            "content": json.dumps({
                "assessment": "Test execution proceeding smoothly",
                "concerns": [],
                "recommendations": ["Continue with remaining steps"]
            })
        }
    
    runner.call_ai = mock_ai_analysis
    
    print("   ✓ Test Runner initialized with browser and agent connections")
    
    # Create a test plan
    print("\n2. Creating Test Plan...")
    test_plan = TestPlan(
        test_id=uuid4(),
        name="User Login Flow",
        description="Test the complete user login process",
        prerequisites=["User account exists", "Browser at home page"],
        steps=[
            TestStep(
                id=uuid4(),
                step_number=1,
                action="Navigate to login page",
                expected_result="Login page displayed with form fields",
                depends_on=[],
                is_critical=True
            ),
            TestStep(
                id=uuid4(),
                step_number=2,
                action="Enter username 'testuser'",
                expected_result="Username entered in field",
                depends_on=[1],
                is_critical=True
            ),
            TestStep(
                id=uuid4(),
                step_number=3,
                action="Enter password",
                expected_result="Password entered (masked)",
                depends_on=[1],
                is_critical=True
            ),
            TestStep(
                id=uuid4(),
                step_number=4,
                action="Click login button",
                expected_result="User redirected to dashboard",
                depends_on=[2, 3],
                is_critical=True
            ),
            TestStep(
                id=uuid4(),
                step_number=5,
                action="Verify user info displayed",
                expected_result="Dashboard shows logged-in user info",
                depends_on=[4],
                is_critical=False
            ),
        ],
        success_criteria=[
            "User can log in with valid credentials",
            "Dashboard displays after login",
            "User info visible on dashboard"
        ],
        edge_cases=["Invalid credentials", "Network timeout"]
    )
    
    print(f"   ✓ Test plan created: '{test_plan.name}'")
    print(f"   ✓ Total steps: {len(test_plan.steps)}")
    
    # Execute test plan
    print("\n3. Executing Test Plan...")
    print("   Starting test execution with visual AI mode...\n")
    
    # Execute
    final_state = await runner.execute_test_plan(
        test_plan,
        initial_url="https://example.com/login"
    )
    
    # Show results
    print("\n4. Test Execution Results:")
    print(f"   Status: {final_state.test_status.upper()}")
    print(f"   Completed Steps: {len(final_state.completed_steps)}/{len(test_plan.steps)}")
    print(f"   Final Context: {final_state.context.get('test_plan_name')}")
    
    # Show execution history
    print("\n5. Execution History:")
    for i, result in enumerate(runner._execution_history, 1):
        print(f"\n   Step {i}: {result.step.action}")
        print(f"   - Success: {'✓' if result.success else '✗'}")
        print(f"   - Mode: {result.execution_mode}")
        print(f"   - Result: {result.actual_result}")
        if result.action_taken:
            print(f"   - Action: {result.action_taken.action_type} at {result.action_taken.grid_cell}")
            print(f"   - Confidence: {result.action_taken.confidence:.0%}")
    
    # Show scripted actions recorded
    print(f"\n6. Scripted Actions Recorded: {len(runner._scripted_actions)}")
    for key, action in runner._scripted_actions.items():
        print(f"   - {key}: {action['action_type']} at ({action['x']:.0f}, {action['y']:.0f})")
    
    # Demonstrate execution modes
    print("\n7. Execution Mode Demo:")
    print("   Re-running test with hybrid mode (scripted + visual fallback)...")
    
    # Reset browser state
    browser.current_page = "home"
    browser.logged_in = False
    
    # Execute again - this time scripted actions will be used
    final_state2 = await runner.execute_test_plan(test_plan, initial_url="https://example.com/login")
    
    print(f"\n   Second run status: {final_state2.test_status.upper()}")
    print("   Scripted actions were used where available!")
    
    # Summary
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print("- Test Runner successfully orchestrated multi-agent test execution")
    print("- Coordinated between Action and Evaluator agents")
    print("- Managed test state and execution flow")
    print("- Recorded actions for future scripted execution")
    print("- Demonstrated visual, scripted, and hybrid execution modes")
    
    print("\nKey Features Demonstrated:")
    print("✓ Test plan execution with dependency management")
    print("✓ Multi-agent coordination (Action + Evaluator)")
    print("✓ State tracking and progress monitoring")
    print("✓ Action recording for scripted playback")
    print("✓ AI-powered progress analysis")
    print("✓ Flexible execution modes (visual/scripted/hybrid)")
    
    print("\nDemo complete!")


async def demonstrate_conditional_execution():
    """Demonstrate conditional execution and branching."""
    print("\n\nConditional Execution Demo")
    print("=" * 50)
    
    # Create components
    browser = MockBrowserDriver()
    action_agent = MockActionAgent()
    evaluator_agent = MockEvaluatorAgent()
    
    runner = TestRunnerAgent(
        browser_driver=browser,
        action_agent=action_agent,
        evaluator_agent=evaluator_agent
    )
    
    # Mock AI for decision making
    decision_count = 0
    
    async def mock_ai_decision(messages, **kwargs):
        nonlocal decision_count
        decision_count += 1
        
        if decision_count == 1:
            return {
                "content": json.dumps({
                    "assessment": "Login failed, need to retry",
                    "concerns": ["Authentication error detected"],
                    "recommendations": ["Retry with different credentials"]
                })
            }
        else:
            return {
                "content": "Proceed with the current step as planned"
            }
    
    runner.call_ai = mock_ai_decision
    
    # Create test plan with conditional steps
    test_plan = TestPlan(
        test_id=uuid4(),
        name="Login with Retry",
        description="Login test with error handling",
        prerequisites=[],
        steps=[
            TestStep(
                id=uuid4(),
                step_number=1,
                action="Attempt login",
                expected_result="Login successful or error displayed",
                depends_on=[],
                is_critical=False  # Non-critical to allow retry
            ),
            TestStep(
                id=uuid4(),
                step_number=2,
                action="Check for errors",
                expected_result="No errors present",
                depends_on=[1],
                is_critical=False
            ),
            TestStep(
                id=uuid4(),
                step_number=3,
                action="Retry login if needed",
                expected_result="Login successful",
                depends_on=[2],
                is_critical=True
            ),
        ],
        success_criteria=["User logged in"],
        edge_cases=[]
    )
    
    # Simulate failure on first attempt
    evaluator_agent.evaluate_result = AsyncMock(side_effect=[
        EvaluationResult(
            step_id=uuid4(),
            success=False,  # First attempt fails
            confidence=0.85,
            expected_outcome="Login successful",
            actual_outcome="Login failed - invalid credentials",
            deviations=["Error message displayed"],
            suggestions=["Check credentials"]
        ),
        EvaluationResult(
            step_id=uuid4(),
            success=True,  # Error check confirms error
            confidence=0.95,
            expected_outcome="No errors present",
            actual_outcome="Error detected: Invalid credentials",
            deviations=[],
            suggestions=[]
        ),
        EvaluationResult(
            step_id=uuid4(),
            success=True,  # Retry succeeds
            confidence=0.90,
            expected_outcome="Login successful",
            actual_outcome="Login successful after retry",
            deviations=[],
            suggestions=[]
        ),
    ])
    
    print("\n1. Executing test with conditional retry logic...")
    
    # Get next action recommendation
    test_state = TestState(
        test_id=test_plan.test_id,
        current_step=0,
        completed_steps=[],
        remaining_steps=[0, 1, 2],
        test_status="in_progress"
    )
    
    recommendation = await runner.get_next_action(test_plan, test_state)
    print(f"\n2. AI Recommendation: {recommendation}")
    
    # Execute with retry
    final_state = await runner.execute_test_plan(test_plan)
    
    print(f"\n3. Final Status: {final_state.test_status.upper()}")
    print("   The test handled the error and retried successfully!")


async def main():
    """Run all demonstrations."""
    # Setup logging
    setup_logging()
    
    print("\nNote: This demo uses mocked components for demonstration purposes.")
    print("In production, real browser driver and AI agents would be used.\n")
    
    try:
        # Main demo
        await demonstrate_test_runner()
        
        # Conditional execution demo
        await demonstrate_conditional_execution()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())