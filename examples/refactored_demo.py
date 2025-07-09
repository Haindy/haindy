"""
Demo script showing the refactored architecture in action.

This demonstrates:
1. Action Agent owning full action lifecycle
2. Test Runner focusing on orchestration
3. Enhanced error reporting with bug reports
4. No separate Evaluator Agent
"""

import asyncio
from datetime import datetime
from pathlib import Path

from src.agents.action_agent_v2 import ActionAgentV2
from src.agents.test_runner_v2 import TestRunnerV2
from src.browser.driver import PlaywrightDriver
from src.core.types import (
    ActionInstruction,
    ActionType,
    TestPlan,
    TestStep
)
from src.monitoring.enhanced_reporter import EnhancedReporter
from src.monitoring.logger import setup_logging


async def create_sample_test_plan() -> TestPlan:
    """Create a sample Wikipedia search test plan."""
    steps = [
        TestStep(
            step_number=1,
            description="Click on the search box",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click on the Wikipedia search input field",
                target="Search input box",
                expected_outcome="Search box is focused and ready for input"
            )
        ),
        TestStep(
            step_number=2,
            description="Type search query",
            action_instruction=ActionInstruction(
                action_type=ActionType.TYPE,
                description="Type 'Artificial Intelligence' in search box",
                target="Search input field",
                value="Artificial Intelligence",
                expected_outcome="Search query is entered in the search box"
            )
        ),
        TestStep(
            step_number=3,
            description="Click search button",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click the search/go button",
                target="Search button or magnifying glass icon",
                expected_outcome="Search is initiated and results page loads"
            )
        )
    ]
    
    return TestPlan(
        name="Wikipedia Search Test",
        description="Test searching for 'Artificial Intelligence' on Wikipedia",
        requirements="User should be able to search Wikipedia for a topic",
        steps=steps
    )


async def run_refactored_demo():
    """Run the demo with refactored architecture."""
    print("\n🚀 HAINDY Refactored Architecture Demo")
    print("=" * 60)
    
    # Setup
    setup_logging(level="INFO")
    browser_driver = PlaywrightDriver(headless=False)
    
    try:
        # Start browser
        print("\n1️⃣ Starting browser...")
        await browser_driver.start()
        
        # Create agents with new architecture
        print("\n2️⃣ Creating refactored agents...")
        action_agent = ActionAgentV2(browser_driver=browser_driver)
        test_runner = TestRunnerV2(
            browser_driver=browser_driver,
            action_agent=action_agent
        )
        
        # Create test plan
        print("\n3️⃣ Creating test plan...")
        test_plan = await create_sample_test_plan()
        print(f"   Test: {test_plan.name}")
        print(f"   Steps: {len(test_plan.steps)}")
        
        # Execute test
        print("\n4️⃣ Executing test with enhanced tracking...")
        print("   Watch the browser window and see the enhanced logging!")
        
        enhanced_state = await test_runner.execute_test_plan(
            test_plan=test_plan,
            initial_url="https://www.wikipedia.org"
        )
        
        # Generate reports
        print("\n5️⃣ Generating enhanced reports...")
        reporter = EnhancedReporter()
        
        # Terminal summary
        reporter.print_terminal_summary(enhanced_state)
        
        # HTML report
        report_path = Path("reports") / f"refactored_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        reporter.generate_html_report(enhanced_state, report_path)
        print(f"\n📊 HTML report saved to: {report_path}")
        
        # Show key improvements
        print("\n✨ Key Improvements Demonstrated:")
        print("   1. Action Agent validates before executing")
        print("   2. Grid screenshots with highlighted cells saved")
        print("   3. Comprehensive bug reports generated")
        print("   4. AI reasoning captured at each step")
        print("   5. Test Runner focuses on orchestration only")
        
        # If there were failures, show bug report details
        if enhanced_state.bug_reports:
            print("\n🐛 Bug Reports Generated:")
            for bug in enhanced_state.bug_reports:
                print(f"\n   Step {bug.step_number}: {bug.severity.upper()}")
                print(f"   Error: {bug.error_message}")
                print(f"   Screenshots: {list(bug.screenshots.keys())}")
                print(f"   Recommendations: {len(bug.recommended_fixes)} provided")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n6️⃣ Cleaning up...")
        await browser_driver.stop()
        print("\n✅ Demo completed!")


async def compare_architectures():
    """Show the architectural differences."""
    print("\n📐 ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    print("\n🔴 OLD ARCHITECTURE:")
    print("├─ Test Runner:")
    print("│  ├─ Takes screenshots")
    print("│  ├─ Calls Action Agent for coordinates")
    print("│  ├─ Executes browser actions")
    print("│  └─ Calls Evaluator for validation")
    print("├─ Action Agent:")
    print("│  └─ Only returns grid coordinates")
    print("└─ Evaluator Agent:")
    print("   └─ Judges success after the fact")
    
    print("\n🟢 NEW ARCHITECTURE:")
    print("├─ Test Runner:")
    print("│  ├─ High-level orchestration")
    print("│  ├─ Provides context to Action Agent")
    print("│  └─ Generates bug reports")
    print("└─ Action Agent:")
    print("   ├─ Validates action feasibility")
    print("   ├─ Determines coordinates")
    print("   ├─ Executes browser action")
    print("   ├─ Captures comprehensive results")
    print("   └─ Analyzes outcome")
    
    print("\n💡 BENEFITS:")
    print("• Better error context (validation → execution flow)")
    print("• Grid screenshots with highlights for debugging")
    print("• AI reasoning captured at each decision point")
    print("• Comprehensive bug reports with recommendations")
    print("• Cleaner separation of concerns")


if __name__ == "__main__":
    # Show architecture comparison
    asyncio.run(compare_architectures())
    
    # Run the demo
    print("\n" + "="*60)
    input("Press Enter to run the refactored demo...")
    
    asyncio.run(run_refactored_demo())