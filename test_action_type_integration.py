#!/usr/bin/env python
"""Test script to verify Test Runner creates proper ActionInstructions."""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

from src.agents.test_runner import TestRunner
from src.core.types import TestCase, TestStep, TestPlan, TestCasePriority


async def test_action_type_integration():
    """Test that Test Runner properly interprets steps and creates ActionInstructions."""
    
    # Create a test step
    test_step = TestStep(
        step_number=1,
        description="Type and submit search",
        action="Type 'machine learning' in the search box and press Enter",
        expected_result="Search results for machine learning are displayed"
    )
    
    # Create test case and plan
    test_case = TestCase(
        test_id="TC001",
        name="Search Test",
        description="Test search functionality",
        priority=TestCasePriority.HIGH,
        steps=[test_step]
    )
    
    test_plan = TestPlan(
        name="Search Feature Test",
        description="Test the search feature",
        requirements_source="User story #123",
        test_cases=[test_case]
    )
    
    # Create test runner
    runner = TestRunner()
    
    # Mock the OpenAI call to return a proper response
    async def mock_ai_response(*args, **kwargs):
        # Check if this is the interpret step call
        messages = kwargs.get("messages", [])
        if messages and "break it down into specific browser actions" in messages[0]["content"]:
            return {
                "content": json.dumps({
                    "actions": [
                        {
                            "type": "click",
                            "target": "search box",
                            "description": "Click on the search box to focus it",
                            "critical": True
                        },
                        {
                            "type": "type",
                            "target": "search box",
                            "value": "machine learning",
                            "description": "Type the search query",
                            "critical": True
                        },
                        {
                            "type": "key_press",
                            "target": "search box",
                            "value": "Enter",
                            "description": "Press Enter to submit search",
                            "critical": True
                        }
                    ]
                })
            }
        return {"content": "{}"}
    
    runner.call_openai = AsyncMock(side_effect=mock_ai_response)
    
    # Test the _interpret_step method
    runner._current_test_case = test_case
    actions = await runner._interpret_step(test_step)
    
    print("\nTest Runner interpreted the step into these actions:")
    for i, action in enumerate(actions):
        print(f"\n{i+1}. Action Type: {action['type']}")
        print(f"   Target: {action.get('target', 'N/A')}")
        print(f"   Value: {action.get('value', 'N/A')}")
        print(f"   Description: {action.get('description', 'N/A')}")
    
    # Verify we got the right action types
    assert len(actions) == 3
    assert actions[0]["type"] == "click"
    assert actions[1]["type"] == "type"
    assert actions[1]["value"] == "machine learning"
    assert actions[2]["type"] == "key_press"
    assert actions[2]["value"] == "Enter"
    
    print("\n✅ Test Runner successfully interpreted the step with proper action types!")
    
    # Now test that _execute_action creates proper ActionInstruction
    # We need to set up more context for this
    runner._current_test_plan = test_plan
    runner._current_test_case = test_case
    runner.action_agent = AsyncMock()
    runner.browser_driver = AsyncMock()
    runner.browser_driver.screenshot = AsyncMock(return_value=b"fake_screenshot")
    
    # Mock the action agent to capture what it receives
    captured_test_step = None
    async def capture_execute_action(**kwargs):
        nonlocal captured_test_step
        captured_test_step = kwargs.get("test_step")
        # Return a mock result
        from src.core.enhanced_types import EnhancedActionResult, ValidationResult, ExecutionResult, CoordinateResult
        return EnhancedActionResult(
            test_step_id=kwargs["test_step"].step_id,
            test_step=kwargs["test_step"],
            test_context=kwargs["test_context"],
            overall_success=True,
            validation=ValidationResult(valid=True, confidence=0.9, reasoning="OK"),
            execution=ExecutionResult(success=True, execution_time_ms=100)
        )
    
    runner.action_agent.execute_action = capture_execute_action
    
    # Execute the first action (click)
    result = await runner._execute_action(actions[0], test_step)
    
    print(f"\n✅ Action executed successfully: {result.get('success', False)}")
    
    # Check that ActionInstruction was created properly
    assert captured_test_step is not None
    assert captured_test_step.action_instruction is not None
    assert captured_test_step.action_instruction.action_type.value == "click"
    assert captured_test_step.action_instruction.target == "search box"
    
    print(f"✅ ActionInstruction created with type: {captured_test_step.action_instruction.action_type.value}")
    print(f"✅ ActionInstruction target: {captured_test_step.action_instruction.target}")
    
    print("\n🎉 All tests passed! The Test Runner properly creates ActionInstructions from AI responses.")


if __name__ == "__main__":
    asyncio.run(test_action_type_integration())