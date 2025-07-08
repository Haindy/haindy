"""
Example of how to properly mock call_openai for ActionAgent in tests.
"""

import json
from unittest.mock import AsyncMock, Mock
import pytest
from src.agents.action_agent import ActionAgent
from src.core.types import ActionInstruction, ActionType


class TestActionAgentMocking:
    """Examples of different mocking strategies for ActionAgent."""
    
    @pytest.mark.asyncio
    async def test_with_real_action_agent_proper_mock(self):
        """Test using a real ActionAgent instance with properly mocked call_openai."""
        # Create a real ActionAgent instance
        mock_browser = Mock()
        mock_browser.get_viewport_size = AsyncMock(return_value=(1920, 1080))
        
        agent = ActionAgent(browser_driver=mock_browser)
        
        # Method 1: Mock call_openai to return a proper response dict
        coordinate_response = {
            "content": json.dumps({
                "cell": "M23",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.9,
                "reasoning": "Button found in center of screen"
            })
        }
        
        # Use AsyncMock but ensure it returns a dict, not a coroutine
        agent.call_openai = AsyncMock(return_value=coordinate_response)
        
        # Now when _parse_coordinate_response is called, it will get a dict
        # not a coroutine, avoiding the confidence 0.1 fallback
        
    @pytest.mark.asyncio  
    async def test_with_side_effect_for_multiple_calls(self):
        """Test using side_effect for multiple call_openai invocations."""
        mock_browser = Mock()
        mock_browser.get_viewport_size = AsyncMock(return_value=(1920, 1080))
        
        agent = ActionAgent(browser_driver=mock_browser)
        
        # Define responses for validation, coordinate, and analysis phases
        validation_response = {
            "content": json.dumps({
                "valid": True,
                "confidence": 0.95,
                "reasoning": "Element is visible"
            })
        }
        
        coordinate_response = {
            "content": json.dumps({
                "cell": "M23",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.9,
                "reasoning": "Button found"
            })
        }
        
        analysis_response = {
            "content": json.dumps({
                "success": True,
                "confidence": 0.85,
                "actual_outcome": "Action completed"
            })
        }
        
        # Use side_effect to return different responses for each call
        agent.call_openai = AsyncMock(
            side_effect=[validation_response, coordinate_response, analysis_response]
        )
        
    @pytest.mark.asyncio
    async def test_with_callable_side_effect(self):
        """Test using a callable for more dynamic mocking."""
        mock_browser = Mock()
        mock_browser.get_viewport_size = AsyncMock(return_value=(1920, 1080))
        
        agent = ActionAgent(browser_driver=mock_browser)
        
        # Track calls for debugging
        call_count = 0
        
        async def mock_call_openai(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Return different responses based on call count
            if call_count == 1:  # Validation
                return {
                    "content": json.dumps({
                        "valid": True,
                        "confidence": 0.95,
                        "reasoning": "Valid action"
                    })
                }
            elif call_count == 2:  # Coordinate determination
                return {
                    "content": json.dumps({
                        "cell": "M23",
                        "offset_x": 0.5,
                        "offset_y": 0.5,
                        "confidence": 0.9,
                        "reasoning": "Located element"
                    })
                }
            else:  # Analysis
                return {
                    "content": json.dumps({
                        "success": True,
                        "confidence": 0.85,
                        "actual_outcome": "Success"
                    })
                }
        
        # Assign the callable directly
        agent.call_openai = mock_call_openai
        
    def test_debugging_mock_issues(self):
        """Example of how to debug AsyncMock issues."""
        # If you're getting confidence 0.1, add logging to understand why
        
        # 1. Check if the mock is returning a coroutine
        mock = AsyncMock(return_value={"content": "test"})
        result = mock()  # This returns a coroutine!
        print(f"Is coroutine: {asyncio.iscoroutine(result)}")  # True
        
        # 2. To get the actual value, you need to await it
        # But _parse_coordinate_response expects a dict, not a coroutine
        
        # 3. The fix is to ensure the mock is set up correctly
        # so that when awaited in the calling code, it returns a dict