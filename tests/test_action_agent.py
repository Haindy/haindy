"""
Tests for the Action Agent with complete action execution lifecycle.

This test file covers the refactored Action Agent that owns the complete
action execution lifecycle including validation, coordinate determination,
execution, and result analysis.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from PIL import Image
from io import BytesIO

from src.agents.action_agent import ActionAgent
from src.browser.driver import BrowserDriver
from src.core.types import (
    ActionInstruction, ActionType, GridCoordinate, TestStep,
    ScrollDirection, VisibilityStatus, TestState
)
from src.core.enhanced_types import (
    EnhancedActionResult, ValidationResult, CoordinateResult,
    ExecutionResult, BrowserState, AIAnalysis
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.grid_size = 60
    settings.grid_confidence_threshold = 0.8
    settings.grid_refinement_enabled = True
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.openai_temperature = 1.0  # o4-mini only supports default temperature
    settings.openai_max_retries = 3
    return settings


@pytest.fixture
def mock_browser_driver():
    """Mock browser driver for testing."""
    driver = AsyncMock(spec=BrowserDriver)
    driver.capture_screenshot.return_value = b"screenshot_data"
    driver.get_url.return_value = "https://example.com"
    driver.get_title.return_value = "Example Page"
    driver.get_viewport_size.return_value = (1920, 1080)
    return driver


@pytest.fixture
def action_agent(mock_settings, mock_browser_driver):
    """Create an ActionAgent instance for testing."""
    with patch("src.agents.action_agent.get_settings", return_value=mock_settings):
        agent = ActionAgent(browser_driver=mock_browser_driver)
        # Mock the OpenAI client
        agent._client = AsyncMock()
        return agent


@pytest.fixture
def sample_screenshot():
    """Create a sample screenshot for testing."""
    image = Image.new('RGB', (1920, 1080), color='white')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def test_step():
    """Create a sample test step."""
    return TestStep(
        step_number=1,
        action="Click the login button",
        expected_result="Login form is displayed",
        action_instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click the login button",
            target="Login button",
            expected_outcome="Login form is displayed"
        )
    )


@pytest.fixture
def test_context():
    """Create test context."""
    return {
        "test_plan_id": str(uuid4()),
        "test_case_id": str(uuid4()),
        "test_case_name": "Login Test",
        "previous_steps": []
    }


class TestActionAgentCore:
    """Test core functionality of the Action Agent."""
    
    @pytest.mark.asyncio
    async def test_execute_action_click_workflow(
        self, action_agent, test_step, test_context, sample_screenshot
    ):
        """Test execute_action with click workflow."""
        # Mock validation response
        validation_response = {
            "content": """{
                "valid": true,
                "confidence": 0.95,
                "reasoning": "Login button is visible on the page"
            }"""
        }
        
        # Mock coordinate response
        coordinate_response = {
            "content": """{
                "cell": "M23",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.9,
                "reasoning": "Login button found in cell M23"
            }"""
        }
        
        # Mock analysis response
        analysis_response = {
            "content": """{
                "success": true,
                "confidence": 0.95,
                "actual_outcome": "Login form appeared after clicking the button",
                "matches_expected": true,
                "ui_changes": ["Login form is now visible", "Background dimmed"],
                "recommendations": [],
                "anomalies": []
            }"""
        }
        
        # Set up mock responses in order
        action_agent.call_openai_with_debug = AsyncMock(side_effect=[
            validation_response,
            coordinate_response,
            analysis_response
        ])
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=test_step,
            test_context=test_context,
            screenshot=sample_screenshot
        )
        
        # Verify result
        assert isinstance(result, EnhancedActionResult)
        assert result.overall_success is True
        assert result.validation.valid is True
        assert result.validation.confidence == 0.95
        assert result.coordinate.grid_cell == "M23"
        assert result.ai_analysis.success is True
        assert result.failure_phase is None
        
        # Verify browser interactions
        action_agent.browser_driver.click.assert_called_once()
        action_agent.browser_driver.capture_screenshot.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_action_navigate_workflow(
        self, action_agent, test_context
    ):
        """Test navigation workflow."""
        nav_step = TestStep(
            step_number=1,
            action="Navigate to login page",
            expected_result="Login page is loaded",
            action_instruction=ActionInstruction(
                action_type=ActionType.NAVIGATE,
                description="Navigate to login page",
                target="Login page",
                value="https://example.com/login",
                expected_outcome="Login page is loaded"
            )
        )
        
        # Mock analysis response
        analysis_response = {
            "content": """{
                "success": true,
                "confidence": 1.0,
                "actual_outcome": "Successfully navigated to login page",
                "matches_expected": true,
                "ui_changes": ["Login page loaded"],
                "recommendations": [],
                "anomalies": []
            }"""
        }
        
        action_agent.call_openai_with_debug = AsyncMock(return_value=analysis_response)
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=nav_step,
            test_context=test_context
        )
        
        # Verify result
        assert result.overall_success is True
        assert result.validation.valid is True  # Navigation doesn't need validation
        assert result.ai_analysis.success is True
        
        # Verify navigation was called
        action_agent.browser_driver.navigate_to.assert_called_once_with("https://example.com/login")
    
    @pytest.mark.asyncio
    async def test_execute_action_type_workflow(
        self, action_agent, test_context, sample_screenshot
    ):
        """Test type/text input workflow."""
        type_step = TestStep(
            step_number=1,
            action="Enter username",
            expected_result="Username is entered in the field",
            action_instruction=ActionInstruction(
                action_type=ActionType.TYPE,
                description="Enter username",
                target="Username field",
                value="testuser@example.com",
                expected_outcome="Username is entered"
            )
        )
        
        # Mock responses
        validation_response = {
            "content": """{
                "valid": true,
                "confidence": 0.9,
                "reasoning": "Username field is visible and ready for input"
            }"""
        }
        
        coordinate_response = {
            "content": """{
                "cell": "K15",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.85,
                "reasoning": "Username input field located"
            }"""
        }
        
        focus_response = {
            "content": """{
                "focused": true,
                "confidence": 0.95,
                "element_type": "input",
                "reasoning": "Input field is focused and ready for text"
            }"""
        }
        
        analysis_response = {
            "content": """{
                "success": true,
                "confidence": 0.9,
                "actual_outcome": "Username was typed into the field",
                "matches_expected": true,
                "ui_changes": ["Input field contains text"],
                "recommendations": [],
                "anomalies": []
            }"""
        }
        
        action_agent.call_openai_with_debug = AsyncMock(side_effect=[
            validation_response,
            coordinate_response,
            focus_response,
            analysis_response
        ])
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=type_step,
            test_context=test_context,
            screenshot=sample_screenshot
        )
        
        # Verify result
        assert result.overall_success is True
        assert result.validation.valid is True
        assert result.coordinate.grid_cell == "K15"
        
        # Verify typing was called
        action_agent.browser_driver.type_text.assert_called_once_with("testuser@example.com")
    
    @pytest.mark.asyncio
    async def test_execute_action_validation_failure(
        self, action_agent, test_step, test_context, sample_screenshot
    ):
        """Test action execution when validation fails."""
        # Mock validation failure
        validation_response = {
            "content": """{
                "valid": false,
                "confidence": 0.8,
                "reasoning": "Login button not found on current page",
                "concerns": ["Page might not be loaded", "Button might be hidden"],
                "suggestions": ["Wait for page to load", "Check if login modal needs to be opened"]
            }"""
        }
        
        action_agent.call_openai_with_debug = AsyncMock(return_value=validation_response)
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=test_step,
            test_context=test_context,
            screenshot=sample_screenshot
        )
        
        # Verify result
        assert result.overall_success is False
        assert result.validation.valid is False
        assert result.failure_phase == "validation"
        assert len(result.validation.concerns) == 2
        assert len(result.validation.suggestions) == 2
        
        # Verify no browser action was taken
        action_agent.browser_driver.click.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_conversation_history_management(
        self, action_agent, test_step, test_context, sample_screenshot
    ):
        """Test conversation history management across action execution."""
        # Initial conversation should be empty
        assert len(action_agent.conversation_history) == 0
        
        # Mock responses
        validation_response = {"content": '{"valid": true, "confidence": 0.9, "reasoning": "OK"}'}
        coordinate_response = {"content": '{"cell": "A1", "offset_x": 0.5, "offset_y": 0.5, "confidence": 0.9, "reasoning": "Found"}'}
        analysis_response = {"content": '{"success": true, "confidence": 0.9, "actual_outcome": "Done", "matches_expected": true, "ui_changes": [], "recommendations": [], "anomalies": []}'}
        
        action_agent.call_openai_with_debug = AsyncMock(side_effect=[
            validation_response,
            coordinate_response,
            analysis_response
        ])
        
        # Execute action
        await action_agent.execute_action(test_step, test_context, sample_screenshot)
        
        # Conversation history should have messages
        assert len(action_agent.conversation_history) > 0
        
        # Reset conversation
        action_agent.reset_conversation()
        assert len(action_agent.conversation_history) == 0
    
    @pytest.mark.asyncio
    async def test_execute_action_unknown_type(self, action_agent, test_context):
        """Test handling of unknown action type."""
        unknown_step = TestStep(
            step_number=1,
            action="Do something unknown",
            expected_result="Unknown result",
            action_instruction=ActionInstruction(
                action_type="unknown_action",  # Invalid type
                description="Unknown action",
                target="Unknown",
                expected_outcome="Unknown"
            )
        )
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=unknown_step,
            test_context=test_context
        )
        
        # Verify result
        assert result.overall_success is False
        assert result.failure_phase == "routing"
        assert "Unknown action type" in result.execution.error_message
    
    @pytest.mark.asyncio
    async def test_coordinate_refinement(
        self, action_agent, sample_screenshot
    ):
        """Test coordinate refinement when confidence is low."""
        instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click small button",
            target="Small button",
            expected_outcome="Button clicked"
        )
        
        # Mock initial low confidence response
        initial_response = {
            "content": """{
                "cell": "B5",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.6,
                "reasoning": "Button might be in B5 but hard to see"
            }"""
        }
        
        # Mock refinement response
        refined_response = {
            "content": """{
                "sub_cell": "B2",
                "offset_x": 0.3,
                "offset_y": 0.7,
                "confidence": 0.95,
                "reasoning": "Button clearly visible in refined view"
            }"""
        }
        
        action_agent.call_openai_with_debug = AsyncMock(side_effect=[
            initial_response,
            refined_response
        ])
        
        # Execute determine_action
        result = await action_agent.determine_action(
            screenshot=sample_screenshot,
            instruction=instruction
        )
        
        # Verify refinement was applied
        assert result.coordinate.confidence == 0.95
        assert result.coordinate.refined is True
        
        # Verify two API calls were made
        assert action_agent.call_openai_with_debug.call_count == 2


class TestActionAgentScrolling:
    """Test scrolling functionality."""
    
    @pytest.mark.asyncio
    async def test_scroll_to_element_workflow(
        self, action_agent, test_context, sample_screenshot
    ):
        """Test scroll to element workflow."""
        scroll_step = TestStep(
            step_number=1,
            action="Scroll to submit button",
            expected_result="Submit button is visible",
            action_instruction=ActionInstruction(
                action_type=ActionType.SCROLL_TO_ELEMENT,
                description="Scroll to submit button",
                target="Submit button",
                expected_outcome="Submit button is visible on screen"
            )
        )
        
        # Mock visibility check responses
        visibility_responses = [
            {"content": '{"visible": false, "confidence": 0.9, "location": "below", "reasoning": "Button is below viewport"}'},
            {"content": '{"visible": true, "confidence": 0.95, "location": "center", "reasoning": "Button is now visible"}'}
        ]
        
        # Mock scroll planning
        scroll_response = {
            "content": """{
                "direction": "down",
                "distance": 500,
                "confidence": 0.85,
                "reasoning": "Need to scroll down to reach submit button"
            }"""
        }
        
        action_agent.call_openai_with_debug = AsyncMock(side_effect=[
            visibility_responses[0],  # Initial check - not visible
            scroll_response,          # Plan scroll
            visibility_responses[1]   # After scroll - visible
        ])
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=scroll_step,
            test_context=test_context,
            screenshot=sample_screenshot
        )
        
        # Verify result
        assert result.overall_success is True
        
        # Verify scrolling was performed
        action_agent.browser_driver.scroll.assert_called()


class TestActionAgentErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_api_error_handling(
        self, action_agent, test_step, test_context, sample_screenshot
    ):
        """Test handling of API errors."""
        # Mock API error
        action_agent.call_openai_with_debug = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=test_step,
            test_context=test_context,
            screenshot=sample_screenshot
        )
        
        # Verify error handling
        assert result.overall_success is False
        assert "API rate limit exceeded" in str(result.execution.error_message)
        assert result.execution.error_traceback is not None
    
    @pytest.mark.asyncio
    async def test_browser_error_handling(
        self, action_agent, test_step, test_context, sample_screenshot
    ):
        """Test handling of browser errors."""
        # Mock successful validation and coordinate determination
        validation_response = {"content": '{"valid": true, "confidence": 0.9, "reasoning": "OK"}'}
        coordinate_response = {"content": '{"cell": "M23", "offset_x": 0.5, "offset_y": 0.5, "confidence": 0.9, "reasoning": "Found"}'}
        
        action_agent.call_openai_with_debug = AsyncMock(side_effect=[
            validation_response,
            coordinate_response
        ])
        
        # Mock browser error
        action_agent.browser_driver.click.side_effect = Exception("Element not interactable")
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=test_step,
            test_context=test_context,
            screenshot=sample_screenshot
        )
        
        # Verify error handling
        assert result.overall_success is False
        assert result.failure_phase == "execution"
        assert "Element not interactable" in result.execution.error_message
    
    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(
        self, action_agent, test_step, test_context, sample_screenshot
    ):
        """Test handling of invalid JSON in AI responses."""
        # Mock invalid JSON response
        invalid_response = {
            "content": "This is not valid JSON {invalid: json}"
        }
        
        action_agent.call_openai_with_debug = AsyncMock(return_value=invalid_response)
        
        # Execute action
        result = await action_agent.execute_action(
            test_step=test_step,
            test_context=test_context,
            screenshot=sample_screenshot
        )
        
        # Verify error handling
        assert result.overall_success is False
        assert result.validation.valid is False
        assert "Failed to parse" in result.validation.reasoning