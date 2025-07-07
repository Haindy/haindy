"""
Tests for the refactored Action Agent with full action lifecycle.
"""

import base64
import json
from datetime import datetime
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

from src.agents.action_agent import ActionAgent
from src.browser.driver import BrowserDriver
from src.core.types import ActionInstruction, ActionType, TestStep


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.grid_size = 60
    settings.grid_confidence_threshold = 0.8
    settings.grid_refinement_enabled = True
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.openai_temperature = 0.7
    settings.openai_max_retries = 3
    return settings


@pytest.fixture
def mock_browser_driver():
    """Mock browser driver for testing."""
    driver = Mock(spec=BrowserDriver)
    driver.get_page_url = AsyncMock(return_value="https://example.com")
    driver.get_page_title = AsyncMock(return_value="Example Page")
    driver.screenshot = AsyncMock()
    driver.click = AsyncMock()
    driver.type_text = AsyncMock()
    driver.wait = AsyncMock()
    driver.get_viewport_size = AsyncMock(return_value=(1920, 1080))
    return driver


@pytest.fixture
def action_agent_with_browser(mock_settings, mock_browser_driver):
    """Create an ActionAgent with browser driver."""
    with patch("src.agents.action_agent.get_settings", return_value=mock_settings):
        agent = ActionAgent(browser_driver=mock_browser_driver)
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
def sample_test_step():
    """Create a sample test step."""
    return TestStep(
        step_number=1,
        description="Click the login button",
        action_instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click the login button",
            target="Login button",
            expected_outcome="Login form is submitted"
        )
    )


@pytest.fixture
def test_context():
    """Create sample test context."""
    return {
        "test_plan_name": "Login Test",
        "current_step_description": "Click login button",
        "previous_steps_summary": "Navigated to login page",
        "total_steps": 3,
        "completed_steps": 1
    }


class TestActionAgentRefactored:
    """Test cases for refactored ActionAgent with full lifecycle."""
    
    @pytest.mark.asyncio
    async def test_execute_action_full_success(
        self, action_agent_with_browser, sample_test_step, test_context, sample_screenshot
    ):
        """Test successful action execution with all phases."""
        # Mock browser screenshot
        action_agent_with_browser.browser_driver.screenshot.return_value = sample_screenshot
        
        # Mock validation response
        validation_response = {
            "content": json.dumps({
                "valid": True,
                "confidence": 0.95,
                "reasoning": "Login button is clearly visible",
                "concerns": [],
                "suggestions": []
            })
        }
        
        # Mock coordinate determination response
        coordinate_response = {
            "content": json.dumps({
                "cell": "M23",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.92,
                "reasoning": "Button found in center of screen"
            })
        }
        
        # Mock analysis response
        analysis_response = {
            "content": json.dumps({
                "success": True,
                "confidence": 0.9,
                "actual_outcome": "Login form submitted successfully",
                "matches_expected": True,
                "ui_changes": ["Form disappeared", "Loading spinner appeared"],
                "recommendations": []
            })
        }
        
        # Set up mock responses in order
        action_agent_with_browser.call_openai = AsyncMock(
            side_effect=[validation_response, coordinate_response, analysis_response]
        )
        
        # Execute action
        result = await action_agent_with_browser.execute_action(
            sample_test_step,
            test_context,
            sample_screenshot
        )
        
        # Verify result structure
        assert result["action_type"] == "click"
        assert result["validation_passed"] is True
        assert result["validation_confidence"] == 0.95
        assert result["grid_cell"] == "M23"
        assert result["coordinate_confidence"] == 0.92
        assert result["execution_success"] is True
        assert result["ai_analysis"]["success"] is True
        
        # Verify browser interactions
        action_agent_with_browser.browser_driver.click.assert_called_once()
        # Screenshot is called once (initial), after screenshot is only taken if execution succeeds
        assert action_agent_with_browser.browser_driver.screenshot.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_execute_action_validation_failure(
        self, action_agent_with_browser, sample_test_step, test_context, sample_screenshot
    ):
        """Test action execution when validation fails."""
        # Mock validation failure
        validation_response = {
            "content": json.dumps({
                "valid": False,
                "confidence": 0.3,
                "reasoning": "Login button not found on screen",
                "concerns": ["Expected element not visible"],
                "suggestions": ["Check if page loaded correctly"]
            })
        }
        
        action_agent_with_browser.call_openai = AsyncMock(return_value=validation_response)
        
        # Execute action
        result = await action_agent_with_browser.execute_action(
            sample_test_step,
            test_context,
            sample_screenshot
        )
        
        # Verify validation failure
        assert result["validation_passed"] is False
        assert result["validation_reasoning"] == "Login button not found on screen"
        assert result["execution_success"] is False
        
        # Verify no browser action was attempted
        action_agent_with_browser.browser_driver.click.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_action_type_with_text(
        self, action_agent_with_browser, test_context, sample_screenshot
    ):
        """Test type action with text input."""
        # Create type action step
        type_step = TestStep(
            step_number=2,
            description="Enter username",
            action_instruction=ActionInstruction(
                action_type=ActionType.TYPE,
                description="Type username in input field",
                target="Username input",
                value="testuser",
                expected_outcome="Username is entered"
            )
        )
        
        # Mock successful validation and coordinate responses
        action_agent_with_browser.call_openai = AsyncMock(
            side_effect=[
                {"content": json.dumps({"valid": True, "confidence": 0.9, "reasoning": "Input visible"})},
                {"content": json.dumps({"cell": "K15", "offset_x": 0.5, "offset_y": 0.5, "confidence": 0.88})},
                {"content": json.dumps({"success": True, "confidence": 0.85, "actual_outcome": "Text entered"})}
            ]
        )
        
        # Execute action
        result = await action_agent_with_browser.execute_action(
            type_step,
            test_context,
            sample_screenshot
        )
        
        # Verify type action execution
        assert result["execution_success"] is True
        action_agent_with_browser.browser_driver.click.assert_called()  # Click to focus
        action_agent_with_browser.browser_driver.type_text.assert_called_with("testuser")
    
    @pytest.mark.asyncio
    async def test_execute_action_low_confidence_skip(
        self, action_agent_with_browser, sample_test_step, test_context, sample_screenshot
    ):
        """Test action skipped when coordinate confidence is too low."""
        # Mock responses with low coordinate confidence
        action_agent_with_browser.call_openai = AsyncMock(
            side_effect=[
                {"content": json.dumps({"valid": True, "confidence": 0.9, "reasoning": "OK"})},
                {"content": json.dumps({"cell": "A1", "offset_x": 0.5, "offset_y": 0.5, "confidence": 0.4})}
            ]
        )
        
        # Execute action
        result = await action_agent_with_browser.execute_action(
            sample_test_step,
            test_context,
            sample_screenshot
        )
        
        # Verify action was not executed due to low confidence
        # Grid refinement adds 0.25 to confidence, so 0.4 + 0.25 = 0.65
        assert result["coordinate_confidence"] == 0.65
        assert result["execution_success"] is False
        action_agent_with_browser.browser_driver.click.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_action_with_browser_error(
        self, action_agent_with_browser, sample_test_step, test_context, sample_screenshot
    ):
        """Test handling of browser execution errors."""
        # Mock successful validation and coordinates
        action_agent_with_browser.call_openai = AsyncMock(
            side_effect=[
                {"content": json.dumps({"valid": True, "confidence": 0.9, "reasoning": "OK"})},
                {"content": json.dumps({"cell": "M23", "offset_x": 0.5, "offset_y": 0.5, "confidence": 0.9})}
            ]
        )
        
        # Mock browser error
        action_agent_with_browser.browser_driver.click.side_effect = Exception("Element not clickable")
        
        # Execute action
        result = await action_agent_with_browser.execute_action(
            sample_test_step,
            test_context,
            sample_screenshot
        )
        
        # Verify error handling
        assert result["execution_success"] is False
        assert result["execution_error"] == "Element not clickable"
    
    def test_create_highlighted_screenshot(
        self, action_agent_with_browser, sample_screenshot
    ):
        """Test highlighted screenshot creation."""
        # Initialize grid
        action_agent_with_browser.grid_overlay.initialize(1920, 1080)
        
        # Create highlighted screenshot
        highlighted = action_agent_with_browser._create_highlighted_screenshot(
            sample_screenshot,
            "M23"
        )
        
        # Verify it's valid image data
        assert highlighted is not None
        assert len(highlighted) > 0
        
        # Verify we can load it
        img = Image.open(BytesIO(highlighted))
        assert img.size == (1920, 1080)
    
    @pytest.mark.asyncio
    async def test_validate_action_parsing_error(
        self, action_agent_with_browser, sample_test_step, test_context, sample_screenshot
    ):
        """Test validation with parsing error."""
        # Mock invalid response
        action_agent_with_browser.call_openai = AsyncMock(
            return_value={"content": "invalid json"}
        )
        
        # Execute validation
        result = await action_agent_with_browser._validate_action(
            sample_test_step.action_instruction,
            sample_screenshot,
            test_context
        )
        
        # Verify error handling
        assert result["valid"] is False
        assert "Failed to validate" in result["reasoning"]
    
    @pytest.mark.asyncio
    async def test_analyze_result_no_screenshots(
        self, action_agent_with_browser, sample_test_step
    ):
        """Test result analysis when screenshots are missing."""
        result = {
            "grid_cell": "M23",
            "execution_time_ms": 500,
            "url_before": "https://example.com",
            "url_after": "https://example.com/dashboard",
            "execution_success": True
        }
        
        # Analyze without screenshots
        analysis = await action_agent_with_browser._analyze_result(
            sample_test_step.action_instruction,
            result
        )
        
        # Verify fallback analysis
        assert analysis["success"] is True
        assert analysis["actual_outcome"] == "No screenshot comparison available"
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_determine_action(
        self, action_agent_with_browser, sample_screenshot
    ):
        """Test that determine_action still works for backward compatibility."""
        instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click button",
            target="Button",
            expected_outcome="Clicked"
        )
        
        # Mock coordinate response
        action_agent_with_browser.call_openai = AsyncMock(
            return_value={
                "content": json.dumps({
                    "cell": "B7",
                    "offset_x": 0.5,
                    "offset_y": 0.5,
                    "confidence": 0.85
                })
            }
        )
        
        # Call original method
        result = await action_agent_with_browser.determine_action(
            sample_screenshot,
            instruction
        )
        
        # Verify it still works
        assert result.coordinate.cell == "B7"
        assert result.coordinate.confidence == 0.85