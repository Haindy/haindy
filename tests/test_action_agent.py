"""
Tests for the Action Agent - simplified to match actual implementation.

These tests focus on the core functionality of the Action Agent without
getting into implementation details that may change.
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
    ActionInstruction, ActionType, TestStep,
    ScrollDirection, VisibilityStatus
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
    settings.openai_model = "gpt-5"
    settings.openai_temperature = 1.0
    settings.openai_max_retries = 3
    settings.openai_request_timeout_seconds = 900
    settings.actions_use_computer_tool = False
    settings.actions_computer_tool_max_turns = 12
    settings.actions_computer_tool_action_timeout_ms = 7000
    settings.actions_computer_tool_stabilization_wait_ms = 1000
    settings.actions_computer_tool_fail_fast_on_safety = True
    return settings


@pytest.fixture
def mock_browser_driver():
    """Mock browser driver for testing."""
    driver = AsyncMock()
    driver.screenshot.return_value = b"screenshot_data"
    driver.get_page_url.return_value = "https://example.com"
    driver.get_page_title.return_value = "Example Page"
    driver.get_viewport_size.return_value = (1920, 1080)
    driver.navigate_to = AsyncMock()
    driver.click = AsyncMock()
    driver.type_text = AsyncMock()
    driver.press_key = AsyncMock()
    driver.scroll = AsyncMock()
    driver.scroll_by_pixels = AsyncMock()
    driver.wait_for_load = AsyncMock()
    driver.wait = AsyncMock()
    return driver


@pytest.fixture
def action_agent(mock_settings, mock_browser_driver):
    """Create an ActionAgent instance for testing."""
    with patch("src.agents.action_agent.get_settings", return_value=mock_settings):
        agent = ActionAgent(browser_driver=mock_browser_driver)
        # Mock the OpenAI client
        agent._client = AsyncMock()
        agent.call_openai = AsyncMock()
        agent.use_computer_tool = False
        agent.settings.actions_use_computer_tool = False
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
        description="Click the login button to open the login form",
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


class TestActionAgentBasics:
    """Basic tests for Action Agent functionality."""
    
    def test_initialization(self, action_agent):
        """Test that action agent initializes correctly."""
        assert action_agent.name == "ActionAgent"
        assert action_agent.browser_driver is not None
        assert action_agent.grid_overlay is not None
        assert action_agent.grid_refinement is not None
        assert action_agent.confidence_threshold == 0.8
        assert action_agent.refinement_enabled is True
    
    def test_conversation_reset(self, action_agent):
        """Test conversation history reset."""
        # Add some dummy history
        action_agent.conversation_history = [{"role": "user", "content": "test"}]
        
        # Reset
        action_agent.reset_conversation()
        
        # Verify cleared
        assert len(action_agent.conversation_history) == 0
    
    @pytest.mark.asyncio
    async def test_execute_action_returns_result(
        self, action_agent, test_step, test_context, sample_screenshot
    ):
        """Test that execute_action returns an EnhancedActionResult."""
        # Mock the workflow method to return a result
        mock_result = EnhancedActionResult(
            test_step_id=test_step.step_id,
            test_step=test_step,
            test_context=test_context,
            overall_success=True,
            timestamp_end=datetime.now(timezone.utc),
            validation=ValidationResult(
                valid=True,
                confidence=0.9,
                reasoning="Test validation"
            )
        )
        
        with patch.object(action_agent, '_execute_click_workflow', AsyncMock(return_value=mock_result)):
            result = await action_agent.execute_action(
                test_step=test_step,
                test_context=test_context,
                screenshot=sample_screenshot
            )
            
            # Verify result
            assert isinstance(result, EnhancedActionResult)
            assert result.test_step_id == test_step.step_id
    
    @pytest.mark.asyncio
    async def test_execute_action_routes_correctly(
        self, action_agent, test_context
    ):
        """Test that execute_action routes to correct workflow based on action type."""
        # Create different action types
        nav_step = TestStep(
            step_number=1,
            description="Navigate to page",
            action="Navigate",
            expected_result="Page loaded",
            action_instruction=ActionInstruction(
                action_type=ActionType.NAVIGATE,
                description="Navigate",
                target="Page",
                value="https://example.com",
                expected_outcome="Page loaded"
            )
        )
        
        # Mock the navigate workflow
        with patch.object(action_agent, '_execute_navigate_workflow', AsyncMock()) as mock_nav:
            await action_agent.execute_action(nav_step, test_context)
            mock_nav.assert_called_once()
        
        # Test click routing
        click_step = TestStep(
            step_number=1,
            description="Click button",
            action="Click",
            expected_result="Clicked",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click",
                target="Button",
                expected_outcome="Button clicked"
            )
        )
        
        with patch.object(action_agent, '_execute_click_workflow', AsyncMock()) as mock_click:
            await action_agent.execute_action(click_step, test_context)
            mock_click.assert_called_once()

        # Test skip navigation routing
        skip_step = TestStep(
            step_number=1,
            description="Confirm current page is already correct",
            action="Skip navigation",
            expected_result="Remain on current page",
            action_instruction=ActionInstruction(
                action_type=ActionType.SKIP_NAVIGATION,
                description="No navigation needed; the target page is already visible.",
                target="Current page state",
                expected_outcome="Remain on current page"
            )
        )

        with patch.object(action_agent, '_execute_skip_navigation_workflow', AsyncMock()) as mock_skip:
            await action_agent.execute_action(skip_step, test_context)
            mock_skip.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_navigation_workflow_success(self, action_agent, test_context):
        """Skip navigation workflow should return success without side effects."""
        skip_step = TestStep(
            step_number=2,
            description="Skip redundant navigation",
            action="Skip navigation",
            expected_result="Stay on the current page",
            action_instruction=ActionInstruction(
                action_type=ActionType.SKIP_NAVIGATION,
                description="Already on the desired page; no navigation needed.",
                target="Current page",
                expected_outcome="Stay on the current page"
            )
        )

        with patch("src.agents.action_agent.get_debug_logger", return_value=None):
            result = await action_agent._execute_skip_navigation_workflow(skip_step, test_context)

        assert result.overall_success is True
        assert result.execution.success is True
        assert result.validation.valid is True
        assert result.execution.error_message is None
    
    @pytest.mark.asyncio
    async def test_determine_action_basic(
        self, action_agent, sample_screenshot
    ):
        """Test basic determine_action functionality."""
        instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click button",
            target="Button",
            expected_outcome="Button clicked"
        )
        
        # Mock the response
        mock_response = {
            "content": '{"cell": "M23", "offset_x": 0.5, "offset_y": 0.5, "confidence": 0.9, "reasoning": "Found button"}'
        }
        
        with patch.object(action_agent, 'call_openai_with_debug', AsyncMock(return_value=mock_response)):
            result = await action_agent.determine_action(
                screenshot=sample_screenshot,
                instruction=instruction
            )
            
            # Just verify it returns a result with coordinates
            assert result.coordinate is not None
            assert hasattr(result.coordinate, 'cell')
            assert hasattr(result.coordinate, 'confidence')


class TestActionAgentIntegration:
    """Integration-style tests that don't rely on specific implementation details."""
    
    @pytest.mark.asyncio
    async def test_navigate_action_workflow(
        self, action_agent, test_context
    ):
        """Test navigation action executes without errors."""
        nav_step = TestStep(
            step_number=1,
            description="Navigate to login page",
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
        
        # Mock the workflow to return a successful result
        mock_result = EnhancedActionResult(
            test_step_id=nav_step.step_id,
            test_step=nav_step,
            test_context=test_context,
            overall_success=True,
            timestamp_end=datetime.now(timezone.utc),
            validation=ValidationResult(
                valid=True,
                confidence=1.0,
                reasoning="Navigation successful"
            )
        )
        
        with patch.object(action_agent, '_execute_navigate_workflow', AsyncMock(return_value=mock_result)):
            result = await action_agent.execute_action(
                test_step=nav_step,
                test_context=test_context
            )
            
            # Basic checks
            assert isinstance(result, EnhancedActionResult)
            assert result.test_step_id == nav_step.step_id
            assert result.overall_success is True
    
    @pytest.mark.asyncio 
    async def test_error_handling_captures_exceptions(
        self, action_agent, test_step, test_context
    ):
        """Test that exceptions are captured in the result."""
        # Make the browser driver raise an exception
        action_agent.browser_driver.screenshot.side_effect = Exception("Test error")
        
        # Execute - should not raise, but capture error
        result = await action_agent.execute_action(
            test_step=test_step,
            test_context=test_context
        )
        
        # Verify error was captured
        assert isinstance(result, EnhancedActionResult)
        assert result.execution is not None
        if result.execution.error_message:
            assert "Test error" in result.execution.error_message or "error" in result.execution.error_message.lower()


class TestActionAgentScrolling:
    """Test scrolling functionality basics."""
    
    @pytest.mark.asyncio
    async def test_scroll_action_routes_correctly(
        self, action_agent, test_context
    ):
        """Test scroll actions route to correct workflows."""
        scroll_step = TestStep(
            step_number=1,
            description="Scroll to element",
            action="Scroll to submit button",
            expected_result="Submit button is visible",
            action_instruction=ActionInstruction(
                action_type=ActionType.SCROLL_TO_ELEMENT,
                description="Scroll to submit button",
                target="Submit button",
                expected_outcome="Submit button is visible"
            )
        )
        
        # Mock the scroll workflow
        with patch.object(action_agent, '_execute_scroll_to_element_workflow', AsyncMock()) as mock_scroll:
            await action_agent.execute_action(scroll_step, test_context)
            mock_scroll.assert_called_once()


class TestActionAgentUtilities:
    """Test utility methods."""
    
    def test_create_overlay_image(self, action_agent, sample_screenshot):
        """Test overlay image creation doesn't crash."""
        # Just verify it returns bytes and doesn't crash
        result = action_agent._create_overlay_image(sample_screenshot)
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_build_analysis_prompt(self, action_agent):
        """Test prompt building."""
        instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click the submit button",
            target="Submit button",
            expected_outcome="Form is submitted"
        )
        
        prompt = action_agent._build_analysis_prompt(instruction)
        
        # Basic checks
        assert isinstance(prompt, str)
        assert "submit button" in prompt.lower()
        assert len(prompt) > 0
