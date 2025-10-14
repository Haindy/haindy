"""
Tests for scroll action functionality.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest

from src.agents.action_agent import ActionAgent
from src.browser.driver import PlaywrightDriver
from src.core.types import (
    ActionInstruction, ActionType, TestStep, TestStatus,
    ScrollDirection, VisibilityStatus, ScrollAction, ScrollState,
    VisibilityResult, GridCoordinate
)
from src.core.enhanced_types import EnhancedActionResult


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.grid_size = 60
    settings.grid_confidence_threshold = 0.8
    settings.grid_refinement_enabled = True
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-5"
    settings.openai_temperature = 0.7
    settings.openai_max_retries = 3
    settings.openai_request_timeout_seconds = 900
    settings.actions_use_computer_tool = False
    settings.actions_computer_tool_max_turns = 12
    settings.actions_computer_tool_action_timeout_ms = 7000
    settings.actions_computer_tool_stabilization_wait_ms = 1000
    settings.actions_computer_tool_fail_fast_on_safety = True
    settings.browser_headless = True
    settings.browser_viewport_width = 1920
    settings.browser_viewport_height = 1080
    settings.browser_timeout = 30000
    return settings


@pytest.fixture
def mock_browser_driver():
    """Create a mock browser driver."""
    from PIL import Image
    import io
    
    # Create a proper mock screenshot
    img = Image.new('RGB', (1920, 1080), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    mock_screenshot_bytes = buffer.getvalue()
    
    driver = AsyncMock(spec=PlaywrightDriver)
    driver.screenshot = AsyncMock(return_value=mock_screenshot_bytes)
    driver.scroll_by_pixels = AsyncMock()
    driver.scroll_to_top = AsyncMock()
    driver.scroll_to_bottom = AsyncMock()
    driver.get_page_url = AsyncMock(return_value="https://example.com")
    driver.get_page_title = AsyncMock(return_value="Test Page")
    driver.get_viewport_size = AsyncMock(return_value=(1920, 1080))
    driver.get_scroll_position = AsyncMock(return_value=(0, 0))
    driver.get_page_dimensions = AsyncMock(return_value=(1920, 1080, 1920, 3000))
    driver.press_key = AsyncMock()
    driver.wait = AsyncMock()
    return driver


@pytest.fixture
def action_agent(mock_settings, mock_browser_driver):
    """Create an ActionAgent instance for testing."""
    with patch("src.agents.action_agent.get_settings", return_value=mock_settings):
        agent = ActionAgent(browser_driver=mock_browser_driver)
        # Mock the OpenAI client
        agent._client = AsyncMock()
        agent.call_openai_with_debug = AsyncMock()
        return agent


@pytest.fixture
def scroll_test_step():
    """Create a test step for scroll to element."""
    return TestStep(
        step_number=1,
        action="Scroll to the submit button",
        expected_result="Submit button is visible",
        action_instruction=ActionInstruction(
            action_type=ActionType.SCROLL_TO_ELEMENT,
            description="Scroll to the submit button",
            target="Submit button",
            expected_outcome="Submit button is visible"
        ),
        description="Scroll to the submit button"
    )


class TestScrollActions:
    """Test scroll action functionality."""
    
    @pytest.mark.asyncio
    async def test_scroll_by_pixels_workflow(self, action_agent, mock_browser_driver):
        """Test scroll by pixels workflow."""
        test_step = TestStep(
            step_number=1,
            action="Scroll down 300 pixels",
            expected_result="Page scrolled down",
            action_instruction=ActionInstruction(
                action_type=ActionType.SCROLL_BY_PIXELS,
                description="Scroll down 300 pixels",
                value="y=300",
                expected_outcome="Page scrolled down"
            ),
            description="Scroll down by 300 pixels"
        )
        
        result = await action_agent._execute_scroll_by_pixels_workflow(
            test_step, {}, None
        )
        
        assert result.overall_success is True
        assert result.execution.success is True
        # Verify the scroll method was called correctly
        mock_browser_driver.scroll_by_pixels.assert_called_once_with(0, 300)
    
    @pytest.mark.asyncio
    async def test_scroll_to_top_workflow(self, action_agent, mock_browser_driver):
        """Test scroll to top workflow."""
        test_step = TestStep(
            step_number=1,
            action="Scroll to top of page",
            expected_result="At top of page",
            action_instruction=ActionInstruction(
                action_type=ActionType.SCROLL_TO_TOP,
                description="Scroll to top of page",
                expected_outcome="At top of page"
            ),
            description="Scroll to top"
        )
        
        # Mock scroll position at top
        mock_browser_driver.get_scroll_position.return_value = (0, 0)
        
        result = await action_agent._execute_scroll_to_top_workflow(
            test_step, {}, None
        )
        
        assert result.overall_success is True
        assert result.execution.success is True
        assert result.ai_analysis.matches_expected is True
        mock_browser_driver.scroll_to_top.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scroll_to_bottom_workflow(self, action_agent, mock_browser_driver):
        """Test scroll to bottom workflow."""
        test_step = TestStep(
            step_number=1,
            action="Scroll to bottom of page",
            expected_result="At bottom of page",
            action_instruction=ActionInstruction(
                action_type=ActionType.SCROLL_TO_BOTTOM,
                description="Scroll to bottom of page",
                expected_outcome="At bottom of page"
            ),
            description="Scroll to bottom"
        )
        
        # Mock scroll position at bottom
        mock_browser_driver.get_scroll_position.return_value = (0, 1920)
        
        result = await action_agent._execute_scroll_to_bottom_workflow(
            test_step, {}, None
        )
        
        assert result.overall_success is True
        assert result.execution.success is True
        mock_browser_driver.scroll_to_bottom.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scroll_horizontal_workflow(self, action_agent, mock_browser_driver):
        """Test horizontal scroll workflow."""
        test_step = TestStep(
            step_number=1,
            action="Scroll right 400 pixels",
            expected_result="Scrolled horizontally",
            action_instruction=ActionInstruction(
                action_type=ActionType.SCROLL_HORIZONTAL,
                description="Scroll right 400 pixels",
                value="400",
                expected_outcome="Scrolled horizontally"
            ),
            description="Scroll right 400 pixels"
        )
        
        result = await action_agent._execute_scroll_horizontal_workflow(
            test_step, {}, None
        )
        
        assert result.overall_success is True
        assert result.execution.success is True
        # Verify the scroll method was called correctly
        mock_browser_driver.scroll_by_pixels.assert_called_once_with(400, 0)
    
    def test_calculate_scroll_distance(self, action_agent):
        """Test scroll distance calculation."""
        state = ScrollState(target_element="Submit button")
        
        # Test high confidence
        visibility = VisibilityResult(
            status=VisibilityStatus.NOT_VISIBLE,
            direction_confidence=0.95
        )
        distance = action_agent._calculate_scroll_distance(state, visibility)
        assert distance == 600  # High confidence = 600px base
        
        # Test medium confidence
        visibility.direction_confidence = 0.75
        distance = action_agent._calculate_scroll_distance(state, visibility)
        assert distance == 400  # Medium confidence = 400px base
        
        # Test low confidence
        visibility.direction_confidence = 0.50
        distance = action_agent._calculate_scroll_distance(state, visibility)
        assert distance == 200  # Low confidence = 200px base
        
        # Test convergence with multiple attempts
        state.attempts = 5
        distance = action_agent._calculate_scroll_distance(state, visibility)
        assert distance < 200  # Should be reduced due to attempts
    
    def test_detect_overshoot(self, action_agent):
        """Test overshoot detection."""
        state = ScrollState(target_element="Submit button")
        
        # No overshoot on first attempt
        visibility = VisibilityResult(status=VisibilityStatus.NOT_VISIBLE)
        assert action_agent._detect_overshoot(state, visibility) is False
        
        # Setup for overshoot detection
        state.attempts = 2
        state.element_partially_visible = True
        state.last_direction = ScrollDirection.DOWN
        state.scroll_history = [
            ScrollAction(direction=ScrollDirection.DOWN, distance=400)
        ]
        
        # Test overshoot: was partially visible, now not visible
        assert action_agent._detect_overshoot(state, visibility) is True
        
        # Test direction reversal with high confidence
        state.element_partially_visible = False
        visibility.suggested_direction = ScrollDirection.UP
        visibility.direction_confidence = 0.85
        assert action_agent._detect_overshoot(state, visibility) is True
    
    def test_is_oscillating(self, action_agent):
        """Test oscillation detection."""
        # No oscillation with few scrolls
        history = [
            ScrollAction(direction=ScrollDirection.DOWN, distance=400),
            ScrollAction(direction=ScrollDirection.DOWN, distance=300)
        ]
        assert action_agent._is_oscillating(history) is False
        
        # Oscillating pattern
        history = [
            ScrollAction(direction=ScrollDirection.DOWN, distance=400),
            ScrollAction(direction=ScrollDirection.UP, distance=200),
            ScrollAction(direction=ScrollDirection.DOWN, distance=300),
            ScrollAction(direction=ScrollDirection.UP, distance=150)
        ]
        assert action_agent._is_oscillating(history) is True
    
    def test_parse_visibility_response(self, action_agent):
        """Test parsing AI visibility response."""
        # Test fully visible response
        response = """STATUS: FULLY_VISIBLE
COORDINATES: M23
CONFIDENCE: 95
VISIBLE_PERCENT: 100
DIRECTION: N/A
DIRECTION_CONFIDENCE: 0
NOTES: Submit button clearly visible in center"""
        
        result = action_agent._parse_visibility_response(response)
        assert result.status == VisibilityStatus.FULLY_VISIBLE
        assert result.coordinates.cell == "M23"
        assert result.coordinates.confidence == 0.95
        
        # Test partially visible response
        response = """STATUS: PARTIALLY_VISIBLE
COORDINATES: B59
CONFIDENCE: 80
VISIBLE_PERCENT: 30
DIRECTION: DOWN
DIRECTION_CONFIDENCE: 90
NOTES: Only top portion of button visible"""
        
        result = action_agent._parse_visibility_response(response)
        assert result.status == VisibilityStatus.PARTIALLY_VISIBLE
        assert result.visible_percentage == 30
        assert result.suggested_direction == ScrollDirection.DOWN
        assert result.direction_confidence == 0.90
        
        # Test not visible response
        response = """STATUS: NOT_VISIBLE
COORDINATES: none
CONFIDENCE: 0
VISIBLE_PERCENT: 0
DIRECTION: DOWN
DIRECTION_CONFIDENCE: 85
NOTES: Button likely below current viewport"""
        
        result = action_agent._parse_visibility_response(response)
        assert result.status == VisibilityStatus.NOT_VISIBLE
        assert result.coordinates is None
        assert result.suggested_direction == ScrollDirection.DOWN


class TestScrollToElement:
    """Test scroll to element workflow specifically."""
    
    @pytest.mark.asyncio
    async def test_scroll_to_element_immediate_success(
        self, action_agent, scroll_test_step, mock_browser_driver
    ):
        """Test when element is immediately visible."""
        # Mock AI response for fully visible element
        action_agent.call_openai_with_debug.return_value = {
            "content": """STATUS: FULLY_VISIBLE
COORDINATES: M23
CONFIDENCE: 95
VISIBLE_PERCENT: 100
DIRECTION: N/A
DIRECTION_CONFIDENCE: 0
NOTES: Submit button clearly visible"""
        }
        
        result = await action_agent._execute_scroll_to_element_workflow(
            scroll_test_step, {}, None
        )
        
        assert result.overall_success is True
        assert result.ai_analysis.success is True
        # No need to check coordinates for scroll actions
        
        # Should not have scrolled
        mock_browser_driver.scroll_by_pixels.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_scroll_to_element_with_scrolling(
        self, action_agent, scroll_test_step, mock_browser_driver
    ):
        """Test when element requires scrolling."""
        # First call: not visible
        # Second call: fully visible
        action_agent.call_openai_with_debug.side_effect = [
            {
                "content": """STATUS: NOT_VISIBLE
COORDINATES: none
CONFIDENCE: 0
VISIBLE_PERCENT: 0
DIRECTION: DOWN
DIRECTION_CONFIDENCE: 90
NOTES: Element likely below viewport"""
            },
            {
                "content": """STATUS: FULLY_VISIBLE
COORDINATES: M45
CONFIDENCE: 95
VISIBLE_PERCENT: 100
DIRECTION: N/A
DIRECTION_CONFIDENCE: 0
NOTES: Submit button now visible"""
            }
        ]
        
        result = await action_agent._execute_scroll_to_element_workflow(
            scroll_test_step, {}, None
        )
        
        assert result.overall_success is True
        assert result.ai_analysis.success is True
        
        # Should have scrolled down
        mock_browser_driver.scroll_by_pixels.assert_called()
        call_args = mock_browser_driver.scroll_by_pixels.call_args[0]
        assert call_args[0] == 0  # x
        assert call_args[1] > 0   # y (positive = down)
    
    @pytest.mark.asyncio
    async def test_scroll_to_element_max_attempts(
        self, action_agent, scroll_test_step, mock_browser_driver
    ):
        """Test when element is not found after max attempts."""
        # Always return not visible
        action_agent.call_openai_with_debug.return_value = {
            "content": """STATUS: NOT_VISIBLE
COORDINATES: none
CONFIDENCE: 0
VISIBLE_PERCENT: 0
DIRECTION: DOWN
DIRECTION_CONFIDENCE: 70
NOTES: Cannot locate element"""
        }
        
        # Reduce max attempts for faster test
        scroll_test_step.action_instruction.target = "Nonexistent button"
        
        # Mock max_attempts to reduce test time
        original_init = ScrollState.__init__
        
        def mock_init(self, **kwargs):
            original_init(self, **kwargs)
            self.max_attempts = 3  # Reduce for test
        
        with patch.object(ScrollState, "__init__", mock_init):
            result = await action_agent._execute_scroll_to_element_workflow(
                scroll_test_step, {}, None
            )
        
        assert result.overall_success is False
        assert result.failure_phase == "max_attempts"
        assert "not found after" in result.execution.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_scroll_to_element_with_overshoot_correction(
        self, action_agent, scroll_test_step, mock_browser_driver
    ):
        """Test overshoot detection and correction."""
        # Sequence: not visible -> partially visible -> not visible (overshot) -> fully visible
        action_agent.call_openai_with_debug.side_effect = [
            {
                "content": """STATUS: NOT_VISIBLE
COORDINATES: none
CONFIDENCE: 0
VISIBLE_PERCENT: 0
DIRECTION: DOWN
DIRECTION_CONFIDENCE: 90
NOTES: Element below viewport"""
            },
            {
                "content": """STATUS: PARTIALLY_VISIBLE
COORDINATES: M58
CONFIDENCE: 70
VISIBLE_PERCENT: 40
DIRECTION: DOWN
DIRECTION_CONFIDENCE: 80
NOTES: Bottom part of button visible"""
            },
            {
                "content": """STATUS: NOT_VISIBLE
COORDINATES: none
CONFIDENCE: 0
VISIBLE_PERCENT: 0
DIRECTION: UP
DIRECTION_CONFIDENCE: 85
NOTES: Scrolled past the element"""
            },
            {
                "content": """STATUS: FULLY_VISIBLE
COORDINATES: M45
CONFIDENCE: 95
VISIBLE_PERCENT: 100
DIRECTION: N/A
DIRECTION_CONFIDENCE: 0
NOTES: Button now fully visible"""
            }
        ]
        
        result = await action_agent._execute_scroll_to_element_workflow(
            scroll_test_step, {}, None
        )
        
        assert result.overall_success is True
        assert result.ai_analysis.success is True
        
        # Verify scroll pattern: down, down, up (correction), done
        calls = mock_browser_driver.scroll_by_pixels.call_args_list
        assert len(calls) == 3
        assert calls[0][0][1] > 0  # First scroll down
        assert calls[1][0][1] > 0  # Second scroll down
        assert calls[2][0][1] < 0  # Third scroll up (correction)


class TestBrowserDriverScrollMethods:
    """Test browser driver scroll methods."""
    
    @pytest.mark.asyncio
    async def test_browser_scroll_by_pixels(self, mock_settings):
        """Test scroll by pixels in browser driver."""
        with patch("src.browser.driver.get_settings", return_value=mock_settings):
            driver = PlaywrightDriver()
            driver._page = AsyncMock()
            driver._page.evaluate = AsyncMock()
            
            await driver.scroll_by_pixels(100, 200, smooth=True)
            
            driver._page.evaluate.assert_called_once()
            call_arg = driver._page.evaluate.call_args[0][0]
            assert "scrollBy" in call_arg
            assert "100" in call_arg  # x
            assert "200" in call_arg  # y
            assert "smooth" in call_arg
    
    @pytest.mark.asyncio
    async def test_browser_scroll_to_top(self, mock_settings):
        """Test scroll to top in browser driver."""
        with patch("src.browser.driver.get_settings", return_value=mock_settings):
            driver = PlaywrightDriver()
            driver._page = AsyncMock()
            driver._page.evaluate = AsyncMock()
            
            await driver.scroll_to_top()
            
            driver._page.evaluate.assert_called_once_with("window.scrollTo(0, 0)")
    
    @pytest.mark.asyncio
    async def test_browser_scroll_to_bottom(self, mock_settings):
        """Test scroll to bottom in browser driver."""
        with patch("src.browser.driver.get_settings", return_value=mock_settings):
            driver = PlaywrightDriver()
            driver._page = AsyncMock()
            driver._page.evaluate = AsyncMock()
            
            await driver.scroll_to_bottom()
            
            driver._page.evaluate.assert_called_once_with(
                "window.scrollTo(0, document.body.scrollHeight)"
            )
    
    @pytest.mark.asyncio
    async def test_browser_get_scroll_position(self, mock_settings):
        """Test getting scroll position."""
        with patch("src.browser.driver.get_settings", return_value=mock_settings):
            driver = PlaywrightDriver()
            driver._page = AsyncMock()
            driver._page.evaluate = AsyncMock(return_value={"x": 100, "y": 500})
            
            x, y = await driver.get_scroll_position()
            
            assert x == 100
            assert y == 500
            driver._page.evaluate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_browser_get_page_dimensions(self, mock_settings):
        """Test getting page dimensions."""
        with patch("src.browser.driver.get_settings", return_value=mock_settings):
            driver = PlaywrightDriver()
            driver._page = AsyncMock()
            driver._page.evaluate = AsyncMock(return_value={
                "viewportWidth": 1920,
                "viewportHeight": 1080,
                "pageWidth": 1920,
                "pageHeight": 3000
            })
            
            vw, vh, pw, ph = await driver.get_page_dimensions()
            
            assert vw == 1920
            assert vh == 1080
            assert pw == 1920
            assert ph == 3000
