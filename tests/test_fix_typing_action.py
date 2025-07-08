"""
Tests for Phase 6: Fix Typing Action enhancements.

This tests the enhanced typing functionality that ensures proper focus
before typing, with multiple click strategies and focus validation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from src.agents.action_agent import ActionAgent
from src.browser.driver import BrowserDriver
from src.core.types import (
    ActionInstruction,
    ActionType,
    TestStep,
    GridCoordinate
)
from src.core.enhanced_types import (
    ValidationResult,
    CoordinateResult,
    ExecutionResult,
    BrowserState,
    AIAnalysis
)


@pytest.fixture
def mock_browser_driver():
    """Create a mock browser driver."""
    # Create a minimal valid PNG image
    from PIL import Image
    import io
    img = Image.new('RGB', (1920, 1080), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    fake_screenshot = buffer.getvalue()
    
    driver = AsyncMock()
    driver.get_viewport_size = AsyncMock(return_value=(1920, 1080))
    driver.get_page_url = AsyncMock(return_value="https://wikipedia.org")
    driver.get_page_title = AsyncMock(return_value="Wikipedia")
    driver.screenshot = AsyncMock(return_value=fake_screenshot)
    driver.click = AsyncMock()
    driver.type_text = AsyncMock()
    driver.wait = AsyncMock()
    return driver


@pytest.fixture
def mock_grid_overlay():
    """Create a mock grid overlay."""
    # Create a minimal valid PNG image for overlay
    from PIL import Image
    import io
    overlay_img = Image.new('RGB', (1920, 1080), color='white')
    buffer = io.BytesIO()
    overlay_img.save(buffer, format='PNG')
    overlay_png_bytes = buffer.getvalue()
    
    overlay = MagicMock()
    overlay.viewport_width = 1920
    overlay.viewport_height = 1080
    overlay.grid_size = 60
    overlay.cell_width = 32.0  # 1920 / 60
    overlay.cell_height = 18.0  # 1080 / 60
    overlay.initialize = MagicMock()
    overlay.create_overlay_image.return_value = overlay_png_bytes
    overlay.coordinate_to_pixels.return_value = (960, 540)
    overlay.get_cell_bounds.return_value = (928, 270, 32, 18)  # x, y, width, height for M15
    overlay.get_refinement_region.return_value = (928, 270, 96, 54)  # 3x3 cells around M15
    # Mock the _parse_cell_identifier to avoid issues
    overlay._parse_cell_identifier = MagicMock(return_value=(12, 14))  # M=12, 15th row
    return overlay


@pytest.fixture
def action_agent(mock_browser_driver, mock_grid_overlay):
    """Create an action agent with mocked dependencies."""
    agent = ActionAgent(
        name="TestActionAgent",
        browser_driver=mock_browser_driver
    )
    agent.grid_overlay = mock_grid_overlay
    
    # Mock the grid refinement to avoid low confidence issues
    from unittest.mock import MagicMock
    agent.grid_refinement = MagicMock()
    agent.grid_refinement.refine_coordinate.return_value = GridCoordinate(
        cell="M15",
        offset_x=0.5,
        offset_y=0.5,
        confidence=0.9,
        refined=True
    )
    
    # Disable refinement or set high threshold to ensure our mocked coordinates are used
    agent.refinement_enabled = False
    
    return agent


@pytest.fixture
def type_test_step():
    """Create a test step for typing action."""
    return TestStep(
        step_id=uuid4(),
        step_number=1,
        description="Type search query",
        action_instruction=ActionInstruction(
            action_type=ActionType.TYPE,
            description="Type 'Artificial Intelligence' in search box",
            target="search box",
            value="Artificial Intelligence",
            expected_outcome="Search term entered"
        ),
        dependencies=[],
        optional=False
    )


class TestEnhancedTypingAction:
    """Test enhanced typing functionality with focus validation."""
    
    @pytest.mark.asyncio
    async def test_type_with_successful_focus_on_first_click(
        self, action_agent, type_test_step, mock_browser_driver
    ):
        """Test typing succeeds when focus is achieved on first click."""
        # Mock the determine_action method to avoid complex AI parsing
        from src.core.types import GridAction
        async def mock_determine_action(screenshot, instruction):
            return GridAction(
                instruction=instruction,
                coordinate=GridCoordinate(
                    cell="M15",
                    offset_x=0.5,
                    offset_y=0.5,
                    confidence=0.9,
                    refined=False
                ),
                screenshot_before=None
            )
        
        # Mock validation to pass
        async def mock_validate_action(instruction, screenshot, context):
            from src.core.enhanced_types import ValidationResult
            return ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Search box visible"
            )
        
        # Mock focus validation to pass
        async def mock_validate_focus(x, y, target_desc=None):
            return True
        
        # Mock result analysis
        async def mock_analyze_result(instruction, result):
            from src.core.enhanced_types import AIAnalysis
            return AIAnalysis(
                success=True,
                confidence=0.95,
                actual_outcome="Text typed successfully",
                matches_expected=True,
                ui_changes=["text appeared in search box"],
                recommendations=[],
                anomalies=[]
            )
        
        with patch.object(action_agent, 'determine_action', new=mock_determine_action), \
             patch.object(action_agent, '_validate_action', new=mock_validate_action), \
             patch.object(action_agent, '_validate_focus_for_typing', new=mock_validate_focus), \
             patch.object(action_agent, '_analyze_result', new=mock_analyze_result):
            
            # Execute action
            result = await action_agent.execute_action(
                type_test_step,
                {"test_plan_name": "Wikipedia Search Test"}
            )
            
            # Debug output
            if not result.overall_success:
                print(f"Overall success: {result.overall_success}")
                print(f"Validation: {result.validation}")
                print(f"Coordinates: {result.coordinates}")
                print(f"Execution: {result.execution}")
                print(f"Failure phase: {result.failure_phase}")
            
            # Verify results
            assert result.overall_success is True
            assert result.validation.valid is True
            assert result.execution.success is True
            
            # Verify click was called once for focus
            assert mock_browser_driver.click.call_count == 1
            mock_browser_driver.click.assert_called_with(960, 540)
            
            # Verify text was typed
            mock_browser_driver.type_text.assert_called_once_with("Artificial Intelligence")
    
    @pytest.mark.asyncio
    async def test_type_with_double_click_strategy(
        self, action_agent, type_test_step, mock_browser_driver
    ):
        """Test typing falls back to double-click when single click doesn't focus."""
        # Mock the methods to simulate first focus failure, then success
        focus_call_count = 0
        async def mock_validate_focus(x, y, target_desc=None):
            nonlocal focus_call_count
            focus_call_count += 1
            # First call fails, second succeeds
            return focus_call_count > 1
        
        # Standard mocks for other methods
        from src.core.types import GridAction
        async def mock_determine_action(screenshot, instruction):
            return GridAction(
                instruction=instruction,
                coordinate=GridCoordinate(
                    cell="M15",
                    offset_x=0.5,
                    offset_y=0.5,
                    confidence=0.9,
                    refined=False
                ),
                screenshot_before=None
            )
        
        async def mock_validate_action(instruction, screenshot, context):
            from src.core.enhanced_types import ValidationResult
            return ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Search box visible"
            )
        
        async def mock_analyze_result(instruction, result):
            from src.core.enhanced_types import AIAnalysis
            return AIAnalysis(
                success=True,
                confidence=0.95,
                actual_outcome="Text typed successfully",
                matches_expected=True,
                ui_changes=["text appeared"],
                recommendations=[],
                anomalies=[]
            )
        
        with patch.object(action_agent, 'determine_action', new=mock_determine_action), \
             patch.object(action_agent, '_validate_action', new=mock_validate_action), \
             patch.object(action_agent, '_validate_focus_for_typing', new=mock_validate_focus), \
             patch.object(action_agent, '_analyze_result', new=mock_analyze_result):
            
            # Execute action
            result = await action_agent.execute_action(
                type_test_step,
                {"test_plan_name": "Wikipedia Search Test"}
            )
            
            # Verify results
            assert result.overall_success is True
            
            # Verify double-click strategy was used (3 clicks total)
            assert mock_browser_driver.click.call_count == 3
            
            # Verify text was typed
            mock_browser_driver.type_text.assert_called_once_with("Artificial Intelligence")
    
    @pytest.mark.asyncio
    async def test_type_with_long_wait_strategy(
        self, action_agent, type_test_step, mock_browser_driver
    ):
        """Test typing falls back to click with longer wait for slow-loading pages."""
        # Mock focus validation to fail twice, then succeed on third attempt
        focus_call_count = 0
        async def mock_validate_focus(x, y, target_desc=None):
            nonlocal focus_call_count
            focus_call_count += 1
            # First two calls fail, third succeeds
            return focus_call_count > 2
        
        # Standard mocks for other methods
        from src.core.types import GridAction
        async def mock_determine_action(screenshot, instruction):
            return GridAction(
                instruction=instruction,
                coordinate=GridCoordinate(
                    cell="M15",
                    offset_x=0.5,
                    offset_y=0.5,
                    confidence=0.9,
                    refined=False
                ),
                screenshot_before=None
            )
        
        async def mock_validate_action(instruction, screenshot, context):
            from src.core.enhanced_types import ValidationResult
            return ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Search box visible"
            )
        
        async def mock_analyze_result(instruction, result):
            from src.core.enhanced_types import AIAnalysis
            return AIAnalysis(
                success=True,
                confidence=0.95,
                actual_outcome="Text typed successfully",
                matches_expected=True,
                ui_changes=["text appeared"],
                recommendations=[],
                anomalies=[]
            )
        
        with patch.object(action_agent, 'determine_action', new=mock_determine_action), \
             patch.object(action_agent, '_validate_action', new=mock_validate_action), \
             patch.object(action_agent, '_validate_focus_for_typing', new=mock_validate_focus), \
             patch.object(action_agent, '_analyze_result', new=mock_analyze_result):
            
            # Execute action
            result = await action_agent.execute_action(
                type_test_step,
                {"test_plan_name": "Wikipedia Search Test"}
            )
            
            # Verify results
            assert result.overall_success is True
            
            # Verify all click strategies were tried (4 clicks total)
            assert mock_browser_driver.click.call_count == 4
            
            # Verify long wait was used
            wait_calls = mock_browser_driver.wait.call_args_list
            assert any(call[0][0] == 1000 for call in wait_calls), "Long wait (1000ms) should be used"
            
            # Verify text was typed
            mock_browser_driver.type_text.assert_called_once_with("Artificial Intelligence")
    
    @pytest.mark.asyncio
    async def test_type_fails_when_element_not_focusable(
        self, action_agent, type_test_step, mock_browser_driver
    ):
        """Test typing fails properly when element cannot be focused."""
        # Mock focus validation to always fail
        async def mock_validate_focus(x, y, target_desc=None):
            return False  # Always fail focus validation
        
        # Standard mocks for other methods
        from src.core.types import GridAction
        async def mock_determine_action(screenshot, instruction):
            return GridAction(
                instruction=instruction,
                coordinate=GridCoordinate(
                    cell="M15",
                    offset_x=0.5,
                    offset_y=0.5,
                    confidence=0.9,
                    refined=False
                ),
                screenshot_before=None
            )
        
        async def mock_validate_action(instruction, screenshot, context):
            from src.core.enhanced_types import ValidationResult
            return ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Element visible"
            )
        
        async def mock_analyze_result(instruction, result):
            from src.core.enhanced_types import AIAnalysis
            return AIAnalysis(
                success=False,
                confidence=0.95,
                actual_outcome="Failed to type - element not focusable",
                matches_expected=False,
                ui_changes=[],
                recommendations=["Ensure target is a text input field"],
                anomalies=["Clicked on non-input element"]
            )
        
        with patch.object(action_agent, 'determine_action', new=mock_determine_action), \
             patch.object(action_agent, '_validate_action', new=mock_validate_action), \
             patch.object(action_agent, '_validate_focus_for_typing', new=mock_validate_focus), \
             patch.object(action_agent, '_analyze_result', new=mock_analyze_result):
            
            # Execute action
            result = await action_agent.execute_action(
                type_test_step,
                {"test_plan_name": "Wikipedia Search Test"}
            )
            
            # Verify failure
            assert result.overall_success is False
            assert result.failure_phase == "execution"
            assert "Failed to type text - element not focusable" in result.execution.error_message
            
            # Verify text was NOT typed
            mock_browser_driver.type_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_validation_checks_focusability_for_type_actions(
        self, action_agent, type_test_step, mock_browser_driver
    ):
        """Test validation phase checks if element is focusable for type actions."""
        # Mock validation to fail due to non-focusable element
        async def mock_validate_action(instruction, screenshot, context):
            from src.core.enhanced_types import ValidationResult
            return ValidationResult(
                valid=False,
                confidence=0.95,
                reasoning="Target appears to be a div element, not a text input field. Cannot type into non-focusable elements.",
                concerns=["Element is not an input field", "No text cursor will appear"],
                suggestions=["Find the actual search input element"]
            )
        
        with patch.object(action_agent, '_validate_action', new=mock_validate_action):
            
            # Execute action
            result = await action_agent.execute_action(
                type_test_step,
                {"test_plan_name": "Wikipedia Search Test"}
            )
            
            # Verify validation failed
            assert result.overall_success is False
            assert result.failure_phase == "validation"
            assert not result.validation.valid
            assert "not a text input field" in result.validation.reasoning
            
            # Verify no clicks or typing occurred
            mock_browser_driver.click.assert_not_called()
            mock_browser_driver.type_text.assert_not_called()


class TestClickWithFocus:
    """Test enhanced click functionality."""
    
    @pytest.mark.asyncio
    async def test_click_action_uses_enhanced_focus(
        self, action_agent, mock_browser_driver
    ):
        """Test click actions use enhanced focus handling."""
        click_step = TestStep(
            step_id=uuid4(),
            step_number=1,
            description="Click search button",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click the search button",
                target="search button",
                expected_outcome="Search initiated"
            ),
            dependencies=[],
            optional=False
        )
        
        # Mock methods for click action
        from src.core.types import GridAction
        async def mock_determine_action(screenshot, instruction):
            return GridAction(
                instruction=instruction,
                coordinate=GridCoordinate(
                    cell="P15",
                    offset_x=0.5,
                    offset_y=0.5,
                    confidence=0.9,
                    refined=False
                ),
                screenshot_before=None
            )
        
        async def mock_validate_action(instruction, screenshot, context):
            from src.core.enhanced_types import ValidationResult
            return ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Search button visible"
            )
        
        async def mock_analyze_result(instruction, result):
            from src.core.enhanced_types import AIAnalysis
            return AIAnalysis(
                success=True,
                confidence=0.95,
                actual_outcome="Button clicked",
                matches_expected=True,
                ui_changes=["page navigated"],
                recommendations=[],
                anomalies=[]
            )
        
        with patch.object(action_agent, 'determine_action', new=mock_determine_action), \
             patch.object(action_agent, '_validate_action', new=mock_validate_action), \
             patch.object(action_agent, '_analyze_result', new=mock_analyze_result):
            
            # Execute action
            result = await action_agent.execute_action(
                click_step,
                {"test_plan_name": "Wikipedia Search Test"}
            )
            
            # Verify results
            assert result.overall_success is True
            
            # Verify enhanced click was used
            assert mock_browser_driver.click.call_count == 1
            assert mock_browser_driver.wait.call_count >= 2  # Wait after click