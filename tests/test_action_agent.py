"""
Tests for the Action Agent.
"""

import base64
import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from PIL import Image

from src.agents.action_agent import ActionAgent
from src.core.types import ActionInstruction, ActionType, GridCoordinate


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
def action_agent(mock_settings):
    """Create an ActionAgent instance for testing."""
    with patch("src.agents.action_agent.get_settings", return_value=mock_settings):
        agent = ActionAgent()
        # Mock the OpenAI client
        agent._client = AsyncMock()
        return agent


@pytest.fixture
def sample_screenshot():
    """Create a sample screenshot for testing."""
    # Create a simple test image
    image = Image.new('RGB', (1920, 1080), color='white')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def sample_instruction():
    """Create a sample action instruction."""
    return ActionInstruction(
        action_type=ActionType.CLICK,
        description="Click the login button",
        target="Login button",
        expected_outcome="Login form is submitted"
    )


class TestActionAgent:
    """Test cases for ActionAgent."""
    
    @pytest.mark.asyncio
    async def test_determine_action_high_confidence(
        self, action_agent, sample_screenshot, sample_instruction
    ):
        """Test action determination with high confidence (no refinement)."""
        # Mock AI response with high confidence
        mock_response = {
            "content": json.dumps({
                "cell": "M23",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.95,
                "reasoning": "Login button clearly visible in cell M23"
            })
        }
        
        action_agent.call_ai = AsyncMock(return_value=mock_response)
        
        # Execute
        result = await action_agent.determine_action(
            sample_screenshot,
            sample_instruction
        )
        
        # Verify
        assert result.instruction == sample_instruction
        assert result.coordinate.cell == "M23"
        assert result.coordinate.offset_x == 0.5
        assert result.coordinate.offset_y == 0.5
        assert result.coordinate.confidence == 0.95
        assert not result.coordinate.refined  # No refinement needed
        
        # Verify AI was called once
        action_agent.call_ai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_determine_action_low_confidence_triggers_refinement(
        self, action_agent, sample_screenshot, sample_instruction
    ):
        """Test that low confidence triggers refinement."""
        # Mock initial AI response with low confidence
        initial_response = {
            "content": json.dumps({
                "cell": "M23",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.6,  # Below threshold
                "reasoning": "Button might be in M23 but not certain"
            })
        }
        
        # Mock refinement response with higher confidence
        refined_response = {
            "content": json.dumps({
                "refined_x": 7,
                "refined_y": 4,
                "confidence": 0.92,
                "reasoning": "Button clearly visible in refined view"
            })
        }
        
        action_agent.call_ai = AsyncMock(
            side_effect=[initial_response, refined_response]
        )
        
        # Execute
        result = await action_agent.determine_action(
            sample_screenshot,
            sample_instruction
        )
        
        # Verify
        assert result.coordinate.cell == "M23"
        # GridRefinement adds 0.25 to confidence, so 0.6 + 0.25 = 0.85
        assert result.coordinate.confidence == 0.85
        assert result.coordinate.refined  # Refinement was applied
        
        # Verify AI was called once (GridRefinement handles its own logic)
        assert action_agent.call_ai.call_count == 1
    
    @pytest.mark.asyncio
    async def test_determine_action_refinement_disabled(
        self, action_agent, sample_screenshot, sample_instruction
    ):
        """Test action determination with refinement disabled."""
        # Disable refinement
        action_agent.refinement_enabled = False
        
        # Mock AI response with low confidence
        mock_response = {
            "content": json.dumps({
                "cell": "B7",
                "offset_x": 0.3,
                "offset_y": 0.7,
                "confidence": 0.5,  # Low confidence
                "reasoning": "Element possibly in B7"
            })
        }
        
        action_agent.call_ai = AsyncMock(return_value=mock_response)
        
        # Execute
        result = await action_agent.determine_action(
            sample_screenshot,
            sample_instruction
        )
        
        # Verify no refinement despite low confidence
        assert result.coordinate.confidence == 0.5
        assert not result.coordinate.refined
        action_agent.call_ai.assert_called_once()
    
    def test_parse_coordinate_response_valid(self, action_agent):
        """Test parsing valid coordinate response."""
        response = {
            "content": json.dumps({
                "cell": "Z45",
                "offset_x": 0.75,
                "offset_y": 0.25,
                "confidence": 0.88,
                "reasoning": "Element found in top-right of cell"
            })
        }
        
        coord = action_agent._parse_coordinate_response(response)
        
        assert coord.cell == "Z45"
        assert coord.offset_x == 0.75
        assert coord.offset_y == 0.25
        assert coord.confidence == 0.88
        assert not coord.refined
    
    def test_parse_coordinate_response_invalid(self, action_agent):
        """Test parsing invalid coordinate response."""
        response = {
            "content": "invalid json"
        }
        
        coord = action_agent._parse_coordinate_response(response)
        
        # Should return default values
        assert coord.cell == "A1"
        assert coord.offset_x == 0.5
        assert coord.offset_y == 0.5
        assert coord.confidence == 0.1  # Low confidence for error
        assert not coord.refined
    
    def test_parse_coordinate_response_out_of_range(self, action_agent):
        """Test parsing response with out-of-range values."""
        response = {
            "content": json.dumps({
                "cell": "AA10",
                "offset_x": 1.5,  # Out of range
                "offset_y": -0.2,  # Out of range
                "confidence": 1.2,  # Out of range
            })
        }
        
        coord = action_agent._parse_coordinate_response(response)
        
        # Values should be clamped to valid range
        assert coord.cell == "AA10"
        assert coord.offset_x == 1.0  # Clamped to max
        assert coord.offset_y == 0.0  # Clamped to min
        assert coord.confidence == 1.0  # Clamped to max
    
    def test_build_analysis_prompt(self, action_agent, sample_instruction):
        """Test building analysis prompt."""
        prompt = action_agent._build_analysis_prompt(sample_instruction)
        
        assert "click" in prompt  # ActionType.CLICK.value is lowercase
        assert "Login button" in prompt
        assert "Click the login button" in prompt
        assert "60x60" in prompt
        assert "JSON format" in prompt
    
    @pytest.mark.asyncio
    async def test_refine_coordinates(self, action_agent, sample_screenshot):
        """Test direct coordinate refinement."""
        initial_coord = GridCoordinate(
            cell="H15",
            offset_x=0.5,
            offset_y=0.5,
            confidence=0.7,
            refined=False
        )
        
        # Mock refinement response
        mock_response = {
            "content": json.dumps({
                "refined_x": 6,
                "refined_y": 3,
                "confidence": 0.95,
                "reasoning": "Precise location found"
            })
        }
        
        action_agent.call_ai = AsyncMock(return_value=mock_response)
        
        # Execute
        refined = await action_agent.refine_coordinates(
            sample_screenshot,
            initial_coord
        )
        
        # Verify
        assert refined.cell == "H15"  # Same cell
        assert refined.refined  # Marked as refined
        assert refined.confidence == 0.95
        # Check offset calculation (6-1)/9 = 5/9 ≈ 0.556
        assert 0.55 < refined.offset_x < 0.56
        assert 0.22 < refined.offset_y < 0.23
    
    def test_create_overlay_image(self, action_agent, sample_screenshot):
        """Test creating overlay image."""
        # Initialize grid first
        action_agent.grid_overlay.initialize(1920, 1080)
        
        # Create overlay
        overlay_bytes = action_agent._create_overlay_image(sample_screenshot)
        
        # Verify it's valid image data
        assert overlay_bytes is not None
        assert len(overlay_bytes) > 0
        
        # Verify we can load it as an image
        overlay_image = Image.open(BytesIO(overlay_bytes))
        assert overlay_image.size == (1920, 1080)
    
    @pytest.mark.asyncio
    async def test_apply_refinement_to_region_error_handling(
        self, action_agent, sample_screenshot
    ):
        """Test refinement error handling."""
        initial_coord = GridCoordinate(
            cell="C10",
            offset_x=0.5,
            offset_y=0.5,
            confidence=0.6,
            refined=False
        )
        
        # Mock AI error
        action_agent.call_ai = AsyncMock(
            return_value={"content": "malformed response"}
        )
        
        # Execute
        result = await action_agent._apply_refinement_to_region(
            sample_screenshot,
            initial_coord
        )
        
        # Should return original coordinates with refined flag
        assert result.cell == "C10"
        assert result.offset_x == 0.5
        assert result.offset_y == 0.5
        assert result.confidence == 0.6
        assert result.refined  # Still marked as refined attempt
