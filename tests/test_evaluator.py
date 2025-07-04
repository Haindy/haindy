"""
Tests for the Evaluator Agent.
"""

import base64
import json
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest
from PIL import Image

from src.agents.evaluator import EvaluatorAgent
from src.core.types import EvaluationResult


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.openai_temperature = 0.7
    settings.openai_max_retries = 3
    return settings


@pytest.fixture
def evaluator_agent(mock_settings):
    """Create an EvaluatorAgent instance for testing."""
    # EvaluatorAgent doesn't use get_settings directly, it inherits from BaseAgent
    agent = EvaluatorAgent()
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


class TestEvaluatorAgent:
    """Test cases for EvaluatorAgent."""
    
    @pytest.mark.asyncio
    async def test_evaluate_result_success(self, evaluator_agent, sample_screenshot):
        """Test successful evaluation."""
        # Mock AI response for successful outcome
        mock_response = {
            "content": json.dumps({
                "success": True,
                "confidence": 0.95,
                "actual_outcome": "Login page displayed with username and password fields visible",
                "deviations": [],
                "suggestions": [],
                "detailed_analysis": {
                    "elements_found": ["username field", "password field", "login button"],
                    "text_content": ["Login", "Username", "Password"],
                    "ui_state": "Login form ready for input",
                    "error_indicators": [],
                    "success_indicators": ["Form fields visible and enabled"]
                }
            })
        }
        
        evaluator_agent.call_ai = AsyncMock(return_value=mock_response)
        
        # Execute
        result = await evaluator_agent.evaluate_result(
            sample_screenshot,
            "Login page should be displayed with username and password fields"
        )
        
        # Verify
        assert isinstance(result, EvaluationResult)
        assert result.success is True
        assert result.confidence == 0.95
        assert result.actual_outcome == "Login page displayed with username and password fields visible"
        assert len(result.deviations) == 0
        assert len(result.suggestions) == 0
        assert result.screenshot_analysis is not None
        assert "elements_found" in result.screenshot_analysis
    
    @pytest.mark.asyncio
    async def test_evaluate_result_failure(self, evaluator_agent, sample_screenshot):
        """Test failed evaluation with deviations."""
        # Mock AI response for failed outcome
        mock_response = {
            "content": json.dumps({
                "success": False,
                "confidence": 0.88,
                "actual_outcome": "Error page displayed instead of dashboard",
                "deviations": [
                    "Expected dashboard but found error page",
                    "Error message: 'Invalid credentials'"
                ],
                "suggestions": [
                    "Check if correct credentials were used",
                    "Verify login process completed successfully",
                    "Look for session/authentication issues"
                ],
                "detailed_analysis": {
                    "elements_found": ["error message", "back button"],
                    "text_content": ["Error", "Invalid credentials", "Please try again"],
                    "ui_state": "Error page displayed",
                    "error_indicators": ["Red error text", "Error icon"],
                    "success_indicators": []
                }
            })
        }
        
        evaluator_agent.call_ai = AsyncMock(return_value=mock_response)
        
        # Execute
        result = await evaluator_agent.evaluate_result(
            sample_screenshot,
            "User should be redirected to dashboard after login"
        )
        
        # Verify
        assert result.success is False
        assert result.confidence == 0.88
        assert "Error page displayed" in result.actual_outcome
        assert len(result.deviations) == 2
        assert "Invalid credentials" in result.deviations[1]
        assert len(result.suggestions) == 3
        assert "Check if correct credentials" in result.suggestions[0]
    
    @pytest.mark.asyncio
    async def test_evaluate_result_partial_success(self, evaluator_agent, sample_screenshot):
        """Test partial success evaluation."""
        mock_response = {
            "content": json.dumps({
                "success": True,
                "confidence": 0.7,  # Lower confidence indicates partial success
                "actual_outcome": "Dashboard loaded but some widgets failed to render",
                "deviations": [
                    "Analytics widget showing loading spinner",
                    "Recent activity section is empty"
                ],
                "suggestions": [
                    "Wait for all widgets to fully load",
                    "Check network requests for failed API calls"
                ],
                "detailed_analysis": None
            })
        }
        
        evaluator_agent.call_ai = AsyncMock(return_value=mock_response)
        step_id = uuid4()
        
        result = await evaluator_agent.evaluate_result(
            sample_screenshot,
            "Dashboard should load with all widgets",
            step_id=step_id
        )
        
        assert result.step_id == step_id
        assert result.success is True  # Still marked as success
        assert result.confidence == 0.7  # But with lower confidence
        assert len(result.deviations) == 2
        assert len(result.suggestions) == 2
    
    @pytest.mark.asyncio
    async def test_compare_screenshots(self, evaluator_agent, sample_screenshot):
        """Test screenshot comparison functionality."""
        # Create a slightly different screenshot
        image2 = Image.new('RGB', (1920, 1080), color='lightgray')
        buffer2 = BytesIO()
        image2.save(buffer2, format='PNG')
        screenshot2 = buffer2.getvalue()
        
        mock_response = {
            "content": json.dumps({
                "changes_detected": [
                    "Background color changed from white to gray",
                    "New button appeared in top right",
                    "Text updated from 'Loading' to 'Welcome'"
                ],
                "expected_changes_found": True,
                "unexpected_changes": [
                    "Footer text changed unexpectedly"
                ],
                "confidence": 0.92,
                "assessment": "Most expected changes occurred correctly with one minor unexpected change"
            })
        }
        
        evaluator_agent.call_ai = AsyncMock(return_value=mock_response)
        
        result = await evaluator_agent.compare_screenshots(
            sample_screenshot,
            screenshot2,
            "Page should transition from loading to welcome state"
        )
        
        assert result["expected_changes_found"] is True
        assert len(result["changes_detected"]) == 3
        assert len(result["unexpected_changes"]) == 1
        assert result["confidence"] == 0.92
    
    @pytest.mark.asyncio
    async def test_check_for_errors(self, evaluator_agent, sample_screenshot):
        """Test error detection functionality."""
        mock_response = {
            "content": json.dumps({
                "has_errors": True,
                "error_count": 2,
                "errors": [
                    {
                        "type": "validation",
                        "message": "Email address is required",
                        "location": "Below email input field",
                        "severity": "medium"
                    },
                    {
                        "type": "system",
                        "message": "Connection timeout",
                        "location": "Modal dialog center screen",
                        "severity": "high"
                    }
                ],
                "confidence": 0.95
            })
        }
        
        evaluator_agent.call_ai = AsyncMock(return_value=mock_response)
        
        result = await evaluator_agent.check_for_errors(sample_screenshot)
        
        assert result["has_errors"] is True
        assert result["error_count"] == 2
        assert len(result["errors"]) == 2
        assert result["errors"][0]["type"] == "validation"
        assert result["errors"][1]["severity"] == "high"
        assert result["confidence"] == 0.95
    
    def test_parse_analysis_response_valid(self, evaluator_agent):
        """Test parsing valid analysis response."""
        response = {
            "content": json.dumps({
                "success": True,
                "confidence": 0.9,
                "actual_outcome": "Form submitted successfully",
                "deviations": ["Minor styling difference"],
                "suggestions": [],
                "detailed_analysis": {"test": "data"}
            })
        }
        
        result = evaluator_agent._parse_analysis_response(response)
        
        assert result["success"] is True
        assert result["confidence"] == 0.9
        assert result["actual_outcome"] == "Form submitted successfully"
        assert len(result["deviations"]) == 1
        assert len(result["suggestions"]) == 0
        assert result["detailed_analysis"] == {"test": "data"}
    
    def test_parse_analysis_response_invalid(self, evaluator_agent):
        """Test parsing invalid analysis response."""
        response = {
            "content": "invalid json"
        }
        
        result = evaluator_agent._parse_analysis_response(response)
        
        # Should return default error response
        assert result["success"] is False
        assert result["confidence"] == 0.1
        assert "Error analyzing screenshot" in result["actual_outcome"]
        assert len(result["deviations"]) == 1
        assert "Failed to parse AI response" in result["deviations"][0]
    
    def test_parse_analysis_response_missing_fields(self, evaluator_agent):
        """Test parsing response with missing fields."""
        response = {
            "content": json.dumps({
                "success": True
                # Missing other required fields
            })
        }
        
        result = evaluator_agent._parse_analysis_response(response)
        
        # Should provide defaults for missing fields
        assert result["success"] is True
        assert result["confidence"] == 0.5  # Default
        assert result["actual_outcome"] == "Unable to determine outcome"
        assert result["deviations"] == []
        assert result["suggestions"] == []
    
    def test_parse_analysis_response_invalid_types(self, evaluator_agent):
        """Test parsing response with invalid field types."""
        response = {
            "content": json.dumps({
                "success": "yes",  # Should be boolean
                "confidence": "high",  # Should be float
                "deviations": "none",  # Should be list
                "suggestions": "none"  # Should be list
            })
        }
        
        result = evaluator_agent._parse_analysis_response(response)
        
        # Should handle type conversions
        assert isinstance(result["success"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["deviations"], list)
        assert isinstance(result["suggestions"], list)
    
    def test_build_analysis_prompt(self, evaluator_agent):
        """Test building analysis prompt."""
        expected_outcome = "User should see success message"
        prompt = evaluator_agent._build_analysis_prompt(expected_outcome)
        
        assert "User should see success message" in prompt
        assert "JSON format" in prompt
        assert "success" in prompt
        assert "confidence" in prompt
        assert "actual_outcome" in prompt
        assert "deviations" in prompt
        assert "suggestions" in prompt
    
    @pytest.mark.asyncio
    async def test_error_detection_no_errors(self, evaluator_agent, sample_screenshot):
        """Test error detection when no errors present."""
        mock_response = {
            "content": json.dumps({
                "has_errors": False,
                "error_count": 0,
                "errors": [],
                "confidence": 0.98
            })
        }
        
        evaluator_agent.call_ai = AsyncMock(return_value=mock_response)
        
        result = await evaluator_agent.check_for_errors(sample_screenshot)
        
        assert result["has_errors"] is False
        assert result["error_count"] == 0
        assert len(result["errors"]) == 0
        assert result["confidence"] == 0.98
    
    @pytest.mark.asyncio
    async def test_confidence_clamping(self, evaluator_agent, sample_screenshot):
        """Test that confidence values are clamped to valid range."""
        mock_response = {
            "content": json.dumps({
                "success": True,
                "confidence": 1.5,  # Out of range
                "actual_outcome": "Test",
                "deviations": [],
                "suggestions": []
            })
        }
        
        evaluator_agent.call_ai = AsyncMock(return_value=mock_response)
        
        result = await evaluator_agent.evaluate_result(
            sample_screenshot,
            "Test outcome"
        )
        
        # Confidence should be clamped to 1.0
        assert result.confidence == 1.0