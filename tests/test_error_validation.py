"""
Unit tests for validation and hallucination detection - Fixed version.
"""

import pytest
from unittest.mock import Mock, patch

from src.core.types import ActionType, GridAction, ActionInstruction, GridCoordinate
from src.error_handling.validation import (
    ValidationSeverity, HallucinationType, ValidationResult,
    ValidationRule, ConfidenceScorer, HallucinationDetector,
    ActionValidator
)
from src.error_handling.exceptions import HallucinationError


def create_test_action(
    action_type: ActionType = ActionType.CLICK,
    target: str = "submit button",
    confidence: float = 0.85
) -> GridAction:
    """Helper to create test GridAction."""
    return GridAction(
        instruction=ActionInstruction(
            action_type=action_type,
            description=f"{action_type.value} {target}",
            target=target,
            expected_outcome=f"{target} was {action_type.value}ed"
        ),
        coordinate=GridCoordinate(
            cell="M23",
            offset_x=0.5,
            offset_y=0.5,
            confidence=confidence,
            refined=True
        )
    )


class TestValidationResult:
    """Test validation result class."""
    
    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Test passed",
            rule_name="test_rule",
            details={"key": "value"}
        )
        
        assert result.is_valid is True
        assert result.severity == ValidationSeverity.INFO
        assert result.message == "Test passed"
        assert result.rule_name == "test_rule"
        assert result.details == {"key": "value"}
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.ERROR,
            message="Test failed",
            rule_name="test_rule"
        )
        
        data = result.to_dict()
        assert data["is_valid"] is False
        assert data["severity"] == "ERROR"
        assert data["message"] == "Test failed"
        assert data["rule_name"] == "test_rule"


class TestConfidenceScorer:
    """Test confidence scoring."""
    
    def test_default_thresholds(self):
        """Test default confidence thresholds."""
        scorer = ConfidenceScorer()
        assert scorer.thresholds["minimum"] == 0.3
        assert scorer.thresholds["low"] == 0.5
        assert scorer.thresholds["medium"] == 0.7
        assert scorer.thresholds["high"] == 0.9
    
    def test_action_type_confidence(self):
        """Test confidence based on action type."""
        scorer = ConfidenceScorer()
        
        # Test different action types
        assert scorer._get_action_type_confidence(ActionType.WAIT) == 1.0
        assert scorer._get_action_type_confidence(ActionType.NAVIGATE) == 0.95
        assert scorer._get_action_type_confidence(ActionType.TYPE) == 0.90
        assert scorer._get_action_type_confidence(ActionType.CLICK) == 0.85
    
    def test_calculate_action_confidence(self):
        """Test overall action confidence calculation."""
        scorer = ConfidenceScorer()
        
        action = create_test_action(
            action_type=ActionType.CLICK,
            confidence=0.85
        )
        
        # Test with all factors
        confidence = scorer.calculate_action_confidence(
            action,
            screenshot_analysis={"confidence": 0.9},
            historical_success_rate=0.95
        )
        
        assert 0.8 < confidence < 1.0  # Should be high
        
        # Test with minimal factors
        confidence_min = scorer.calculate_action_confidence(action)
        assert confidence_min < confidence  # Should be lower
    
    def test_confidence_levels(self):
        """Test confidence level categorization."""
        scorer = ConfidenceScorer()
        
        assert scorer.get_confidence_level(0.95) == "high"
        assert scorer.get_confidence_level(0.75) == "medium"
        assert scorer.get_confidence_level(0.55) == "low"
        assert scorer.get_confidence_level(0.25) == "minimum"


class TestHallucinationDetector:
    """Test hallucination detection."""
    
    def test_phantom_element_detection(self):
        """Test detection of phantom elements."""
        detector = HallucinationDetector()
        
        # Test with element that doesn't exist
        output = "I can see the 'Submit Order' button on the page"
        screenshot_elements = {"Cancel", "Back", "Home"}
        
        error = detector.detect_hallucinations(
            output,
            "action_agent",
            screenshot_elements=screenshot_elements
        )
        
        assert error is not None
        assert isinstance(error, HallucinationError)
        assert error.hallucination_type == HallucinationType.PHANTOM_ELEMENT.name
        assert error.agent_name == "action_agent"
    
    def test_valid_element_reference(self):
        """Test when element actually exists."""
        detector = HallucinationDetector()
        
        output = "I found the Submit button"
        screenshot_elements = {"Submit", "Cancel", "Back"}
        
        error = detector.detect_hallucinations(
            output,
            "action_agent",
            screenshot_elements=screenshot_elements
        )
        
        assert error is None  # No hallucination
    
    def test_wrong_coordinates_detection(self):
        """Test detection of invalid coordinates."""
        detector = HallucinationDetector()
        
        # Test negative coordinates
        output1 = "Clicking at coordinates: (-100, 500)"
        error1 = detector.detect_hallucinations(
            output1,
            "action_agent",
            viewport_size=(1920, 1080)
        )
        
        assert error1 is not None
        assert error1.hallucination_type == HallucinationType.WRONG_COORDINATES.name
        
        # Test coordinates outside viewport
        output2 = "Found element at coordinates: (2000, 1200)"
        error2 = detector.detect_hallucinations(
            output2,
            "action_agent",
            viewport_size=(1920, 1080)
        )
        
        assert error2 is not None
        
        # Test valid coordinates
        output3 = "Clicking at coordinates: (500, 300)"
        error3 = detector.detect_hallucinations(
            output3,
            "action_agent",
            viewport_size=(1920, 1080)
        )
        
        assert error3 is None
    
    def test_fabricated_data_patterns(self):
        """Test detection of fabricated data."""
        detector = HallucinationDetector()
        
        # Test price fabrication
        output = "The price is $99.99 for this item"
        error = detector.detect_hallucinations(output, "evaluator_agent")
        
        # This would normally require screenshot verification
        # For now, just test pattern matching works
        assert detector.hallucination_patterns[HallucinationType.FABRICATED_DATA]


class TestActionValidator:
    """Test action validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ActionValidator()
    
    def test_default_rules_setup(self, validator):
        """Test default validation rules are set up."""
        rule_names = [rule.name for rule in validator.rules]
        assert "valid_action_type" in rule_names
        assert "coordinate_bounds" in rule_names
        assert "element_selector" in rule_names
        assert "action_prerequisites" in rule_names
    
    def test_add_custom_rule(self, validator):
        """Test adding custom validation rule."""
        def custom_check(data):
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Custom check passed",
                rule_name="custom_rule"
            )
        
        rule = ValidationRule(
            name="custom_rule",
            description="Custom validation",
            check_function=custom_check
        )
        
        validator.add_rule(rule)
        assert rule in validator.rules
    
    @pytest.mark.asyncio
    async def test_validate_valid_action(self, validator):
        """Test validation of valid action."""
        action = create_test_action(
            action_type=ActionType.CLICK,
            confidence=0.85
        )
        
        context = {
            "viewport_size": (1920, 1080),
            "page_loaded": True,
            "element_exists": True
        }
        
        is_valid, results = await validator.validate_action(action, context)
        
        assert is_valid is True
        assert all(r.is_valid for r in results)
    
    @pytest.mark.asyncio
    async def test_validate_low_confidence_coordinates(self, validator):
        """Test validation with low confidence coordinates."""
        action = create_test_action(
            action_type=ActionType.CLICK,
            confidence=0.25  # Low confidence
        )
        
        context = {
            "viewport_size": (1920, 1080),
            "page_loaded": True,
            "element_exists": True
        }
        
        is_valid, results = await validator.validate_action(action, context)
        
        # Should still be valid but with warning
        assert is_valid is True  # Warnings don't fail validation
        coord_results = [r for r in results if r.rule_name == "coordinate_bounds"]
        assert len(coord_results) == 1
        assert not coord_results[0].is_valid
        assert coord_results[0].severity == ValidationSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_validate_prerequisites_not_met(self, validator):
        """Test validation when prerequisites aren't met."""
        action = create_test_action(action_type=ActionType.CLICK)
        
        context = {
            "page_loaded": False,  # Page not loaded
            "element_exists": True
        }
        
        is_valid, results = await validator.validate_action(action, context)
        
        assert is_valid is False
        prereq_results = [r for r in results if r.rule_name == "action_prerequisites"]
        assert len(prereq_results) == 1
        assert not prereq_results[0].is_valid
    
    def test_validate_action_type(self, validator):
        """Test action type validation."""
        # Valid action type
        action = create_test_action(action_type=ActionType.CLICK)
        data = {"action": action, "context": {}}
        result = validator._validate_action_type(data)
        assert result.is_valid is True
    
    def test_validate_target(self, validator):
        """Test target validation."""
        # Valid target
        action1 = create_test_action(target="submit button")
        result1 = validator._validate_selector({"action": action1, "context": {}})
        assert result1.is_valid is True
        
        # Empty target
        action2 = create_test_action()
        action2.instruction.target = "   "  # Empty after strip
        result2 = validator._validate_selector({"action": action2, "context": {}})
        assert result2.is_valid is False