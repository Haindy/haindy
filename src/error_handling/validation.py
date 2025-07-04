"""
Validation and hallucination detection for AI agent outputs.

Provides confidence scoring, action validation, and detection of
AI hallucinations to ensure reliable test execution.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

from src.core.types import ActionType, TestStep, ActionInstruction, GridAction
from .exceptions import ValidationError, HallucinationError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class HallucinationType(Enum):
    """Types of hallucinations that can be detected."""
    PHANTOM_ELEMENT = auto()  # Element doesn't exist
    WRONG_COORDINATES = auto()  # Coordinates outside viewport
    IMPOSSIBLE_ACTION = auto()  # Action not possible in context
    INCONSISTENT_STATE = auto()  # State description doesn't match reality
    FABRICATED_DATA = auto()  # Made up data in response


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    rule_name: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "is_valid": self.is_valid,
            "severity": self.severity.name,
            "message": self.message,
            "rule_name": self.rule_name,
            "details": self.details
        }


@dataclass
class ValidationRule:
    """A single validation rule."""
    name: str
    description: str
    check_function: Callable[[Any], ValidationResult]
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True


class ConfidenceScorer:
    """Calculates confidence scores for AI agent actions."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "minimum": 0.3,
            "low": 0.5,
            "medium": 0.7,
            "high": 0.9
        }
    
    def calculate_action_confidence(
        self,
        action: GridAction,
        screenshot_analysis: Optional[Dict[str, Any]] = None,
        historical_success_rate: Optional[float] = None
    ) -> float:
        """
        Calculate confidence score for an action.
        
        Factors considered:
        - Grid coordinate precision
        - Element visibility/detectability
        - Historical success rate for similar actions
        - Screenshot analysis results
        """
        scores = []
        weights = []
        
        # Grid coordinate confidence from the action's coordinate
        if action.coordinate:
            coord_confidence = action.coordinate.confidence
            scores.append(coord_confidence)
            weights.append(2.0)  # High weight for coordinates
        
        # Screenshot analysis confidence
        if screenshot_analysis:
            visual_confidence = screenshot_analysis.get("confidence", 0.5)
            scores.append(visual_confidence)
            weights.append(1.5)
        
        # Historical success rate
        if historical_success_rate is not None:
            scores.append(historical_success_rate)
            weights.append(1.0)
        
        # Action type confidence (some actions are more reliable)
        action_confidence = self._get_action_type_confidence(action.instruction.action_type)
        scores.append(action_confidence)
        weights.append(0.5)
        
        # Calculate weighted average
        if not scores:
            return 0.5  # Default confidence
        
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return min(1.0, max(0.0, weighted_sum / total_weight))
    
    def _calculate_coordinate_confidence(
        self,
        grid_coords: Dict[str, Any]
    ) -> float:
        """Calculate confidence based on grid coordinate data."""
        confidence = 0.5  # Base confidence
        
        # Check if refinement was applied
        if grid_coords.get("refinement_applied"):
            confidence += 0.2
        
        # Check initial vs final confidence
        initial = grid_coords.get("initial_confidence", 0.5)
        final = grid_coords.get("final_confidence", initial)
        improvement = final - initial
        
        if improvement > 0:
            confidence += min(0.2, improvement)
        
        # Check for sub-cell precision
        if "offset" in str(grid_coords.get("refined_coordinates", "")):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _get_action_type_confidence(self, action_type: ActionType) -> float:
        """Get base confidence for action type."""
        action_confidences = {
            ActionType.NAVIGATE: 0.95,  # Very reliable
            ActionType.CLICK: 0.85,  # Generally reliable
            ActionType.TYPE: 0.90,  # Reliable with correct element
            ActionType.SCROLL: 0.80,
            ActionType.WAIT: 1.0,  # Always succeeds
            ActionType.SCREENSHOT: 1.0,
            ActionType.ASSERT: 0.75  # Depends on assertion type
        }
        return action_confidences.get(action_type, 0.7)
    
    def get_confidence_level(self, score: float) -> str:
        """Get confidence level name from score."""
        if score >= self.thresholds["high"]:
            return "high"
        elif score >= self.thresholds["medium"]:
            return "medium"
        elif score >= self.thresholds["low"]:
            return "low"
        else:
            return "minimum"


class HallucinationDetector:
    """Detects potential hallucinations in AI agent outputs."""
    
    def __init__(self):
        self.hallucination_patterns = {
            HallucinationType.PHANTOM_ELEMENT: [
                r"I can see .* button",
                r"There is a .* element",
                r"I found .* on the page"
            ],
            HallucinationType.WRONG_COORDINATES: [
                r"coordinates?: \(-?\d+, -?\d+\)",  # Negative coordinates
                r"grid position: [A-Z]\d{3,}",  # Grid position too large
            ],
            HallucinationType.FABRICATED_DATA: [
                r"The (price|total|amount) is \$[\d,]+\.?\d*",
                r"I see \d+ items? in the cart",
                r"The form has been submitted successfully"
            ]
        }
        
        self.valid_viewport_ranges = {
            "x": (0, 1920),  # Common max width
            "y": (0, 1080)   # Common max height
        }
    
    def detect_hallucinations(
        self,
        agent_output: str,
        agent_name: str,
        screenshot_elements: Optional[Set[str]] = None,
        viewport_size: Optional[Tuple[int, int]] = None
    ) -> Optional[HallucinationError]:
        """
        Detect potential hallucinations in agent output.
        
        Args:
            agent_output: The text output from the agent
            agent_name: Name of the agent
            screenshot_elements: Set of elements actually visible
            viewport_size: Current viewport dimensions
            
        Returns:
            HallucinationError if hallucination detected, None otherwise
        """
        # Check for pattern-based hallucinations
        for hall_type, patterns in self.hallucination_patterns.items():
            for pattern in patterns:
                if re.search(pattern, agent_output, re.IGNORECASE):
                    # Verify against screenshot if available
                    if hall_type == HallucinationType.PHANTOM_ELEMENT and screenshot_elements:
                        if not self._verify_element_exists(agent_output, screenshot_elements):
                            return HallucinationError(
                                f"Agent claims to see element that doesn't exist",
                                agent_name=agent_name,
                                hallucination_type=hall_type.name,
                                confidence_score=0.8,
                                evidence=[f"Pattern matched: {pattern}", "Element not in screenshot"]
                            )
        
        # Check coordinate validity
        coord_match = re.search(r"coordinates?: \((-?\d+), (-?\d+)\)", agent_output)
        if coord_match and viewport_size:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            if not self._validate_coordinates(x, y, viewport_size):
                return HallucinationError(
                    f"Agent provided invalid coordinates: ({x}, {y})",
                    agent_name=agent_name,
                    hallucination_type=HallucinationType.WRONG_COORDINATES.name,
                    confidence_score=0.95,
                    evidence=[f"Coordinates outside viewport: {viewport_size}"]
                )
        
        return None
    
    def _verify_element_exists(
        self,
        agent_output: str,
        screenshot_elements: Set[str]
    ) -> bool:
        """Verify if claimed element exists in screenshot."""
        # Extract element claims from output
        element_claims = re.findall(
            r"(?:see|found|clicked on|typing in) (?:a |the )?(['\"]?)(.+?)\1 (?:button|link|field|element)",
            agent_output,
            re.IGNORECASE
        )
        
        for _, element_text in element_claims:
            element_lower = element_text.lower()
            # Check if any screenshot element contains the claimed text
            if not any(element_lower in elem.lower() for elem in screenshot_elements):
                return False
        
        return True
    
    def _validate_coordinates(
        self,
        x: int,
        y: int,
        viewport_size: Tuple[int, int]
    ) -> bool:
        """Validate coordinates are within viewport."""
        width, height = viewport_size
        return 0 <= x <= width and 0 <= y <= height


class ActionValidator:
    """Validates AI agent actions before execution."""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.confidence_scorer = ConfidenceScorer()
        self.hallucination_detector = HallucinationDetector()
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Set up default validation rules."""
        self.add_rule(ValidationRule(
            name="valid_action_type",
            description="Check if action type is valid",
            check_function=self._validate_action_type,
            severity=ValidationSeverity.ERROR
        ))
        
        self.add_rule(ValidationRule(
            name="coordinate_bounds",
            description="Check if coordinates are within bounds",
            check_function=self._validate_coordinates,
            severity=ValidationSeverity.ERROR
        ))
        
        self.add_rule(ValidationRule(
            name="element_selector",
            description="Validate element selector format",
            check_function=self._validate_selector,
            severity=ValidationSeverity.WARNING
        ))
        
        self.add_rule(ValidationRule(
            name="action_prerequisites",
            description="Check action prerequisites are met",
            check_function=self._validate_prerequisites,
            severity=ValidationSeverity.ERROR
        ))
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)
    
    async def validate_action(
        self,
        action: GridAction,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate an action before execution.
        
        Args:
            action: The action to validate
            context: Execution context (viewport, page state, etc.)
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                result = rule.check_function({"action": action, "context": context})
                results.append(result)
            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Rule execution failed: {str(e)}",
                    rule_name=rule.name
                ))
        
        # Check for any critical or error severity failures
        has_errors = any(
            not r.is_valid and r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for r in results
        )
        
        return not has_errors, results
    
    def _validate_action_type(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate action type is supported."""
        action: GridAction = data["action"]
        
        if action.instruction.action_type not in ActionType:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid action type: {action.instruction.action_type}",
                rule_name="valid_action_type"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Action type is valid",
            rule_name="valid_action_type"
        )
    
    def _validate_coordinates(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate coordinates are within viewport bounds."""
        action: GridAction = data["action"]
        context: Dict[str, Any] = data["context"]
        
        # Grid coordinates are already validated by the grid system
        # This is a placeholder for additional coordinate validation if needed
        if action.coordinate and action.coordinate.confidence < 0.3:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Low confidence grid coordinate: {action.coordinate.cell}",
                rule_name="coordinate_bounds",
                details={"confidence": action.coordinate.confidence}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Coordinates valid",
            rule_name="coordinate_bounds"
        )
    
    def _validate_selector(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate element selector format."""
        action: GridAction = data["action"]
        
        # In grid-based system, we use target from instruction instead of selector
        if action.instruction.target:
            # Basic target validation
            if len(action.instruction.target.strip()) == 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Empty target description",
                    rule_name="element_selector"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Selector format valid",
            rule_name="element_selector"
        )
    
    def _validate_prerequisites(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate action prerequisites are met."""
        action: GridAction = data["action"]
        context: Dict[str, Any] = data["context"]
        
        # Check page is loaded for non-navigation actions
        if action.instruction.action_type != ActionType.NAVIGATE:
            if not context.get("page_loaded", True):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Page not loaded for action",
                    rule_name="action_prerequisites"
                )
        
        # Check element exists for interaction actions
        if action.instruction.action_type in [ActionType.CLICK, ActionType.TYPE]:
            if not context.get("element_exists", True):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Element not found for {action.instruction.action_type}",
                    rule_name="action_prerequisites"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Prerequisites met",
            rule_name="action_prerequisites"
        )