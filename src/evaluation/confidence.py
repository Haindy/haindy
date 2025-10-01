"""
Confidence scoring utilities for test evaluation.

Provides configurable confidence thresholds and scoring mechanisms.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ConfidenceThresholds:
    """Configurable confidence thresholds."""
    very_high: float = 0.95
    high: float = 0.85
    medium: float = 0.70
    low: float = 0.50
    very_low: float = 0.0
    
    def get_level(self, score: float) -> ConfidenceLevel:
        """Get confidence level for a score."""
        if score >= self.very_high:
            return ConfidenceLevel.VERY_HIGH
        elif score >= self.high:
            return ConfidenceLevel.HIGH
        elif score >= self.medium:
            return ConfidenceLevel.MEDIUM
        elif score >= self.low:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class ConfidenceScorer:
    """
    Utilities for scoring confidence in test evaluations.
    
    Provides methods to calculate and interpret confidence scores
    based on various factors.
    """
    
    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        """Initialize with configurable thresholds."""
        self.thresholds = thresholds or ConfidenceThresholds()
    
    def calculate_action_confidence(
        self,
        ai_confidence: float,
        validation_passed: bool,
        execution_success: bool,
        has_errors: bool = False
    ) -> Tuple[float, ConfidenceLevel]:
        """
        Calculate overall confidence for an action result.
        
        Args:
            ai_confidence: AI's reported confidence (0-1)
            validation_passed: Whether pre-validation passed
            execution_success: Whether execution succeeded
            has_errors: Whether errors were detected
            
        Returns:
            Tuple of (confidence_score, confidence_level)
        """
        # Start with AI confidence
        confidence = ai_confidence
        
        # Adjust based on validation
        if not validation_passed:
            confidence *= 0.5  # Major penalty for failed validation
            
        # Adjust based on execution
        if not execution_success:
            confidence *= 0.7  # Penalty for failed execution
            
        # Adjust based on errors
        if has_errors:
            confidence *= 0.8  # Penalty for detected errors
            
        # Ensure in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        level = self.thresholds.get_level(confidence)
        return confidence, level
    
    def calculate_visual_match_confidence(
        self,
        expected_elements: List[str],
        found_elements: List[str],
        unexpected_elements: List[str]
    ) -> float:
        """
        Calculate confidence based on visual element matching.
        
        Args:
            expected_elements: Elements we expected to find
            found_elements: Elements actually found
            unexpected_elements: Unexpected elements found
            
        Returns:
            Confidence score (0-1)
        """
        if not expected_elements:
            # No expectations, so full confidence if no unexpected
            return 0.9 if not unexpected_elements else 0.7
            
        # Calculate match ratio
        matches = sum(1 for elem in expected_elements if elem in found_elements)
        match_ratio = matches / len(expected_elements)
        
        # Penalize for unexpected elements
        penalty = len(unexpected_elements) * 0.05
        
        confidence = match_ratio - penalty
        return max(0.0, min(1.0, confidence))
    
    def aggregate_step_confidence(
        self,
        action_confidences: List[float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Aggregate confidence scores from multiple actions in a step.
        
        Args:
            action_confidences: List of confidence scores
            
        Returns:
            Tuple of (overall_confidence, statistics)
        """
        if not action_confidences:
            return 0.0, {"min": 0.0, "max": 0.0, "avg": 0.0}
            
        # Calculate statistics
        min_conf = min(action_confidences)
        max_conf = max(action_confidences)
        avg_conf = sum(action_confidences) / len(action_confidences)
        
        # Overall confidence is weighted average with penalty for low scores
        if min_conf < self.thresholds.low:
            # If any action has very low confidence, penalize overall
            overall = min_conf * 0.5 + avg_conf * 0.5
        else:
            # Otherwise use average
            overall = avg_conf
            
        return overall, {
            "min": min_conf,
            "max": max_conf,
            "avg": avg_conf,
            "count": len(action_confidences)
        }
    
    def should_retry_action(
        self,
        confidence: float,
        attempt_number: int,
        max_retries: int = 3
    ) -> bool:
        """
        Determine if an action should be retried based on confidence.
        
        Args:
            confidence: Current confidence score
            attempt_number: Which attempt this is (1-based)
            max_retries: Maximum retry attempts
            
        Returns:
            Whether to retry
        """
        if attempt_number >= max_retries:
            return False
            
        # Retry if confidence is low but not very low
        level = self.thresholds.get_level(confidence)
        return level in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]
    
    def get_confidence_interpretation(
        self,
        score: float,
        level: ConfidenceLevel
    ) -> str:
        """
        Get human-readable interpretation of confidence.
        
        Args:
            score: Confidence score
            level: Confidence level
            
        Returns:
            Human-readable interpretation
        """
        interpretations = {
            ConfidenceLevel.VERY_HIGH: f"Very high confidence ({score:.0%}) - Result is almost certainly correct",
            ConfidenceLevel.HIGH: f"High confidence ({score:.0%}) - Result is likely correct",
            ConfidenceLevel.MEDIUM: f"Medium confidence ({score:.0%}) - Result may be correct but verify",
            ConfidenceLevel.LOW: f"Low confidence ({score:.0%}) - Result is questionable",
            ConfidenceLevel.VERY_LOW: f"Very low confidence ({score:.0%}) - Result is likely incorrect"
        }
        return interpretations.get(level, f"Unknown confidence level ({score:.0%})")