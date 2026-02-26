"""Validation and hallucination detection tests for provider-neutral actions."""

import pytest

from src.core.types import (
    ActionInstruction,
    ActionType,
    CoordinateReference,
    ResolvedAction,
)
from src.error_handling.validation import (
    ActionValidator,
    ConfidenceScorer,
    HallucinationDetector,
    ValidationSeverity,
)


def _resolved_action(confidence: float = 0.9) -> ResolvedAction:
    return ResolvedAction(
        instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click login",
            target="Login button",
            expected_outcome="Login page opens",
        ),
        coordinates=CoordinateReference(
            target_reference="login_button",
            pixel_coordinates=(100, 120),
            confidence=confidence,
        ),
    )


def test_confidence_scorer_returns_reasonable_weighted_score() -> None:
    scorer = ConfidenceScorer()
    score = scorer.calculate_action_confidence(
        _resolved_action(0.8),
        screenshot_analysis={"confidence": 0.75},
        historical_success_rate=0.9,
    )
    assert 0.0 <= score <= 1.0
    assert scorer.get_confidence_level(score) in {"minimum", "low", "medium", "high"}


@pytest.mark.asyncio
async def test_action_validator_reports_low_confidence_coordinate_as_warning() -> None:
    validator = ActionValidator()
    valid, results = await validator.validate_action(
        _resolved_action(0.2),
        {"page_loaded": True, "element_exists": True},
    )

    assert valid is True
    coordinate_result = next(r for r in results if r.rule_name == "coordinate_bounds")
    assert coordinate_result.is_valid is False
    assert coordinate_result.severity == ValidationSeverity.WARNING


def test_hallucination_detector_flags_coordinates_outside_viewport() -> None:
    detector = HallucinationDetector()
    error = detector.detect_hallucinations(
        agent_output="I clicked at coordinates: (5000, 4000)",
        agent_name="ActionAgent",
        viewport_size=(1920, 1080),
    )
    assert error is not None
    assert "invalid coordinates" in str(error).lower()
