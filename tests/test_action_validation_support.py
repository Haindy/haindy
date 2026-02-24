"""Validation support tests for provider-neutral actions."""

from src.core.types import (
    ActionInstruction,
    ActionType,
    CoordinateReference,
    ResolvedAction,
)
from src.error_handling.validation import ConfidenceScorer


def test_confidence_scorer_uses_coordinate_confidence() -> None:
    scorer = ConfidenceScorer()
    action = ResolvedAction(
        instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click search",
            target="Search field",
            expected_outcome="Search field focused",
        ),
        coordinates=CoordinateReference(
            target_reference="search_field",
            pixel_coordinates=(400, 200),
            confidence=0.9,
        ),
    )

    score = scorer.calculate_action_confidence(action)
    assert 0.0 <= score <= 1.0
    assert score >= 0.7
