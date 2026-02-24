"""Provider-neutral coordinate model tests."""

import pytest
from pydantic import ValidationError

from src.core.enhanced_types import CoordinateResult
from src.core.types import CoordinateReference


def test_coordinate_reference_defaults() -> None:
    ref = CoordinateReference(target_reference="search_input", pixel_coordinates=(100, 200), confidence=0.9)
    assert ref.target_reference == "search_input"
    assert ref.pixel_coordinates == (100, 200)
    assert ref.relative_x == 0.5
    assert ref.relative_y == 0.5
    assert ref.adjusted is False


def test_coordinate_reference_rejects_out_of_bounds_relative_value() -> None:
    with pytest.raises(ValidationError):
        CoordinateReference(relative_x=1.2)


def test_coordinate_result_requires_target_and_pixels() -> None:
    result = CoordinateResult(
        target_reference="submit_button",
        pixel_coordinates=(640, 360),
        relative_x=0.4,
        relative_y=0.6,
        confidence=0.82,
        reasoning="Center-right click target",
        adjusted=True,
    )
    assert result.target_reference == "submit_button"
    assert result.pixel_coordinates == (640, 360)
    assert result.adjusted is True
