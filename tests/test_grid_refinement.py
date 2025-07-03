"""
Unit tests for grid refinement logic.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.core.types import GridCoordinate
from src.grid.overlay import GridOverlay
from src.grid.refinement import GridRefinement


class TestGridRefinement:
    """Tests for GridRefinement class."""

    @pytest.fixture
    def base_grid(self):
        """Create a base grid for testing."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)
        return grid

    @pytest.fixture
    def refinement(self, base_grid):
        """Create a grid refinement instance."""
        return GridRefinement(base_grid)

    @pytest.fixture
    def test_screenshot(self):
        """Create a test screenshot."""
        img = Image.new("RGB", (1920, 1080), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    def test_initialization(self, refinement, base_grid):
        """Test refinement initialization."""
        assert refinement.base_grid == base_grid
        assert refinement.settings is not None

    def test_should_refine_enabled(self, refinement):
        """Test should_refine when refinement is enabled."""
        # Low confidence, not refined
        coord = GridCoordinate(cell="A1", confidence=0.6, refined=False)
        assert refinement.should_refine(coord) is True

        # High confidence
        coord = GridCoordinate(cell="A1", confidence=0.9, refined=False)
        assert refinement.should_refine(coord) is False

        # Already refined
        coord = GridCoordinate(cell="A1", confidence=0.6, refined=True)
        assert refinement.should_refine(coord) is False

    @patch("src.grid.refinement.get_settings")
    def test_should_refine_disabled(self, mock_settings, base_grid):
        """Test should_refine when refinement is disabled."""
        mock_settings.return_value.grid_refinement_enabled = False
        refinement = GridRefinement(base_grid)

        coord = GridCoordinate(cell="A1", confidence=0.1, refined=False)
        assert refinement.should_refine(coord) is False

    def test_refine_coordinate_high_confidence(self, refinement, test_screenshot):
        """Test that high confidence coordinates are not refined."""
        coord = GridCoordinate(
            cell="C3",
            offset_x=0.5,
            offset_y=0.5,
            confidence=0.9,
            refined=False,
        )

        result = refinement.refine_coordinate(
            test_screenshot,
            coord,
            "test target",
        )

        # Should return original coordinate
        assert result == coord
        assert result.refined is False

    def test_refine_coordinate_low_confidence(self, refinement, test_screenshot):
        """Test refinement of low confidence coordinate."""
        coord = GridCoordinate(
            cell="C3",
            offset_x=0.5,
            offset_y=0.5,
            confidence=0.6,
            refined=False,
        )

        result = refinement.refine_coordinate(
            test_screenshot,
            coord,
            "test target",
        )

        # Should return refined coordinate
        assert result.cell == coord.cell
        assert result.refined is True
        assert result.confidence > coord.confidence
        assert 0.0 <= result.offset_x <= 1.0
        assert 0.0 <= result.offset_y <= 1.0

    @patch("src.grid.refinement.get_settings")
    def test_refine_coordinate_disabled(self, mock_settings, base_grid, test_screenshot):
        """Test refinement when disabled in settings."""
        mock_settings.return_value.grid_refinement_enabled = False
        mock_settings.return_value.grid_confidence_threshold = 0.8
        refinement = GridRefinement(base_grid)

        coord = GridCoordinate(
            cell="C3",
            confidence=0.5,
            refined=False,
        )

        result = refinement.refine_coordinate(
            test_screenshot,
            coord,
            "test target",
        )

        # Should return original coordinate
        assert result == coord

    def test_crop_refinement_region(self, refinement, test_screenshot):
        """Test cropping screenshot to refinement region."""
        cell = "C3"
        cropped_bytes, bounds = refinement.crop_refinement_region(
            test_screenshot,
            cell,
        )

        # Verify bounds
        x, y, width, height = bounds
        assert x == 32   # Column B
        assert y == 18   # Row 2
        assert width == 96   # 3 cells
        assert height == 54  # 3 cells

        # Verify cropped image
        cropped_img = Image.open(io.BytesIO(cropped_bytes))
        assert cropped_img.size == (96, 54)

    def test_create_refinement_overlay(self, refinement):
        """Test creating fine grid overlay on cropped region."""
        # Create a small test image
        img = Image.new("RGB", (90, 90), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        cropped_screenshot = img_bytes.getvalue()

        # Create overlay
        result_bytes = refinement.create_refinement_overlay(
            cropped_screenshot,
            fine_grid_size=9,
        )

        # Verify result
        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (90, 90)
        # Should be different due to grid lines
        assert result_bytes != cropped_screenshot

    def test_calculate_refined_pixel_position(self, refinement):
        """Test calculating absolute pixel position from refinement."""
        coord = GridCoordinate(cell="C3", confidence=0.9)
        refinement_bounds = (100, 200, 90, 90)  # x, y, width, height
        fine_grid_position = (4, 4)  # Center of 9x9 grid

        x, y = refinement.calculate_refined_pixel_position(
            coord,
            refinement_bounds,
            fine_grid_position,
            fine_grid_size=9,
        )

        # Fine grid cell 4,4 is center, each cell is 10x10
        # Center of cell 4,4 is at offset 45,45
        assert x == 145  # 100 + 45
        assert y == 245  # 200 + 45

    def test_refinement_edge_cases(self, refinement):
        """Test refinement at grid edges."""
        coord = GridCoordinate(
            cell="A1",  # Top-left corner
            offset_x=0.1,
            offset_y=0.1,
            confidence=0.5,
        )

        # Create a test screenshot
        img = Image.new("RGB", (1920, 1080), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        screenshot = img_bytes.getvalue()

        result = refinement.refine_coordinate(
            screenshot,
            coord,
            "corner target",
        )

        # Should handle edge case properly
        assert result.cell == "A1"
        assert result.refined is True
        assert 0.0 <= result.offset_x <= 1.0
        assert 0.0 <= result.offset_y <= 1.0