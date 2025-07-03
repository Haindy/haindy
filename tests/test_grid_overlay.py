"""
Unit tests for grid overlay system.
"""

import io
from PIL import Image
import pytest

from src.core.types import GridCoordinate
from src.grid.overlay import GridOverlay


class TestGridOverlay:
    """Tests for GridOverlay class."""

    def test_initialization(self):
        """Test grid overlay initialization."""
        grid = GridOverlay(grid_size=60)
        assert grid.grid_size == 60
        assert grid.viewport_width == 0
        assert grid.viewport_height == 0

    def test_initialize_dimensions(self):
        """Test initializing grid with viewport dimensions."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        assert grid.viewport_width == 1920
        assert grid.viewport_height == 1080
        assert grid.cell_width == 32.0  # 1920 / 60
        assert grid.cell_height == 18.0  # 1080 / 60

    def test_coordinate_to_pixels_center(self):
        """Test converting grid coordinate to pixels (center of cell)."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        coord = GridCoordinate(
            cell="A1",
            offset_x=0.5,
            offset_y=0.5,
            confidence=0.9,
        )

        x, y = grid.coordinate_to_pixels(coord)
        assert x == 16  # Center of first cell (32 * 0.5)
        assert y == 9   # Center of first cell (18 * 0.5)

    def test_coordinate_to_pixels_with_offset(self):
        """Test converting grid coordinate with custom offset."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        coord = GridCoordinate(
            cell="B2",  # Second column, second row
            offset_x=0.75,
            offset_y=0.25,
            confidence=0.9,
        )

        x, y = grid.coordinate_to_pixels(coord)
        # B2 starts at (32, 18), with offset (0.75, 0.25)
        assert x == 32 + int(32 * 0.75)  # 32 + 24 = 56
        assert y == 18 + int(18 * 0.25)  # 18 + 4 = 22

    def test_coordinate_to_pixels_not_initialized(self):
        """Test error when grid not initialized."""
        grid = GridOverlay()
        coord = GridCoordinate(cell="A1", confidence=0.9)

        with pytest.raises(RuntimeError, match="Grid not initialized"):
            grid.coordinate_to_pixels(coord)

    def test_get_cell_bounds(self):
        """Test getting pixel bounds of a cell."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        x, y, width, height = grid.get_cell_bounds("C3")
        # C3 is column 2 (0-indexed), row 2
        assert x == 64   # 2 * 32
        assert y == 36   # 2 * 18
        assert width == 32
        assert height == 18

    def test_get_cell_from_pixel(self):
        """Test getting cell from pixel coordinates."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        # Test various pixel positions
        assert grid.get_cell_from_pixel(0, 0) == "A1"
        assert grid.get_cell_from_pixel(32, 18) == "B2"
        assert grid.get_cell_from_pixel(100, 100) == "D6"  # 100/32=3.1, 100/18=5.5

    def test_cell_naming(self):
        """Test cell naming convention."""
        grid = GridOverlay(grid_size=60)

        # Test single letter columns
        assert grid._get_cell_name(0, 0) == "A1"
        assert grid._get_cell_name(25, 0) == "Z1"

        # Test double letter columns
        assert grid._get_cell_name(26, 0) == "AA1"
        assert grid._get_cell_name(27, 0) == "AB1"
        assert grid._get_cell_name(51, 0) == "AZ1"

    def test_parse_cell_identifier(self):
        """Test parsing cell identifiers."""
        grid = GridOverlay(grid_size=60)

        # Valid cells
        assert grid._parse_cell_identifier("A1") == (0, 0)
        assert grid._parse_cell_identifier("B2") == (1, 1)
        assert grid._parse_cell_identifier("Z10") == (25, 9)
        assert grid._parse_cell_identifier("AA1") == (26, 0)

        # Invalid cells
        with pytest.raises(ValueError, match="Invalid cell identifier"):
            grid._parse_cell_identifier("123")

        with pytest.raises(ValueError, match="Invalid cell identifier"):
            grid._parse_cell_identifier("ABC")

        with pytest.raises(ValueError, match="out of grid bounds"):
            grid._parse_cell_identifier("ZZ99")  # Beyond 60x60

    def test_get_neighboring_cells(self):
        """Test getting neighboring cells."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        # Center cell
        neighbors = grid.get_neighboring_cells("C3")
        expected = ["B2", "C2", "D2", "B3", "D3", "B4", "C4", "D4"]
        assert sorted(neighbors) == sorted(expected)

        # Corner cell
        neighbors = grid.get_neighboring_cells("A1")
        expected = ["B1", "A2", "B2"]
        assert sorted(neighbors) == sorted(expected)

        # Edge cell
        neighbors = grid.get_neighboring_cells("A5")
        expected = ["A4", "B4", "B5", "A6", "B6"]
        assert sorted(neighbors) == sorted(expected)

    def test_get_refinement_region(self):
        """Test getting refinement region bounds."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        # Center cell
        x, y, width, height = grid.get_refinement_region("C3")
        # Should cover B2 to D4 (3x3 cells)
        assert x == 32   # Column B starts at 32
        assert y == 18   # Row 2 starts at 18
        assert width == 96   # 3 cells * 32
        assert height == 54  # 3 cells * 18

        # Corner cell
        x, y, width, height = grid.get_refinement_region("A1")
        # Should cover A1 to B2 (2x2 cells at corner)
        assert x == 0
        assert y == 0
        assert width == 64   # 2 cells * 32
        assert height == 36  # 2 cells * 18

    def test_create_overlay_image(self):
        """Test creating overlay image."""
        grid = GridOverlay(grid_size=10)  # Smaller grid for testing

        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        screenshot = img_bytes.getvalue()

        # Create overlay
        overlay_bytes = grid.create_overlay_image(screenshot)

        # Load result and verify it's valid
        result_img = Image.open(io.BytesIO(overlay_bytes))
        assert result_img.size == (100, 100)
        assert result_img.mode in ["RGB", "RGBA"]

        # The overlay should have added grid lines, so it should be different
        assert overlay_bytes != screenshot

    def test_grid_bounds_clamping(self):
        """Test that coordinates are clamped to viewport bounds."""
        grid = GridOverlay(grid_size=60)
        grid.initialize(1920, 1080)

        # Coordinate that would exceed viewport
        coord = GridCoordinate(
            cell="BH59",  # Last cell
            offset_x=0.99,
            offset_y=0.99,
            confidence=0.9,
        )

        x, y = grid.coordinate_to_pixels(coord)
        assert x <= 1919  # Max valid x coordinate
        assert y <= 1079  # Max valid y coordinate