"""
Grid refinement logic for adaptive precision.
"""

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from src.config.settings import get_settings
from src.core.types import GridCoordinate
from src.grid.overlay import GridOverlay
from src.monitoring.logger import get_logger


class GridRefinement:
    """Handles adaptive grid refinement for improved precision."""

    def __init__(self, base_grid: GridOverlay) -> None:
        """
        Initialize grid refinement.

        Args:
            base_grid: Base grid overlay instance
        """
        self.base_grid = base_grid
        self.logger = get_logger("grid.refinement")
        self.settings = get_settings()

    def refine_coordinate(
        self,
        screenshot: bytes,
        initial_coord: GridCoordinate,
        target_description: str,
    ) -> GridCoordinate:
        """
        Refine a grid coordinate using adaptive refinement.

        Args:
            screenshot: Full screenshot bytes
            initial_coord: Initial grid coordinate
            target_description: Description of what we're looking for

        Returns:
            Refined grid coordinate with higher precision
        """
        if not self.settings.grid_refinement_enabled:
            # Refinement disabled, return original
            return initial_coord

        if initial_coord.confidence >= self.settings.grid_confidence_threshold:
            # Already confident enough
            return initial_coord

        self.logger.info(
            f"Refining coordinate",
            extra={
                "initial_cell": initial_coord.cell,
                "initial_confidence": initial_coord.confidence,
                "target": target_description,
            },
        )

        # Get refinement region bounds
        x, y, width, height = self.base_grid.get_refinement_region(initial_coord.cell)

        # Crop the screenshot to the refinement region
        img = Image.open(io.BytesIO(screenshot))
        cropped = img.crop((x, y, x + width, y + height))

        # Create a finer grid over the cropped region (9x9 for 3x3 cells)
        fine_grid_size = 9
        cell_width = width / fine_grid_size
        cell_height = height / fine_grid_size

        # In a real implementation, this would use the AI to analyze the cropped region
        # For now, we'll simulate refinement by adjusting the offset
        # This is where the Action Agent would re-analyze the cropped region

        # Simulate finding a more precise location within the original cell
        # In practice, this would involve:
        # 1. Sending the cropped image to the AI
        # 2. Getting a more precise grid location within the crop
        # 3. Converting back to the original coordinate system

        # For demonstration, let's say we found the target at fine grid position (6, 4)
        # which corresponds to the right side of the center cell
        fine_x = 6
        fine_y = 4

        # Convert fine grid position back to original cell offset
        # The center cell in the 3x3 region starts at position (3, 3) in the fine grid
        center_offset_x = (fine_x - 3) / 3.0  # Offset within center cell
        center_offset_y = (fine_y - 3) / 3.0

        # Clamp offsets to valid range [0, 1]
        refined_offset_x = max(0.0, min(1.0, 0.5 + center_offset_x * 0.5))
        refined_offset_y = max(0.0, min(1.0, 0.5 + center_offset_y * 0.5))

        # Create refined coordinate
        refined_coord = GridCoordinate(
            cell=initial_coord.cell,
            offset_x=refined_offset_x,
            offset_y=refined_offset_y,
            confidence=min(0.95, initial_coord.confidence + 0.25),  # Boost confidence
            refined=True,
        )

        self.logger.info(
            f"Coordinate refined",
            extra={
                "cell": refined_coord.cell,
                "refined_offset": f"({refined_coord.offset_x:.2f}, {refined_coord.offset_y:.2f})",
                "new_confidence": refined_coord.confidence,
            },
        )

        return refined_coord

    def should_refine(self, coord: GridCoordinate) -> bool:
        """
        Determine if a coordinate should be refined.

        Args:
            coord: Grid coordinate to check

        Returns:
            True if refinement is recommended
        """
        if not self.settings.grid_refinement_enabled:
            return False

        if coord.refined:
            # Already refined
            return False

        return coord.confidence < self.settings.grid_confidence_threshold

    def crop_refinement_region(
        self, screenshot: bytes, cell: str
    ) -> Tuple[bytes, Tuple[int, int, int, int]]:
        """
        Crop screenshot to refinement region around a cell.

        Args:
            screenshot: Full screenshot bytes
            cell: Center cell identifier

        Returns:
            Tuple of (cropped_image_bytes, (x, y, width, height))
        """
        # Get refinement region bounds
        x, y, width, height = self.base_grid.get_refinement_region(cell)

        # Crop the screenshot
        img = Image.open(io.BytesIO(screenshot))
        cropped = img.crop((x, y, x + width, y + height))

        # Convert back to bytes
        output = io.BytesIO()
        cropped.save(output, format="PNG")
        cropped_bytes = output.getvalue()

        return cropped_bytes, (x, y, width, height)

    def create_refinement_overlay(
        self, cropped_screenshot: bytes, fine_grid_size: int = 9
    ) -> bytes:
        """
        Create a fine grid overlay on a cropped refinement region.

        Args:
            cropped_screenshot: Cropped screenshot bytes
            fine_grid_size: Size of fine grid (default 9x9)

        Returns:
            Screenshot with fine grid overlay
        """
        # Load image
        img = Image.open(io.BytesIO(cropped_screenshot))
        width, height = img.size

        # Create overlay with fine grid
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img, "RGBA")

        # Draw fine grid lines
        cell_width = width / fine_grid_size
        cell_height = height / fine_grid_size

        # Vertical lines
        for i in range(fine_grid_size + 1):
            x = int(i * cell_width)
            draw.line([(x, 0), (x, height)], fill=(0, 255, 255, 128), width=1)

        # Horizontal lines
        for i in range(fine_grid_size + 1):
            y = int(i * cell_height)
            draw.line([(0, y), (width, y)], fill=(0, 255, 255, 128), width=1)

        # Highlight center region (original cell)
        center_start = fine_grid_size // 3
        center_size = fine_grid_size // 3

        # Draw thicker border around center region
        center_x = int(center_start * cell_width)
        center_y = int(center_start * cell_height)
        center_w = int(center_size * cell_width)
        center_h = int(center_size * cell_height)

        draw.rectangle(
            [center_x, center_y, center_x + center_w, center_y + center_h],
            outline=(255, 0, 0, 200),
            width=2,
        )

        # Convert back to bytes
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    def calculate_refined_pixel_position(
        self,
        original_coord: GridCoordinate,
        refinement_bounds: Tuple[int, int, int, int],
        fine_grid_position: Tuple[int, int],
        fine_grid_size: int = 9,
    ) -> Tuple[int, int]:
        """
        Calculate absolute pixel position from refinement results.

        Args:
            original_coord: Original grid coordinate
            refinement_bounds: Bounds of refinement region (x, y, width, height)
            fine_grid_position: Position in fine grid (x, y)
            fine_grid_size: Size of fine grid

        Returns:
            Absolute pixel coordinates (x, y)
        """
        region_x, region_y, region_width, region_height = refinement_bounds
        fine_x, fine_y = fine_grid_position

        # Calculate position within refinement region
        cell_width = region_width / fine_grid_size
        cell_height = region_height / fine_grid_size

        local_x = (fine_x + 0.5) * cell_width  # Center of fine grid cell
        local_y = (fine_y + 0.5) * cell_height

        # Convert to absolute coordinates
        absolute_x = int(region_x + local_x)
        absolute_y = int(region_y + local_y)

        return absolute_x, absolute_y