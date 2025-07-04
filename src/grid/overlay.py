"""
Grid overlay system for visual interaction.
"""

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config.settings import get_settings
from src.core.interfaces import GridSystem
from src.core.types import GridCoordinate


class GridOverlay(GridSystem):
    """Adaptive grid overlay system for browser interaction."""

    def __init__(self, grid_size: Optional[int] = None) -> None:
        """
        Initialize grid overlay.

        Args:
            grid_size: Number of grid cells (NxN), defaults to config
        """
        settings = get_settings()
        self.grid_size = grid_size or settings.grid_size
        
        # Lazy import to avoid circular dependency
        from src.monitoring.logger import get_logger
        self.logger = get_logger("grid.overlay")

        # Grid dimensions (will be set by initialize)
        self.viewport_width = 0
        self.viewport_height = 0
        self.cell_width = 0.0
        self.cell_height = 0.0

        # Grid cell naming (A-Z, AA-AZ, etc. for columns, 1-N for rows)
        self._column_names = self._generate_column_names()

    def initialize(self, width: int, height: int, grid_size: Optional[int] = None) -> None:
        """
        Initialize grid with viewport dimensions.

        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
            grid_size: Optional override for grid size
        """
        if grid_size:
            self.grid_size = grid_size

        self.viewport_width = width
        self.viewport_height = height
        self.cell_width = width / self.grid_size
        self.cell_height = height / self.grid_size

        self.logger.info(
            f"Grid initialized",
            extra={
                "viewport": f"{width}x{height}",
                "grid_size": f"{self.grid_size}x{self.grid_size}",
                "cell_size": f"{self.cell_width:.1f}x{self.cell_height:.1f}",
            },
        )

    def coordinate_to_pixels(self, coord: GridCoordinate) -> Tuple[int, int]:
        """
        Convert grid coordinate to pixel position.

        Args:
            coord: Grid coordinate

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        if self.viewport_width == 0 or self.viewport_height == 0:
            raise RuntimeError("Grid not initialized. Call initialize() first.")

        # Parse cell identifier (e.g., "M23" -> column M, row 23)
        col_idx, row_idx = self._parse_cell_identifier(coord.cell)

        # Calculate base pixel position (top-left of cell)
        base_x = col_idx * self.cell_width
        base_y = row_idx * self.cell_height

        # Apply offset within cell
        pixel_x = int(base_x + (coord.offset_x * self.cell_width))
        pixel_y = int(base_y + (coord.offset_y * self.cell_height))

        # Ensure within viewport bounds
        pixel_x = max(0, min(pixel_x, self.viewport_width - 1))
        pixel_y = max(0, min(pixel_y, self.viewport_height - 1))

        return pixel_x, pixel_y

    def get_cell_bounds(self, cell: str) -> Tuple[int, int, int, int]:
        """
        Get pixel bounds of a grid cell.

        Args:
            cell: Cell identifier (e.g., 'M23')

        Returns:
            Tuple of (x, y, width, height)
        """
        if self.viewport_width == 0 or self.viewport_height == 0:
            raise RuntimeError("Grid not initialized. Call initialize() first.")

        col_idx, row_idx = self._parse_cell_identifier(cell)

        x = int(col_idx * self.cell_width)
        y = int(row_idx * self.cell_height)
        width = int(self.cell_width)
        height = int(self.cell_height)

        return x, y, width, height

    def create_overlay_image(self, screenshot: bytes) -> bytes:
        """
        Create screenshot with grid overlay for debugging.

        Args:
            screenshot: Original screenshot bytes

        Returns:
            Screenshot with grid overlay as bytes
        """
        # Load screenshot
        img = Image.open(io.BytesIO(screenshot))
        width, height = img.size

        # Initialize grid if needed
        if self.viewport_width != width or self.viewport_height != height:
            self.initialize(width, height)

        # Create drawing context
        draw = ImageDraw.Draw(img, "RGBA")

        # Draw grid lines
        # Vertical lines
        for i in range(self.grid_size + 1):
            x = int(i * self.cell_width)
            draw.line([(x, 0), (x, height)], fill=(255, 255, 0, 128), width=1)

        # Horizontal lines
        for i in range(self.grid_size + 1):
            y = int(i * self.cell_height)
            draw.line([(0, y), (width, y)], fill=(255, 255, 0, 128), width=1)

        # Try to load a font for labels (fallback to default if not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Add cell labels (only for cells that fit labels)
        label_frequency = max(1, self.grid_size // 20)  # Show labels every N cells
        for col_idx in range(0, self.grid_size, label_frequency):
            for row_idx in range(0, self.grid_size, label_frequency):
                cell_name = self._get_cell_name(col_idx, row_idx)
                x = int(col_idx * self.cell_width + 2)
                y = int(row_idx * self.cell_height + 2)

                # Draw text with background for visibility
                bbox = draw.textbbox((x, y), cell_name, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 200))
                draw.text((x, y), cell_name, fill=(255, 255, 0, 255), font=font)

        # Convert back to bytes
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    def get_cell_from_pixel(self, x: int, y: int) -> str:
        """
        Get cell identifier from pixel coordinates.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate

        Returns:
            Cell identifier (e.g., 'M23')
        """
        if self.viewport_width == 0 or self.viewport_height == 0:
            raise RuntimeError("Grid not initialized. Call initialize() first.")

        col_idx = int(x / self.cell_width)
        row_idx = int(y / self.cell_height)

        # Clamp to grid bounds
        col_idx = max(0, min(col_idx, self.grid_size - 1))
        row_idx = max(0, min(row_idx, self.grid_size - 1))

        return self._get_cell_name(col_idx, row_idx)

    def get_neighboring_cells(self, cell: str) -> List[str]:
        """
        Get the 8 neighboring cells for refinement.

        Args:
            cell: Center cell identifier

        Returns:
            List of neighboring cell identifiers
        """
        col_idx, row_idx = self._parse_cell_identifier(cell)
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip center cell

                new_col = col_idx + dx
                new_row = row_idx + dy

                # Check bounds
                if 0 <= new_col < self.grid_size and 0 <= new_row < self.grid_size:
                    neighbors.append(self._get_cell_name(new_col, new_row))

        return neighbors

    def get_refinement_region(self, cell: str) -> Tuple[int, int, int, int]:
        """
        Get pixel bounds for a 3x3 refinement region around a cell.

        Args:
            cell: Center cell identifier

        Returns:
            Tuple of (x, y, width, height) for the refinement region
        """
        col_idx, row_idx = self._parse_cell_identifier(cell)

        # Calculate 3x3 region bounds
        start_col = max(0, col_idx - 1)
        start_row = max(0, row_idx - 1)
        end_col = min(self.grid_size, col_idx + 2)
        end_row = min(self.grid_size, row_idx + 2)

        x = int(start_col * self.cell_width)
        y = int(start_row * self.cell_height)
        width = int((end_col - start_col) * self.cell_width)
        height = int((end_row - start_row) * self.cell_height)

        return x, y, width, height

    def _generate_column_names(self) -> List[str]:
        """Generate column names (A-Z, AA-AZ, BA-BZ, etc.)."""
        names = []
        for i in range(self.grid_size * 2):  # Generate more than needed
            if i < 26:
                names.append(chr(ord('A') + i))
            else:
                # Multi-letter columns
                first = (i - 26) // 26
                second = (i - 26) % 26
                names.append(chr(ord('A') + first) + chr(ord('A') + second))
        return names

    def _get_cell_name(self, col_idx: int, row_idx: int) -> str:
        """
        Get cell name from indices.

        Args:
            col_idx: Column index (0-based)
            row_idx: Row index (0-based)

        Returns:
            Cell name (e.g., 'M23')
        """
        col_name = self._column_names[col_idx]
        row_name = str(row_idx + 1)  # 1-based row numbers
        return f"{col_name}{row_name}"

    def _parse_cell_identifier(self, cell: str) -> Tuple[int, int]:
        """
        Parse cell identifier into column and row indices.

        Args:
            cell: Cell identifier (e.g., 'M23')

        Returns:
            Tuple of (column_index, row_index)
        """
        # Split letters and numbers
        i = 0
        while i < len(cell) and cell[i].isalpha():
            i += 1

        if i == 0 or i == len(cell):
            raise ValueError(f"Invalid cell identifier: {cell}")

        col_part = cell[:i]
        row_part = cell[i:]

        # Convert column letters to index
        col_idx = 0
        for char in col_part:
            col_idx = col_idx * 26 + (ord(char.upper()) - ord('A') + 1)
        col_idx -= 1  # Convert to 0-based

        # Convert row number to index
        try:
            row_idx = int(row_part) - 1  # Convert to 0-based
        except ValueError:
            raise ValueError(f"Invalid row number in cell identifier: {cell}")

        if col_idx < 0 or col_idx >= self.grid_size or row_idx < 0 or row_idx >= self.grid_size:
            raise ValueError(f"Cell {cell} is out of grid bounds (0-{self.grid_size - 1})")

        return col_idx, row_idx