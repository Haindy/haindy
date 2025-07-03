"""
Integration tests for browser and grid system.
"""

import asyncio
from pathlib import Path

import pytest

from src.browser.controller import BrowserController
from src.core.types import GridCoordinate


@pytest.mark.integration
class TestBrowserIntegration:
    """Integration tests for browser controller."""

    @pytest.mark.asyncio
    async def test_browser_lifecycle(self):
        """Test basic browser lifecycle."""
        controller = BrowserController(headless=True)

        # Start browser
        await controller.start()
        state = await controller.get_current_state()
        assert state["initialized"] is True

        # Navigate to a page
        await controller.navigate("https://example.com")
        state = await controller.get_current_state()
        assert "example.com" in state["url"]
        assert state["viewport"]["width"] > 0
        assert state["viewport"]["height"] > 0

        # Stop browser
        await controller.stop()
        state = await controller.get_current_state()
        assert state["initialized"] is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using browser controller as context manager."""
        async with BrowserController(headless=True) as controller:
            await controller.navigate("https://example.com")
            state = await controller.get_current_state()
            assert state["initialized"] is True

        # Should be stopped after context
        state = await controller.get_current_state()
        assert state["initialized"] is False

    @pytest.mark.asyncio
    async def test_screenshot_with_grid(self, tmp_path):
        """Test taking screenshot with grid overlay."""
        async with BrowserController(headless=True) as controller:
            await controller.navigate("https://example.com")

            # Take screenshot without grid
            screenshot = await controller.screenshot_with_grid(show_overlay=False)
            assert len(screenshot) > 0

            # Take screenshot with grid
            screenshot_with_grid = await controller.screenshot_with_grid(
                show_overlay=True
            )
            assert len(screenshot_with_grid) > 0
            # Grid overlay should make it larger
            assert len(screenshot_with_grid) >= len(screenshot)

            # Save screenshot
            save_path = tmp_path / "test_screenshot.png"
            await controller.screenshot_with_grid(
                save_path=save_path,
                show_overlay=True,
            )
            assert save_path.exists()

    @pytest.mark.asyncio
    async def test_grid_click(self):
        """Test clicking at grid coordinates."""
        async with BrowserController(headless=True) as controller:
            await controller.navigate("https://example.com")

            # Click at a grid coordinate
            coord = GridCoordinate(
                cell="M15",
                offset_x=0.5,
                offset_y=0.5,
                confidence=0.9,
            )

            result_coord = await controller.click_at_grid(coord, refine=False)
            assert result_coord.cell == "M15"
            assert not result_coord.refined

    @pytest.mark.asyncio
    async def test_refinement_region_screenshot(self, tmp_path):
        """Test taking screenshot of refinement region."""
        async with BrowserController(headless=True) as controller:
            await controller.navigate("https://example.com")

            # Take refinement region screenshot
            save_path = tmp_path / "refinement_C3.png"
            screenshot = await controller.screenshot_refinement_region(
                "C3",
                save_path=save_path,
            )

            assert len(screenshot) > 0
            assert save_path.exists()

    @pytest.mark.asyncio
    async def test_grid_initialization(self):
        """Test grid initialization matches viewport."""
        async with BrowserController(headless=True, grid_size=50) as controller:
            state = await controller.get_current_state()

            assert state["grid"]["size"] == 50
            assert state["grid"]["cell_size"]["width"] > 0
            assert state["grid"]["cell_size"]["height"] > 0

            # Cell size should match viewport divided by grid size
            expected_cell_width = state["viewport"]["width"] / 50
            expected_cell_height = state["viewport"]["height"] / 50
            assert state["grid"]["cell_size"]["width"] == expected_cell_width
            assert state["grid"]["cell_size"]["height"] == expected_cell_height

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multiple_navigation(self):
        """Test navigating to multiple pages."""
        urls = [
            "https://example.com",
            "https://httpbin.org",
            "https://www.google.com",
        ]

        async with BrowserController(headless=True) as controller:
            for url in urls:
                await controller.navigate(url)
                state = await controller.get_current_state()
                assert state["url"].startswith(url.rstrip("/"))
                
                # Small delay between navigations
                await asyncio.sleep(0.5)