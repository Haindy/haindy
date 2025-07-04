#!/usr/bin/env python3
"""
Demonstration of the HAINDY grid system.

This script shows how the browser controller and grid system work together.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.browser.controller import BrowserController
from src.config.settings import get_settings
from src.core.types import GridCoordinate


async def main():
    """Run grid system demonstration."""
    settings = get_settings()
    
    print("HAINDY Grid System Demonstration")
    print("=" * 50)
    print(f"Grid Size: {settings.grid_size}x{settings.grid_size}")
    print(f"Refinement: {'Enabled' if settings.grid_refinement_enabled else 'Disabled'}")
    print()

    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    async with BrowserController(headless=True) as controller:
        # Navigate to example page
        print("1. Navigating to test page...")
        test_page = Path(__file__).parent / "test_page.html"
        await controller.navigate(f"file://{test_page.absolute()}")
        await asyncio.sleep(1)

        # Get current state
        state = await controller.get_current_state()
        print(f"   - Page Title: {state['title']}")
        print(f"   - Viewport: {state['viewport']['width']}x{state['viewport']['height']}")
        print(f"   - Cell Size: {state['grid']['cell_size']['width']:.1f}x{state['grid']['cell_size']['height']:.1f}")
        print()

        # Take screenshot without grid
        print("2. Taking screenshot without grid...")
        await controller.screenshot_with_grid(
            save_path=output_dir / "01_no_grid.png",
            show_overlay=False,
        )
        print(f"   - Saved: {output_dir / '01_no_grid.png'}")
        print()

        # Take screenshot with grid
        print("3. Taking screenshot with grid overlay...")
        await controller.screenshot_with_grid(
            save_path=output_dir / "02_with_grid.png",
            show_overlay=True,
        )
        print(f"   - Saved: {output_dir / '02_with_grid.png'}")
        print()

        # Demonstrate clicking at various grid positions
        print("4. Demonstrating grid clicks...")
        
        # Click positions with descriptions
        click_positions = [
            ("M10", 0.5, 0.5, "Center of page (M10)"),
            ("A1", 0.1, 0.1, "Top-left corner (A1)"),
            ("BH30", 0.9, 0.5, "Right side (BH30)"),
        ]

        for cell, offset_x, offset_y, description in click_positions:
            coord = GridCoordinate(
                cell=cell,
                offset_x=offset_x,
                offset_y=offset_y,
                confidence=0.7,  # Medium confidence to trigger refinement
            )
            
            print(f"   - Clicking at {description}")
            result = await controller.click_at_grid(coord)
            
            if result.refined:
                print(f"     ✓ Coordinate refined: confidence {coord.confidence:.2f} → {result.confidence:.2f}")
            else:
                print(f"     - Used original coordinate")
            
            await asyncio.sleep(0.5)
        print()

        # Demonstrate refinement region
        print("5. Capturing refinement region...")
        await controller.screenshot_refinement_region(
            "M15",
            save_path=output_dir / "03_refinement_M15.png",
        )
        print(f"   - Saved refinement region for M15: {output_dir / '03_refinement_M15.png'}")
        print()

        # Navigate to a more complex page
        print("6. Testing on a more complex page...")
        await controller.navigate("https://www.wikipedia.org")
        await asyncio.sleep(2)

        # Take screenshot with grid
        await controller.screenshot_with_grid(
            save_path=output_dir / "04_wikipedia_grid.png",
            show_overlay=True,
        )
        print(f"   - Saved: {output_dir / '04_wikipedia_grid.png'}")
        print()

        print("Demo complete! Check the 'demo_output' directory for screenshots.")


if __name__ == "__main__":
    asyncio.run(main())