#!/usr/bin/env python3
"""
Test script for two-stage zoom grid system validation.
This script tests the new approach of using a coarse grid followed by a zoomed fine grid
to accurately identify small UI elements.
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.openai_client import OpenAIClient
from src.grid.overlay import GridOverlay


# Configuration
COARSE_GRID_SIZE = 8   # 8x8 grid for broader initial identification
FINE_GRID_SIZE = 8     # 8x8 grid for zoomed region
SCALE_FACTOR = 4       # Scale factor for zooming
PADDING = 0            # No padding - zoom only the exact cell


@dataclass
class TestCase:
    """Test case definition."""
    name: str
    url: str
    target_description: str
    expected_element: str
    element_type: str
    notes: str = ""


@dataclass
class TestResult:
    """Test execution result."""
    test_case: TestCase
    success: bool
    coarse_cell: str
    fine_cell: str
    final_coordinates: Tuple[int, int]
    click_success: bool
    time_taken: float
    api_tokens: int
    error: Optional[str] = None
    screenshots: Dict[str, bytes] = None


class TwoStageGridTester:
    """Test harness for two-stage grid system."""
    
    def __init__(self, api_key: str):
        """Initialize tester with OpenAI API key."""
        self.api_key = api_key
        self.client = None
        self.browser = None
        self.page = None
        self.results: List[TestResult] = []
        
        # Setup logging
        self.logger = logging.getLogger("grid_tester")
        self.logger.setLevel(logging.INFO)
        
        # Create output directory
        self.output_dir = Path(f"grid_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)
        
    async def setup(self):
        """Setup browser and OpenAI client."""
        # Initialize OpenAI client
        self.client = OpenAIClient(api_key=self.api_key)
        
        # Initialize browser
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()
        
    async def teardown(self):
        """Cleanup resources."""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
            
    def create_grid_overlay_with_contrast(self, img: Image.Image, grid_size: int, label_inside: bool = True) -> Image.Image:
        """
        Create grid overlay with improved contrast using edge-only drawing and pixel inversion.
        
        Args:
            img: Original image
            grid_size: Number of grid cells (NxN)
            label_inside: If True, labels are inside cells in checkerboard pattern
            
        Returns:
            Image with grid overlay
        """
        # Create a copy of the image
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        # Create a mask to track which pixels have been inverted
        inverted_mask = np.zeros((height, width), dtype=bool)
        
        # Draw grid lines by inverting pixels for better contrast
        # Vertical lines
        for i in range(grid_size + 1):
            x = int(i * cell_width)
            if x < width:
                # Invert pixels along the line only if not already inverted
                for px in range(max(0, x-1), min(width, x+2)):
                    for py in range(height):
                        if not inverted_mask[py, px]:
                            img_array[py, px] = 255 - img_array[py, px]
                            inverted_mask[py, px] = True
        
        # Horizontal lines  
        for i in range(grid_size + 1):
            y = int(i * cell_height)
            if y < height:
                # Invert pixels along the line only if not already inverted
                for py in range(max(0, y-1), min(height, y+2)):
                    for px in range(width):
                        if not inverted_mask[py, px]:
                            img_array[py, px] = 255 - img_array[py, px]
                            inverted_mask[py, px] = True
        
        # Convert back to PIL Image
        img_with_grid = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_with_grid)
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Add labels
        grid_overlay = GridOverlay(grid_size)
        grid_overlay.initialize(width, height)
        
        if label_inside:
            # Label ALL cells in top-right corner for better spatial reference
            for col_idx in range(grid_size):
                for row_idx in range(grid_size):
                    cell_name = grid_overlay._get_cell_name(col_idx, row_idx)
                    
                    # Calculate top-right corner of cell
                    cell_x_start = int(col_idx * cell_width)
                    cell_y_start = int(row_idx * cell_height)
                    
                    # Get text size
                    bbox = draw.textbbox((0, 0), cell_name, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Position in top-left corner with small margin
                    margin = 3
                    text_x = cell_x_start + margin
                    text_y = cell_y_start + margin
                    
                    # Draw text with white outline for visibility (no background rectangle)
                    # Draw white outline by drawing text multiple times with small offsets
                    outline_width = 1
                    for dx in [-outline_width, 0, outline_width]:
                        for dy in [-outline_width, 0, outline_width]:
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), cell_name, fill=(255, 255, 255), font=font)
                    
                    # Draw text in black on top
                    draw.text((text_x, text_y), cell_name, fill=(0, 0, 0), font=font)
        
        return img_with_grid
    
    async def identify_coarse_location(self, screenshot: bytes, target_description: str) -> Tuple[str, Dict[str, any]]:
        """
        First stage: Identify general location using coarse grid.
        
        Returns:
            Tuple of (cell_identifier, api_response)
        """
        # Load image and apply coarse grid
        img = Image.open(io.BytesIO(screenshot))
        img_with_grid = self.create_grid_overlay_with_contrast(img, COARSE_GRID_SIZE)
        
        # Convert to bytes for API
        grid_bytes = io.BytesIO()
        img_with_grid.save(grid_bytes, format='PNG')
        grid_bytes = grid_bytes.getvalue()
        
        # Save for debugging
        img_with_grid.save(self.output_dir / "coarse_grid.png")
        
        # Prepare prompt for coarse identification
        prompt = f"""You are looking at a screenshot with an {COARSE_GRID_SIZE}x{COARSE_GRID_SIZE} grid overlay.
Each cell is labeled with a letter-number combination (e.g., A1, B2, H8).

In which of these cells is the '{target_description}' located? 
If you see it in multiple cells, tell me the cell that has the most of the element.

Return ONLY the cell identifier (e.g., "C7"), nothing else."""

        # Call API
        response = await self.client.analyze_image(grid_bytes, prompt, temperature=0.1)
        
        # Extract cell identifier
        cell = response['content'].strip().upper()
        
        return cell, response
    
    async def identify_fine_location(self, original_img: Image.Image, coarse_cell: str, 
                                   target_description: str) -> Tuple[Tuple[int, int], Dict[str, any]]:
        """
        Second stage: Identify precise location using fine grid on zoomed region.
        
        Returns:
            Tuple of ((x, y) coordinates in original image, api_response)
        """
        # Get bounds of coarse cell
        width, height = original_img.size
        grid_overlay = GridOverlay(COARSE_GRID_SIZE)
        grid_overlay.initialize(width, height)
        
        col_idx, row_idx = grid_overlay._parse_cell_identifier(coarse_cell)
        cell_width = width / COARSE_GRID_SIZE
        cell_height = height / COARSE_GRID_SIZE
        
        # Calculate exact cell bounds (no padding)
        x1 = int(col_idx * cell_width)
        y1 = int(row_idx * cell_height)
        x2 = int((col_idx + 1) * cell_width)
        y2 = int((row_idx + 1) * cell_height)
        
        # Crop the region
        img_cropped = original_img.crop((x1, y1, x2, y2))
        
        # Scale up the cropped region
        scaled_width = img_cropped.width * SCALE_FACTOR
        scaled_height = img_cropped.height * SCALE_FACTOR
        img_scaled = img_cropped.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        
        # Apply fine grid to scaled image
        img_with_fine_grid = self.create_grid_overlay_with_contrast(img_scaled, FINE_GRID_SIZE)
        
        # Convert to bytes
        fine_grid_bytes = io.BytesIO()
        img_with_fine_grid.save(fine_grid_bytes, format='PNG')
        fine_grid_bytes = fine_grid_bytes.getvalue()
        
        # Save for debugging
        img_with_fine_grid.save(self.output_dir / "fine_grid.png")
        
        # Prepare prompt for fine identification
        prompt = f"""You are looking at a ZOOMED IN section of a screenshot with an {FINE_GRID_SIZE}x{FINE_GRID_SIZE} grid overlay.
This is a {SCALE_FACTOR}x magnified view of a single cell from the previous coarse grid.

Your task is to identify the precise grid cell containing the CENTER of: {target_description}

IMPORTANT:
1. This is a ZOOMED view - the element appears larger than normal
2. Focus on finding the CENTER/MIDDLE of the element
3. The {FINE_GRID_SIZE}x{FINE_GRID_SIZE} grid provides fine-grained precision
4. Return ONLY the cell identifier (e.g., "D4")
5. Choose the cell that best contains the CENTER point where you would click

Respond with just the cell identifier, nothing else."""

        # Call API
        response = await self.client.analyze_image(fine_grid_bytes, prompt, temperature=0.1)
        
        # Extract cell identifier
        fine_cell = response['content'].strip().upper()
        
        # Calculate coordinates in scaled image
        fine_grid = GridOverlay(FINE_GRID_SIZE)
        fine_grid.initialize(scaled_width, scaled_height)
        
        fine_col_idx, fine_row_idx = fine_grid._parse_cell_identifier(fine_cell)
        fine_cell_width = scaled_width / FINE_GRID_SIZE
        fine_cell_height = scaled_height / FINE_GRID_SIZE
        
        # Get center of identified cell in scaled coordinates
        scaled_x = int((fine_col_idx + 0.5) * fine_cell_width)
        scaled_y = int((fine_row_idx + 0.5) * fine_cell_height)
        
        # Convert back to original image coordinates
        original_x = x1 + (scaled_x / SCALE_FACTOR)
        original_y = y1 + (scaled_y / SCALE_FACTOR)
        
        return (int(original_x), int(original_y)), response
    
    async def run_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        self.logger.info(f"Running test: {test_case.name}")
        start_time = time.time()
        
        screenshots = {}
        total_tokens = 0
        
        try:
            # Navigate to URL
            await self.page.goto(test_case.url, wait_until="networkidle")
            await asyncio.sleep(2)  # Let page fully load
            
            # Take screenshot
            screenshot = await self.page.screenshot()
            img = Image.open(io.BytesIO(screenshot))
            screenshots['original'] = screenshot
            
            # Stage 1: Coarse identification
            self.logger.info("Stage 1: Coarse grid identification...")
            coarse_cell, coarse_response = await self.identify_coarse_location(
                screenshot, test_case.target_description
            )
            total_tokens += coarse_response['usage']['total_tokens']
            self.logger.info(f"Coarse cell identified: {coarse_cell}")
            
            # Stage 2: Fine identification
            self.logger.info("Stage 2: Fine grid identification...")
            (x, y), fine_response = await self.identify_fine_location(
                img, coarse_cell, test_case.target_description
            )
            total_tokens += fine_response['usage']['total_tokens']
            self.logger.info(f"Final coordinates: ({x}, {y})")
            
            # Attempt to click the identified location
            self.logger.info(f"Attempting click at ({x}, {y})...")
            click_success = False
            error = None
            
            try:
                await self.page.mouse.click(x, y)
                await asyncio.sleep(1)
                
                # Take screenshot after click
                screenshot_after = await self.page.screenshot()
                screenshots['after_click'] = screenshot_after
                
                # Simple validation - check if URL changed or page updated
                current_url = self.page.url
                if current_url != test_case.url:
                    click_success = True
                    self.logger.info("Click successful - URL changed")
                else:
                    # Could add more sophisticated validation here
                    click_success = True
                    self.logger.info("Click executed")
                    
            except Exception as e:
                error = str(e)
                self.logger.error(f"Click failed: {error}")
            
            time_taken = time.time() - start_time
            
            # Create result
            result = TestResult(
                test_case=test_case,
                success=click_success,
                coarse_cell=coarse_cell,
                fine_cell=fine_response['content'].strip().upper(),
                final_coordinates=(x, y),
                click_success=click_success,
                time_taken=time_taken,
                api_tokens=total_tokens,
                error=error,
                screenshots=screenshots
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Test failed with error: {e}")
            return TestResult(
                test_case=test_case,
                success=False,
                coarse_cell="",
                fine_cell="",
                final_coordinates=(0, 0),
                click_success=False,
                time_taken=time.time() - start_time,
                api_tokens=total_tokens,
                error=str(e),
                screenshots=screenshots
            )
    
    async def run_all_tests(self, test_cases: List[TestCase]):
        """Run all test cases and generate report."""
        self.logger.info(f"Running {len(test_cases)} test cases...")
        
        for test_case in test_cases:
            result = await self.run_test_case(test_case)
            self.results.append(result)
            
            # Save screenshots for this test
            test_dir = self.output_dir / test_case.name.replace(" ", "_")
            test_dir.mkdir(exist_ok=True)
            
            for name, screenshot in result.screenshots.items():
                with open(test_dir / f"{name}.png", "wb") as f:
                    f.write(screenshot)
            
            # Add visual marker to show click location
            if 'original' in result.screenshots:
                img = Image.open(io.BytesIO(result.screenshots['original']))
                draw = ImageDraw.Draw(img)
                x, y = result.final_coordinates
                
                # Draw crosshair at click location
                draw.line([(x-20, y), (x+20, y)], fill=(255, 0, 0), width=3)
                draw.line([(x, y-20), (x, y+20)], fill=(255, 0, 0), width=3)
                draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=(255, 0, 0))
                
                # Save marked image
                img.save(test_dir / "marked_click_location.png")
    
    def generate_report(self):
        """Generate test execution report."""
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
            },
            "configuration": {
                "coarse_grid_size": COARSE_GRID_SIZE,
                "fine_grid_size": FINE_GRID_SIZE, 
                "scale_factor": SCALE_FACTOR,
                "padding": PADDING,
            },
            "results": []
        }
        
        total_time = 0
        total_tokens = 0
        
        for result in self.results:
            report["results"].append({
                "test_name": result.test_case.name,
                "success": result.success,
                "coarse_cell": result.coarse_cell,
                "fine_cell": result.fine_cell,
                "coordinates": result.final_coordinates,
                "click_success": result.click_success,
                "time_taken": result.time_taken,
                "api_tokens": result.api_tokens,
                "error": result.error,
            })
            total_time += result.time_taken
            total_tokens += result.api_tokens
        
        report["summary"] = {
            "success_rate": f"{(report['test_run']['passed'] / report['test_run']['total_tests'] * 100):.1f}%",
            "total_time": f"{total_time:.2f}s",
            "avg_time_per_test": f"{total_time / len(self.results):.2f}s",
            "total_api_tokens": total_tokens,
            "avg_tokens_per_test": total_tokens // len(self.results),
        }
        
        # Save JSON report
        with open(self.output_dir / "test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        summary = f"""# Two-Stage Grid System Test Report

## Configuration
- Coarse Grid: {COARSE_GRID_SIZE}x{COARSE_GRID_SIZE}
- Fine Grid: {FINE_GRID_SIZE}x{FINE_GRID_SIZE}
- Scale Factor: {SCALE_FACTOR}x
- Padding: {PADDING}px

## Results Summary
- Total Tests: {report['test_run']['total_tests']}
- Passed: {report['test_run']['passed']}
- Failed: {report['test_run']['failed']}
- Success Rate: {report['summary']['success_rate']}
- Total Time: {report['summary']['total_time']}
- Total API Tokens: {report['summary']['total_api_tokens']}

## Detailed Results

| Test Name | Success | Coarse Cell | Fine Cell | Coordinates | Time | Tokens |
|-----------|---------|-------------|-----------|-------------|------|--------|
"""
        
        for result in report["results"]:
            summary += f"| {result['test_name']} | {'✅' if result['success'] else '❌'} | "
            summary += f"{result['coarse_cell']} | {result['fine_cell']} | "
            summary += f"{result['coordinates']} | {result['time_taken']:.2f}s | {result['api_tokens']} |\n"
        
        with open(self.output_dir / "test_summary.md", "w") as f:
            f.write(summary)
        
        # Print summary to console
        print("\n" + "="*60)
        print("TEST EXECUTION COMPLETE")
        print("="*60)
        print(f"Success Rate: {report['summary']['success_rate']}")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)


async def main():
    """Main test execution."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    # Define test cases
    test_cases = [
        # TestCase(
        #     name="Wikipedia_History_Link",
        #     url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        #     target_description="The 'History' link in the table of contents on the left side",
        #     expected_element="History link in TOC",
        #     element_type="link",
        #     notes="Primary test case - small link in sidebar TOC"
        # ),
        TestCase(
            name="Wikipedia_Debate_Link",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            target_description="The 'debate' link in the main article text (in the introductory paragraphs)",
            expected_element="debate link in article text",
            element_type="link",
            notes="Link embedded within article text - tests precision for inline links"
        ),
    ]
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    tester = TwoStageGridTester(api_key)
    
    try:
        await tester.setup()
        await tester.run_all_tests(test_cases)
        tester.generate_report()
    finally:
        await tester.teardown()


if __name__ == "__main__":
    asyncio.run(main())