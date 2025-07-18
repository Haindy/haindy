#!/usr/bin/env python3
"""
Test script for two-stage zoom grid system validation.
This script tests the new approach of using a coarse grid followed by a zoomed fine grid
to accurately identify small UI elements.
"""

import asyncio
import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
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
from src.models.gemini_client import GeminiClient
from src.grid.overlay import GridOverlay


# Configuration
COARSE_GRID_SIZE = 7   # 7x7 grid for broader initial identification
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
    expected_coarse_cells: Optional[List[str]] = None  # List of acceptable cells
    expected_fine_cells: Optional[List[str]] = None   # List of acceptable cells


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
    run_number: int = 1


class TwoStageGridTester:
    """Test harness for two-stage grid system."""
    
    def __init__(self, api_key: str, bypass_coarse_validation: bool = False, reasoning_effort: str = "medium", model_type: str = "openai"):
        """Initialize tester with API key.
        
        Args:
            api_key: API key for the selected model
            bypass_coarse_validation: Whether to continue with fine grid even if coarse is wrong
            reasoning_effort: Reasoning effort level (low/medium/high)
            model_type: Which model to use ('openai' or 'gemini')
        """
        self.api_key = api_key
        self.model_type = model_type
        self.client = None
        self.browser = None
        self.page = None
        self.results: List[TestResult] = []
        self.ai_debug_insights: List[Dict[str, str]] = []  # Collect AI's self-reported failures
        self.base_screenshots: Dict[str, bytes] = {}  # Cache screenshots per test case
        self.bypass_coarse_validation = bypass_coarse_validation
        self.reasoning_effort = reasoning_effort
        
        # Setup logging
        self.logger = logging.getLogger("grid_tester")
        self.logger.setLevel(logging.INFO)
        
        # Create output directory
        self.output_dir = Path(f"grid_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)
        
    async def setup(self):
        """Setup browser and AI client."""
        # Initialize AI client based on model type
        if self.model_type == "openai":
            self.client = OpenAIClient(api_key=self.api_key, reasoning_effort=self.reasoning_effort)
            self.logger.info("Using OpenAI model (o4-mini)")
        elif self.model_type == "gemini":
            self.client = GeminiClient(api_key=self.api_key, reasoning_effort=self.reasoning_effort)
            self.logger.info("Using Google Gemini model (2.5 Flash)")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Initialize browser
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page(viewport={'width': 1920, 'height': 1080})
        
    async def teardown(self):
        """Cleanup resources."""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
            
    def save_and_compress_image(self, img: Image.Image, path: Path, quality: int = 80) -> bytes:
        """
        Save image to disk as JPEG and return compressed bytes for API transmission.
        
        Args:
            img: PIL Image to save
            path: Path to save the image
            quality: JPEG quality (1-100, default 90)
            
        Returns:
            Compressed JPEG bytes
        """
        # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
        if img.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Save to disk as JPEG
        img.save(path.with_suffix('.jpg'), 'JPEG', quality=quality, optimize=True)
        
        # Return compressed bytes for API
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG', quality=quality, optimize=True)
        return buffer.getvalue()
    
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
    
    def _build_fine_grid_content(self, coarse_cell: str, target_description: str, 
                                 img_no_grid_bytes: bytes, img_grid_bytes: bytes) -> List[Dict]:
        """Build content for fine grid identification based on model type."""
        if hasattr(self, 'model_type') and self.model_type == 'gemini':
            # For Gemini, send both images with clear explanation
            return [
                {
                    "type": "text", 
                    "text": f"Ok now look at these two images. They are both a 4x zoom of the {coarse_cell} cell you selected before. The first image is the clean screenshot without any grid overlay so you can see the actual content. The second image is the same screenshot but with a finer 8x8 grid overlay. Which cell in this new grid do I need to select to hit the center of the element?\nKeep in mind that I want to hit the center of the element.\nPlease reply just with the cell you have chosen."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(img_no_grid_bytes).decode('utf-8')}",
                        "detail": "high"
                    }
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(img_grid_bytes).decode('utf-8')}",
                        "detail": "high"
                    }
                }
            ]
        else:
            # For OpenAI, send both images
            return [
                {
                    "type": "text", 
                    "text": f"Here are 2 screenshots which are a zoom in of the cell you selected.\nOne without a grid and the other has a grid like the screenshot in the past chat.\nPlease identify where I should click to interact with: {target_description}\n\nUse the grid to tell me what cell I should click for the most success.\n\nRespond with just the cell identifier, nothing else."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(img_no_grid_bytes).decode('utf-8')}",
                        "detail": "high"
                    }
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(img_grid_bytes).decode('utf-8')}",
                        "detail": "high"
                    }
                }
            ]
    
    async def identify_coarse_location(self, screenshot: bytes, target_description: str) -> Tuple[str, Dict[str, any]]:
        """
        First stage: Identify general location using coarse grid.
        
        Returns:
            Tuple of (cell_identifier, api_response)
        """
        # Load image and apply coarse grid
        img = Image.open(io.BytesIO(screenshot))
        img_with_grid = self.create_grid_overlay_with_contrast(img, COARSE_GRID_SIZE)
        
        # Save for debugging and get compressed bytes for API
        filename = "coarse_grid"
        if hasattr(self, 'current_run_number'):
            filename = f"coarse_grid_run{self.current_run_number}"
        
        # Save as JPEG and get compressed bytes
        grid_bytes = self.save_and_compress_image(img_with_grid, self.output_dir / filename)
        
        # Use different prompts for different models
        if hasattr(self, 'model_type') and self.model_type == 'gemini':
            # GEMINI SIMPLE PROMPT
            system_prompt = ""
            user_prompt = f"""What cell do I need to choose in the following screenshot if I were to place my mouse over the {target_description}?\nKeep in mind that I want to hit the center of the element.\nPlease reply just with the cell."""
        else:
            # OPENAI CENTROID-FOCUSED PROMPT - 80% success rate:
            system_prompt = """You are a manual tester helping to test a new grid overlay system. Your task is to identify which grid cell contains the UI element described by the user.

Imagine you need to move your mouse cursor to click on the requested element. First, locate the element visually in the browser screenshot, then determine which grid cell contains the CENTER or CENTROID of that element.

IMPORTANT: 
1. Do not group nearby buttons or elements together. Focus ONLY on the specific element requested.
2. Find the geometric center of THIS specific element only, ignoring any neighboring elements.
3. Do not focus on where the text begins. Instead, identify the geometric center of the clickable area.
4. If an element spans multiple cells, choose the cell containing the element's center point, not where it starts."""
        
            # Prepare user prompt for coarse identification
            user_prompt = f"""This is a browser screenshot with a {COARSE_GRID_SIZE}x{COARSE_GRID_SIZE} grid overlay. Each cell has a label in its top-left corner (like A1, B2, C3, etc.).

I need to find: {target_description}

Which grid cell contains this element?

Respond with ONLY the cell identifier (e.g., "D6"), nothing else."""

        # Call API with system prompt
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(grid_bytes).decode('utf-8')}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        else:
            # For Gemini with no system prompt
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(grid_bytes).decode('utf-8')}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        
        response = await self.client.call(messages=messages, temperature=0.1)
        
        # Extract cell identifier
        cell = response['content'].strip().upper()
        
        return cell, response, messages
    
    async def identify_fine_location(self, original_img: Image.Image, coarse_cell: str, 
                                   target_description: str, coarse_response: Dict[str, any]) -> Tuple[Tuple[int, int], Dict[str, any]]:
        """
        Second stage: Identify precise location using fine grid on zoomed region.
        Uses two-image approach with conversational context.
        
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
        
        # Save scaled image WITHOUT grid
        fine_no_grid_filename = "fine_no_grid"
        if hasattr(self, 'current_run_number'):
            fine_no_grid_filename = f"fine_no_grid_run{self.current_run_number}"
        img_scaled_no_grid_bytes = self.save_and_compress_image(img_scaled, self.output_dir / fine_no_grid_filename)
        
        # Apply fine grid to scaled image
        img_with_fine_grid = self.create_grid_overlay_with_contrast(img_scaled, FINE_GRID_SIZE)
        
        # Save with run number if available
        fine_grid_filename = "fine_grid"
        if hasattr(self, 'current_run_number'):
            fine_grid_filename = f"fine_grid_run{self.current_run_number}"
        fine_grid_bytes = self.save_and_compress_image(img_with_fine_grid, self.output_dir / fine_grid_filename)
        
        # Build conversation context
        messages = []
        
        # Add system message only for non-Gemini models
        if not (hasattr(self, 'model_type') and self.model_type == 'gemini'):
            messages.append({
                "role": "system",
                "content": "You are helping identify precise click locations on UI elements."
            })
        
        # Add conversation messages
        messages.extend([
            {
                "role": "user",
                "content": f"I'm looking for: {target_description}"
            },
            {
                "role": "assistant", 
                "content": f"I found it in cell {coarse_cell} of the coarse grid."
            },
            {
                "role": "user",
                "content": self._build_fine_grid_content(coarse_cell, target_description, img_scaled_no_grid_bytes, fine_grid_bytes)
            }
        ])
        
        # Call API with full conversation context
        response = await self.client.call(messages=messages, temperature=0.1)
        
        # Extract cell identifier from response
        content = response['content']
        # Look for cell pattern (letter + number)
        cell_match = re.search(r'\b([A-H][1-8])\b', content)
        if cell_match:
            fine_cell = cell_match.group(1).upper()
        else:
            # Fallback to stripping and taking first word
            fine_cell = content.strip().split()[0].upper()
        
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
    
    async def debug_ai_failure(self, messages: List[Dict], selected_cell: str, correct_cell: str, 
                              test_name: str) -> Dict[str, str]:
        """
        Ask AI why it failed and what changes would help.
        
        Returns:
            Dictionary with failure reason and suggested improvement
        """
        # Add follow-up message to the conversation
        debug_prompt = f"""You selected cell {selected_cell}, but the correct answer is {correct_cell}.

The target element ({test_name}) is actually located in cell {correct_cell}.

Two questions:
1. Why did you fail to select the right cell? What caused you to choose {selected_cell} instead?
2. What is the most important change necessary to the prompt or instructions that would help you find the correct cell?

Please be specific and concise in your answers."""

        # Extend the conversation with the debug question
        messages.append({
            "role": "user",
            "content": debug_prompt
        })
        
        # Get AI's self-analysis
        debug_response = await self.client.call(messages=messages, temperature=0.3)
        
        insight = {
            "test_name": test_name,
            "selected": selected_cell,
            "correct": correct_cell,
            "ai_explanation": debug_response['content'],
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"AI Debug Insight: {debug_response['content']}")
        
        return insight
    
    def load_existing_screenshots(self, test_dir: str, test_case: TestCase):
        """Load screenshots from a previous test run."""
        test_path = Path(test_dir)
        if not test_path.exists():
            raise ValueError(f"Test directory not found: {test_dir}")
            
        # Load base screenshot
        base_path = test_path / f"{test_case.name.replace(' ', '_')}_base.jpg"
        if base_path.exists():
            with open(base_path, 'rb') as f:
                self.base_screenshots[test_case.name] = f.read()
            self.logger.info(f"Loaded existing base screenshot from: {base_path}")
        else:
            raise ValueError(f"Base screenshot not found: {base_path}")
            
    async def capture_base_screenshot(self, test_case: TestCase) -> bytes:
        """Capture and cache the base screenshot for a test case."""
        if test_case.name not in self.base_screenshots:
            self.logger.info(f"Capturing base screenshot for: {test_case.name}")
            await self.page.goto(test_case.url, wait_until="networkidle")
            await asyncio.sleep(2)  # Let page fully stabilize
            # Zoom to 125%
            await self.page.evaluate("document.body.style.zoom = '125%'")
            await asyncio.sleep(1)  # Let zoom take effect
            screenshot = await self.page.screenshot()
            self.base_screenshots[test_case.name] = screenshot
            
            # Save the base screenshot as JPEG
            img = Image.open(io.BytesIO(screenshot))
            base_filename = f"{test_case.name.replace(' ', '_')}_base"
            self.save_and_compress_image(img, self.output_dir / base_filename)
        
        return self.base_screenshots[test_case.name]
    
    async def run_test_case(self, test_case: TestCase, use_cached_screenshot: bool = True, test_stage: str = 'both') -> TestResult:
        """Execute a single test case.
        
        Args:
            test_case: The test case to run
            use_cached_screenshot: Whether to use cached screenshot
            test_stage: Which stage to test ('coarse', 'fine', or 'both')
        """
        self.logger.info(f"Running test: {test_case.name} (stage: {test_stage})")
        start_time = time.time()
        
        screenshots = {}
        total_tokens = 0
        
        try:
            # Use cached screenshot or navigate to URL
            if use_cached_screenshot:
                screenshot = self.base_screenshots.get(test_case.name)
                if not screenshot:
                    self.logger.error(f"No cached screenshot for {test_case.name}. Call capture_base_screenshot first.")
                    raise ValueError("No cached screenshot available")
            else:
                # Navigate to URL
                await self.page.goto(test_case.url, wait_until="networkidle")
                await asyncio.sleep(2)  # Let page fully load
                screenshot = await self.page.screenshot()
            
            img = Image.open(io.BytesIO(screenshot))
            screenshots['original'] = screenshot
            
            # Default values for when we skip stages
            coarse_cell = ""
            fine_cell = ""
            x, y = 0, 0
            
            # Stage 1: Coarse identification (if needed)
            if test_stage in ['coarse', 'both']:
                self.logger.info("Stage 1: Coarse grid identification...")
                print(f"[Progress] Starting coarse grid analysis...")
                coarse_cell, coarse_response, coarse_messages = await self.identify_coarse_location(
                    screenshot, test_case.target_description
                )
                total_tokens += coarse_response['usage']['total_tokens']
                self.logger.info(f"Coarse cell identified: {coarse_cell}")
                print(f"[Progress] Coarse cell: {coarse_cell}")
            
            # Check if coarse cell matches expected (if provided) - only when testing coarse
            coarse_correct = True
            if test_stage in ['coarse', 'both'] and test_case.expected_coarse_cells:
                coarse_correct = coarse_cell in test_case.expected_coarse_cells
                if not coarse_correct:
                    self.logger.warning(f"Coarse cell mismatch! Expected one of: {test_case.expected_coarse_cells}, Got: {coarse_cell}")
                    print(f"[Progress] ❌ Coarse cell INCORRECT (expected {test_case.expected_coarse_cells})")
                    
                    # Debug the failure
                    print(f"[Progress] 🔍 Asking AI why it failed...")
                    debug_insight = await self.debug_ai_failure(
                        coarse_messages.copy(),  # Use copy to avoid modifying original
                        coarse_cell,
                        test_case.expected_coarse_cells[0],  # Use first expected as correct
                        test_case.name
                    )
                    self.ai_debug_insights.append(debug_insight)
                    total_tokens += 200  # Estimate for debug conversation
                else:
                    print(f"[Progress] ✅ Coarse cell CORRECT")
            
            # For fine-only testing, use the expected coarse cell
            if test_stage == 'fine' and test_case.expected_coarse_cells:
                coarse_cell = test_case.expected_coarse_cells[0]
                self.logger.info(f"Using expected coarse cell for fine-only testing: {coarse_cell}")
            
            # Only proceed to fine grid if appropriate
            if test_stage in ['fine', 'both'] and (coarse_correct or self.bypass_coarse_validation or test_stage == 'fine'):
                # Stage 2: Fine identification
                self.logger.info("Stage 2: Fine grid identification...")
                print(f"[Progress] Starting fine grid analysis...")
                
                # For fine-only testing, we need a dummy coarse response
                if test_stage == 'fine':
                    coarse_response = {'content': coarse_cell, 'usage': {'total_tokens': 0}}
                
                (x, y), fine_response = await self.identify_fine_location(
                    img, coarse_cell, test_case.target_description, coarse_response
                )
                total_tokens += fine_response['usage']['total_tokens']
                self.logger.info(f"Final coordinates: ({x}, {y})")
                print(f"[Progress] Final coordinates: ({x}, {y})")
                fine_cell = fine_response['content'].strip().upper()
                
                # Check if fine cell matches expected (if provided)
                if test_case.expected_fine_cells and fine_cell:
                    if fine_cell in test_case.expected_fine_cells:
                        print(f"[Progress] ✅ Fine cell CORRECT")
                    else:
                        print(f"[Progress] ❌ Fine cell INCORRECT (expected {test_case.expected_fine_cells}, got {fine_cell})")
            else:
                self.logger.info("Skipping fine grid analysis due to incorrect coarse cell")
                print(f"[Progress] ⚠️  Skipping fine grid - coarse cell was incorrect")
            
            # Attempt to click the identified location only if we have valid coordinates
            click_success = False
            error = None
            
            if (coarse_correct or self.bypass_coarse_validation) and x > 0 and y > 0:
                if use_cached_screenshot:
                    # Skip actual click when using cached screenshot
                    self.logger.info(f"Would click at ({x}, {y}) - skipping actual click for cached screenshot")
                    click_success = True  # Assume success based on correct identification
                else:
                    self.logger.info(f"Attempting click at ({x}, {y})...")
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
            else:
                self.logger.info("Skipping click due to incorrect identification")
                error = "Skipped - coarse grid incorrect"
            
            time_taken = time.time() - start_time
            
            # Create result
            # Success is true only if coarse was correct AND click succeeded (or bypass is enabled)
            overall_success = (coarse_correct or self.bypass_coarse_validation) and click_success
            
            result = TestResult(
                test_case=test_case,
                success=overall_success,
                coarse_cell=coarse_cell,
                fine_cell=fine_cell,
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
    
    async def run_all_tests(self, test_cases: List[TestCase], repeat_count: int = 1, test_stage: str = 'both'):
        """Run all test cases multiple times and generate report."""
        total_runs = len(test_cases) * repeat_count
        self.logger.info(f"Running {len(test_cases)} test cases {repeat_count} times each (total: {total_runs} runs)...")
        
        # Capture base screenshots for all test cases first (unless using existing)
        if not hasattr(self, 'using_existing_screenshots'):
            for test_case in test_cases:
                await self.capture_base_screenshot(test_case)
        
        # Store all results including repeats
        all_results = []
        
        for test_case in test_cases:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Test Case: {test_case.name}")
            self.logger.info(f"{'='*60}")
            
            for run_num in range(1, repeat_count + 1):
                self.logger.info(f"\nRun {run_num}/{repeat_count}")
                self.current_run_number = run_num  # Set current run number
                result = await self.run_test_case(test_case, test_stage=test_stage)
                result.run_number = run_num  # Add run number to result
                self.results.append(result)
                all_results.append(result)
                
                # Save screenshots for first run only to avoid clutter
                if run_num == 1:
                    test_dir = self.output_dir / test_case.name.replace(" ", "_")
                    test_dir.mkdir(exist_ok=True)
                    
                    # Save original screenshot as JPEG
                    if 'original' in result.screenshots:
                        img = Image.open(io.BytesIO(result.screenshots['original']))
                        self.save_and_compress_image(img, test_dir / "original")
                    
                    # Add visual marker to show click location
                    if 'original' in result.screenshots:
                        img = Image.open(io.BytesIO(result.screenshots['original']))
                        draw = ImageDraw.Draw(img)
                        x, y = result.final_coordinates
                        
                        # Draw crosshair at click location
                        draw.line([(x-20, y), (x+20, y)], fill=(255, 0, 0), width=3)
                        draw.line([(x, y-20), (x, y+20)], fill=(255, 0, 0), width=3)
                        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=(255, 0, 0))
                        
                        # Save marked image as JPEG
                        self.save_and_compress_image(img, test_dir / "marked_click_location")
                
                # Print progress to prevent timeout
                print(f"[Progress] Completed run {run_num}/{repeat_count} for {test_case.name}")
                
                # Brief pause between runs
                if run_num < repeat_count:
                    await asyncio.sleep(1)
    
    def save_debug_insights(self):
        """Save AI debug insights to a separate file."""
        if not self.ai_debug_insights:
            return
            
        insights_path = self.output_dir / "ai_debug_insights.json"
        with open(insights_path, "w") as f:
            json.dump(self.ai_debug_insights, f, indent=2)
            
        # Also create a markdown summary
        summary_path = self.output_dir / "ai_debug_summary.md"
        with open(summary_path, "w") as f:
            f.write("# AI Self-Debugging Insights\n\n")
            f.write(f"Total failures analyzed: {len(self.ai_debug_insights)}\n\n")
            
            for i, insight in enumerate(self.ai_debug_insights, 1):
                f.write(f"## Failure {i}\n")
                f.write(f"- Test: {insight['test_name']}\n")
                f.write(f"- Selected: {insight['selected']} (incorrect)\n")
                f.write(f"- Correct: {insight['correct']}\n")
                f.write(f"- Time: {insight['timestamp']}\n\n")
                f.write("### AI's Explanation:\n")
                f.write(f"{insight['ai_explanation']}\n\n")
                f.write("---\n\n")
        
        print(f"\n[Progress] 📝 AI debug insights saved to {summary_path}")
    
    def generate_report(self, repeat_count: int = 1):
        """Generate test execution report with aggregate statistics."""
        # Group results by test case
        results_by_test = defaultdict(list)
        for result in self.results:
            results_by_test[result.test_case.name].append(result)
        
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(self.results),
                "unique_tests": len(results_by_test),
                "repeat_count": repeat_count,
                "passed": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
            },
            "configuration": {
                "model": self.model_type,
                "coarse_grid_size": COARSE_GRID_SIZE,
                "fine_grid_size": FINE_GRID_SIZE, 
                "scale_factor": SCALE_FACTOR,
                "padding": PADDING,
                "reasoning_effort": self.reasoning_effort,
            },
            "aggregate_results": {},
            "detailed_results": []
        }
        
        total_time = 0
        total_tokens = 0
        
        # Calculate aggregate statistics for each test case
        for test_name, results in results_by_test.items():
            success_count = sum(1 for r in results if r.success)
            
            # Count coarse and fine cell selections
            coarse_cells = defaultdict(int)
            fine_cells = defaultdict(int)
            
            for result in results:
                coarse_cells[result.coarse_cell] += 1
                fine_cells[result.fine_cell] += 1
                
            # Find most common selections
            most_common_coarse = max(coarse_cells.items(), key=lambda x: x[1])
            most_common_fine = max(fine_cells.items(), key=lambda x: x[1])
            
            report["aggregate_results"][test_name] = {
                "total_runs": len(results),
                "success_count": success_count,
                "success_rate": f"{(success_count / len(results) * 100):.1f}%",
                "coarse_cell_distribution": dict(coarse_cells),
                "fine_cell_distribution": dict(fine_cells),
                "most_common_coarse": f"{most_common_coarse[0]} ({most_common_coarse[1]}/{len(results)} runs)",
                "most_common_fine": f"{most_common_fine[0]} ({most_common_fine[1]}/{len(results)} runs)",
                "avg_time": f"{sum(r.time_taken for r in results) / len(results):.2f}s",
                "avg_tokens": sum(r.api_tokens for r in results) // len(results),
            }
        
        # Add detailed results
        for result in self.results:
            report["detailed_results"].append({
                "test_name": result.test_case.name,
                "run_number": result.run_number,
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
            "overall_success_rate": f"{(report['test_run']['passed'] / report['test_run']['total_runs'] * 100):.1f}%",
            "total_time": f"{total_time:.2f}s",
            "avg_time_per_run": f"{total_time / len(self.results):.2f}s",
            "total_api_tokens": total_tokens,
            "avg_tokens_per_run": total_tokens // len(self.results),
        }
        
        # Save JSON report
        with open(self.output_dir / "test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        summary = f"""# Two-Stage Grid System Test Report

## Configuration
- Model: {self.model_type.upper()} {'(o4-mini)' if self.model_type == 'openai' else '(Gemini 2.5 Flash)'}
- Coarse Grid: {COARSE_GRID_SIZE}x{COARSE_GRID_SIZE}
- Fine Grid: {FINE_GRID_SIZE}x{FINE_GRID_SIZE}
- Scale Factor: {SCALE_FACTOR}x
- Padding: {PADDING}px
- Repeat Count: {repeat_count}
- Reasoning Effort: {self.reasoning_effort}

## Overall Summary
- Total Runs: {report['test_run']['total_runs']}
- Unique Tests: {report['test_run']['unique_tests']}
- Passed: {report['test_run']['passed']}
- Failed: {report['test_run']['failed']}
- Success Rate: {report['summary']['overall_success_rate']}
- Total Time: {report['summary']['total_time']}
- Total API Tokens: {report['summary']['total_api_tokens']}

## Aggregate Results by Test Case

"""
        
        for test_name, agg_result in report["aggregate_results"].items():
            summary += f"### {test_name}\n"
            summary += f"- Success Rate: {agg_result['success_rate']} ({agg_result['success_count']}/{agg_result['total_runs']} runs)\n"
            summary += f"- Most Common Coarse Cell: {agg_result['most_common_coarse']}\n"
            summary += f"- Most Common Fine Cell: {agg_result['most_common_fine']}\n"
            summary += f"- Average Time: {agg_result['avg_time']}\n"
            summary += f"- Average Tokens: {agg_result['avg_tokens']}\n\n"
            
            # Show cell distribution if there's variation
            if len(agg_result['coarse_cell_distribution']) > 1:
                summary += "**Coarse Cell Distribution:**\n"
                for cell, count in sorted(agg_result['coarse_cell_distribution'].items()):
                    summary += f"  - {cell}: {count} times\n"
                summary += "\n"
            
            if len(agg_result['fine_cell_distribution']) > 1:
                summary += "**Fine Cell Distribution:**\n"
                for cell, count in sorted(agg_result['fine_cell_distribution'].items()):
                    summary += f"  - {cell}: {count} times\n"
                summary += "\n"

        summary += "## Detailed Results\n\n"
        summary += "| Test Name | Run # | Success | Coarse Cell | Fine Cell | Coordinates | Time | Tokens |\n"
        summary += "|-----------|-------|---------|-------------|-----------|-------------|------|--------|\n"
        
        for result in report["detailed_results"]:
            summary += f"| {result['test_name']} | {result['run_number']} | {'✅' if result['success'] else '❌'} | "
            summary += f"{result['coarse_cell']} | {result['fine_cell']} | "
            summary += f"{result['coordinates']} | {result['time_taken']:.2f}s | {result['api_tokens']} |\n"
        
        with open(self.output_dir / "test_summary.md", "w") as f:
            f.write(summary)
        
        # Print summary to console
        print("\n" + "="*60)
        print("TEST EXECUTION COMPLETE")
        print("="*60)
        print(f"Success Rate: {report['summary']['overall_success_rate']}")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)


async def main():
    """Main test execution."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test two-stage grid system for UI element identification')
    parser.add_argument('-n', '--repeat', type=int, default=1,
                        help='Number of times to repeat each test (default: 1)')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'history', 'debate', 'google', 'github_star', 'youtube_play', 'amazon_cart'],
                        help='Which test case to run (default: all)')
    parser.add_argument('--bypass-coarse-validation', action='store_true',
                        help='Bypass coarse grid validation and continue to fine grid even if coarse is incorrect')
    parser.add_argument('--reasoning-effort', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort level for AI model (default: medium)')
    parser.add_argument('--model', type=str, default='openai',
                        choices=['openai', 'gemini'],
                        help='Which AI model to use for grid analysis (default: openai)')
    parser.add_argument('--use-existing-screenshot', type=str, 
                        help='Path to existing test results directory to reuse screenshots')
    parser.add_argument('--test-stage', type=str, default='both',
                        choices=['coarse', 'fine', 'both'],
                        help='Which stage to test (default: both)')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get appropriate API key based on model selection
    if args.model == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment")
            sys.exit(1)
    elif args.model == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY not found in environment")
            sys.exit(1)
    
    # Define all test cases
    all_test_cases = {
        'history': TestCase(
            name="Wikipedia_History_Link",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            target_description="The 'History' link in the table of contents on the left side",
            expected_element="History link in TOC",
            element_type="link",
            notes="Primary test case - small link in sidebar TOC"
        ),
        'debate': TestCase(
            name="Wikipedia_Debate_Link",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            target_description="'debate' link" if args.model == "gemini" else "The 'debate' link in the main article text (in the introductory paragraphs)",
            expected_element="debate link in article text",
            element_type="link",
            notes="Link embedded within article text - tests precision for inline links",
            expected_coarse_cells=["B5"],  # B5 is the only correct cell
            expected_fine_cells=["E3", "F3"]  # E3 or F3 based on fine grid analysis
        ),
        'google': TestCase(
            name="Google_GDPR_Accept",
            url="https://www.google.com",
            target_description="The 'Accept' or 'Aceptar' button on the GDPR/cookie consent banner (likely blue button)",
            expected_element="GDPR cookie consent accept button",
            element_type="button",
            notes="GDPR consent button - tests banner/modal UI elements",
            expected_coarse_cells=["D6", "E6", "E7"],  # Accept all three - E7 covers part of button
            expected_fine_cells=["A1", "B1", "C1", "A7", "B7", "C7", "D7", "A8", "B8", "C8", "D8"],  # Button spans these cells
        ),
        'github_star': TestCase(
            name="GitHub_Star_Button",
            url="https://github.com/microsoft/vscode",
            target_description="The 'Star' button with star icon and count (in the top right area of the repository header)",
            expected_element="GitHub star button",
            element_type="button",
            notes="Custom button with icon and dynamic count - tests compound UI elements",
            expected_coarse_cells=["G1"],  # Top right corner - only valid cell
            expected_fine_cells=["C7", "D7", "E7", "F7", "G7", "D8", "E8", "F8", "G8"],  # Button spans these cells
        ),
        'youtube_play': TestCase(
            name="YouTube_Play_Button",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            target_description="The play/pause button in the video player control bar at the bottom left of the player",
            expected_element="YouTube video play button",
            element_type="button",
            notes="Control bar button - tests player controls UI elements",
            expected_coarse_cells=["A6"],  # Bottom left of player
            expected_fine_cells=["B2", "C2", "B3", "C3", "B4", "C4"],  # Play button spans these cells
        ),
        'amazon_cart': TestCase(
            name="eBay_Cart_Icon",
            url="https://www.ebay.com",
            target_description="The shopping cart icon in the top navigation bar",
            expected_element="eBay shopping cart",
            element_type="button",
            notes="Icon button - tests small interactive elements in navigation",
            expected_coarse_cells=["G1"],  # Top right corner
            expected_fine_cells=["F1", "G1", "F2", "G2"],  # Small icon spans these 4 cells
        ),
    }
    
    # Select test cases based on argument
    if args.test == 'all':
        test_cases = list(all_test_cases.values())
    else:
        test_cases = [all_test_cases[args.test]]
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print test configuration
    print(f"\n{'='*60}")
    print(f"Two-Stage Grid System Test")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Model: {args.model.upper()} {'(o4-mini)' if args.model == 'openai' else '(Gemini 2.5 Flash)'}")
    print(f"  - Test Stage: {args.test_stage.upper()}")
    print(f"  - Coarse Grid: {COARSE_GRID_SIZE}x{COARSE_GRID_SIZE}")
    print(f"  - Fine Grid: {FINE_GRID_SIZE}x{FINE_GRID_SIZE}")
    print(f"  - Repeat Count: {args.repeat}")
    print(f"  - Reasoning Effort: {args.reasoning_effort}")
    print(f"  - Test Cases: {', '.join(tc.name for tc in test_cases)}")
    if args.use_existing_screenshot:
        print(f"  - Using screenshots from: {args.use_existing_screenshot}")
    print(f"{'='*60}\n")
    
    # Run tests
    tester = TwoStageGridTester(api_key, bypass_coarse_validation=args.bypass_coarse_validation, 
                               reasoning_effort=args.reasoning_effort, model_type=args.model)
    
    try:
        # Setup
        if args.use_existing_screenshot:
            # Skip browser setup when using existing screenshots
            tester.using_existing_screenshots = True
            tester.output_dir = Path(f"grid_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            tester.output_dir.mkdir(exist_ok=True)
            # Load existing screenshots
            for test_case in test_cases:
                tester.load_existing_screenshots(args.use_existing_screenshot, test_case)
            # Initialize client only
            if tester.model_type == "openai":
                tester.client = OpenAIClient(api_key=tester.api_key, reasoning_effort=tester.reasoning_effort)
                tester.logger.info("Using OpenAI model (o4-mini)")
            elif tester.model_type == "gemini":
                tester.client = GeminiClient(api_key=tester.api_key, reasoning_effort=tester.reasoning_effort)
                tester.logger.info("Using Google Gemini model (2.5 Flash)")
        else:
            await tester.setup()
            
        # Run tests
        await tester.run_all_tests(test_cases, repeat_count=args.repeat, test_stage=args.test_stage)
        tester.generate_report(repeat_count=args.repeat)
        tester.save_debug_insights()
    finally:
        if not args.use_existing_screenshot:
            await tester.teardown()


if __name__ == "__main__":
    asyncio.run(main())