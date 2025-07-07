"""
Refactored Action Agent with full action execution responsibility.

This is the prototype for the architectural refactoring where Action Agent
owns the entire action lifecycle: validation → coordinates → execution → results.
"""

import asyncio
import base64
import json
import traceback
from io import BytesIO
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw

from src.agents.base_agent import BaseAgent
from src.browser.driver import BrowserDriver
from src.config.agent_prompts import ACTION_AGENT_SYSTEM_PROMPT
from src.config.settings import get_settings
from src.core.enhanced_types import (
    EnhancedActionResult,
    ValidationStatus
)
from src.core.types import ActionInstruction, TestStep
from src.grid.overlay import GridOverlay
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class ActionAgentV2(BaseAgent):
    """
    Refactored AI agent that owns the complete action execution lifecycle.
    
    This agent:
    1. Validates if an action makes sense in the current context
    2. Determines grid coordinates for the action
    3. Executes the browser action
    4. Captures comprehensive results for debugging
    """
    
    def __init__(
        self,
        name: str = "ActionAgentV2",
        browser_driver: Optional[BrowserDriver] = None,
        **kwargs
    ):
        """Initialize the refactored Action Agent."""
        super().__init__(name=name, **kwargs)
        self.system_prompt = ACTION_AGENT_SYSTEM_PROMPT
        self.browser_driver = browser_driver
        
        # Initialize grid system
        settings = get_settings()
        self.grid_overlay = GridOverlay(grid_size=settings.grid_size)
        self.confidence_threshold = settings.grid_confidence_threshold
        
    async def execute_action(
        self,
        test_step: TestStep,
        test_context: Dict,
        screenshot: Optional[bytes] = None
    ) -> EnhancedActionResult:
        """
        Execute a complete action with validation, coordination, and execution.
        
        Args:
            test_step: The test step to execute
            test_context: Context about the test plan and previous steps
            screenshot: Optional pre-captured screenshot
            
        Returns:
            Comprehensive action result with debugging information
        """
        logger.info("Executing action with full lifecycle", extra={
            "step_number": test_step.step_number,
            "action_type": test_step.action_instruction.action_type.value,
            "description": test_step.description
        })
        
        # Initialize result
        result = EnhancedActionResult(
            action_type=test_step.action_instruction.action_type.value,
            validation_passed=False,
            validation_status=ValidationStatus.SKIPPED,
            validation_reasoning="Not yet validated",
            grid_cell="",
            grid_coordinates=(0, 0),
            coordinate_reasoning="",
            url_before="",
            url_after="",
            page_title_before="",
            page_title_after="",
            execution_success=False,
            execution_time_ms=0.0,
            test_context=test_context
        )
        
        try:
            # Capture initial browser state
            if self.browser_driver:
                result.url_before = await self.browser_driver.get_page_url()
                result.page_title_before = await self.browser_driver.get_page_title()
                
                # Get screenshot if not provided
                if not screenshot:
                    screenshot = await self.browser_driver.screenshot()
                result.screenshot_before = screenshot
            
            # Phase 1: Validation
            validation_result = await self._validate_action(
                test_step.action_instruction,
                screenshot,
                test_context
            )
            
            result.validation_passed = validation_result["valid"]
            result.validation_status = ValidationStatus(validation_result["status"])
            result.validation_reasoning = validation_result["reasoning"]
            result.validation_confidence = validation_result["confidence"]
            
            if not result.validation_passed:
                logger.warning("Action validation failed", extra={
                    "reasoning": result.validation_reasoning,
                    "confidence": result.validation_confidence
                })
                return result
            
            # Phase 2: Coordinate Determination
            coord_result = await self._determine_coordinates(
                screenshot,
                test_step.action_instruction
            )
            
            result.grid_cell = coord_result["cell"]
            result.offset_x = coord_result["offset_x"]
            result.offset_y = coord_result["offset_y"]
            result.coordinate_confidence = coord_result["confidence"]
            result.coordinate_reasoning = coord_result["reasoning"]
            
            # Create grid overlay screenshot
            result.grid_screenshot_before = self.grid_overlay.create_overlay_image(screenshot)
            
            # Create highlighted screenshot
            result.grid_screenshot_highlighted = self._create_highlighted_screenshot(
                screenshot,
                result.grid_cell
            )
            
            # Phase 3: Action Execution
            if self.browser_driver and result.coordinate_confidence >= 0.7:
                execution_start = asyncio.get_event_loop().time()
                
                try:
                    # Convert grid coordinates to pixels
                    viewport_width, viewport_height = await self.browser_driver.get_viewport_size()
                    self.grid_overlay.initialize(viewport_width, viewport_height)
                    
                    from src.core.types import GridCoordinate
                    coord = GridCoordinate(
                        cell=result.grid_cell,
                        offset_x=result.offset_x,
                        offset_y=result.offset_y,
                        confidence=result.coordinate_confidence
                    )
                    
                    x, y = self.grid_overlay.coordinate_to_pixels(coord)
                    result.grid_coordinates = (x, y)
                    
                    # Execute the action
                    if test_step.action_instruction.action_type.value == "click":
                        await self.browser_driver.click(x, y)
                    elif test_step.action_instruction.action_type.value == "type":
                        # Click to focus first
                        await self.browser_driver.click(x, y)
                        await self.browser_driver.wait(200)
                        # Type the text
                        if test_step.action_instruction.value:
                            await self.browser_driver.type_text(test_step.action_instruction.value)
                            await self.browser_driver.wait(500)
                    
                    # Wait for UI update
                    await self.browser_driver.wait(1000)
                    
                    # Capture post-action state
                    result.screenshot_after = await self.browser_driver.screenshot()
                    result.url_after = await self.browser_driver.get_page_url()
                    result.page_title_after = await self.browser_driver.get_page_title()
                    
                    result.execution_success = True
                    result.execution_time_ms = (asyncio.get_event_loop().time() - execution_start) * 1000
                    
                except Exception as e:
                    result.execution_success = False
                    result.execution_error = str(e)
                    result.execution_error_type = type(e).__name__
                    result.execution_traceback = traceback.format_exc()
                    logger.error("Action execution failed", extra={
                        "error": str(e),
                        "traceback": result.execution_traceback
                    })
            
            # Phase 4: Result Analysis
            if result.execution_success and result.screenshot_after:
                analysis = await self._analyze_result(
                    test_step.action_instruction,
                    result
                )
                result.ai_analysis = analysis
            
            return result
            
        except Exception as e:
            logger.error("Unexpected error in action execution", extra={
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            result.execution_error = str(e)
            result.execution_error_type = type(e).__name__
            result.execution_traceback = traceback.format_exc()
            return result
    
    async def _validate_action(
        self,
        instruction: ActionInstruction,
        screenshot: bytes,
        context: Dict
    ) -> Dict:
        """Validate if the action makes sense in current context."""
        # Prepare validation prompt
        prompt = f"""Analyze this screenshot and determine if the following action is valid and makes sense.

Test Context:
- Test Plan: {context.get('test_plan_name', 'Unknown')}
- Current Step: {context.get('current_step_description', 'Unknown')}
- Previous Steps: {context.get('previous_steps_summary', 'None')}

Action to Validate:
- Type: {instruction.action_type.value}
- Target: {instruction.target or 'Not specified'}
- Description: {instruction.description}
- Expected Outcome: {instruction.expected_outcome}

Please analyze and respond in JSON format:
{{
    "valid": true/false,
    "status": "valid/invalid/warning",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation",
    "concerns": ["List of any concerns"],
    "suggestions": ["Alternative approaches if needed"]
}}

Consider:
1. Is the target element visible on screen?
2. Does the action make sense given the current UI state?
3. Are there any obvious blockers (popups, loading screens, etc.)?
4. Is this the right time to perform this action in the test flow?"""

        # Convert screenshot to base64
        base64_image = base64.b64encode(screenshot).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        response = await self.call_openai(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
            
            return {
                "valid": content.get("valid", False),
                "status": content.get("status", "invalid"),
                "confidence": float(content.get("confidence", 0.0)),
                "reasoning": content.get("reasoning", "No reasoning provided"),
                "concerns": content.get("concerns", []),
                "suggestions": content.get("suggestions", [])
            }
        except Exception as e:
            logger.error("Failed to parse validation response", extra={"error": str(e)})
            return {
                "valid": False,
                "status": "invalid",
                "confidence": 0.0,
                "reasoning": f"Failed to validate: {str(e)}",
                "concerns": ["Validation parsing failed"],
                "suggestions": []
            }
    
    async def _determine_coordinates(
        self,
        screenshot: bytes,
        instruction: ActionInstruction
    ) -> Dict:
        """Determine grid coordinates for the action."""
        # Create grid overlay
        overlay_image = self.grid_overlay.create_overlay_image(screenshot)
        
        prompt = f"""Analyze this screenshot with grid overlay to find the target element.

Action: {instruction.action_type.value}
Target: {instruction.target or instruction.description}

The grid has {self.grid_overlay.grid_size}x{self.grid_overlay.grid_size} cells.
Provide the exact location in JSON format:
{{
    "cell": "Grid cell (e.g., 'M23')",
    "offset_x": 0.5,
    "offset_y": 0.5,
    "confidence": 0.0-1.0,
    "reasoning": "Why this location was chosen"
}}"""

        base64_image = base64.b64encode(overlay_image).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        response = await self.call_openai(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
            
            return {
                "cell": content.get("cell", "A1"),
                "offset_x": float(content.get("offset_x", 0.5)),
                "offset_y": float(content.get("offset_y", 0.5)),
                "confidence": float(content.get("confidence", 0.0)),
                "reasoning": content.get("reasoning", "")
            }
        except Exception as e:
            logger.error("Failed to parse coordinate response", extra={"error": str(e)})
            return {
                "cell": "A1",
                "offset_x": 0.5,
                "offset_y": 0.5,
                "confidence": 0.0,
                "reasoning": f"Failed to determine coordinates: {str(e)}"
            }
    
    async def _analyze_result(
        self,
        instruction: ActionInstruction,
        result: EnhancedActionResult
    ) -> Dict:
        """Analyze the result of the action."""
        prompt = f"""Analyze the result of this action.

Action Performed:
- Type: {instruction.action_type.value}
- Target: {instruction.target or instruction.description}
- Expected: {instruction.expected_outcome}

Execution Details:
- Grid Cell: {result.grid_cell}
- Execution Time: {result.execution_time_ms}ms
- URL Change: {result.url_before} → {result.url_after}

Compare the before and after screenshots and provide analysis in JSON format:
{{
    "success": true/false,
    "confidence": 0.0-1.0,
    "actual_outcome": "What actually happened",
    "matches_expected": true/false,
    "ui_changes": ["List of observed UI changes"],
    "recommendations": ["Any recommendations for next steps"]
}}"""

        # Create comparison image
        if result.screenshot_before and result.screenshot_after:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(result.screenshot_before).decode('utf-8')}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(result.screenshot_after).decode('utf-8')}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.call_openai(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            try:
                content = response.get("content", {})
                if isinstance(content, str):
                    content = json.loads(content)
                return content
            except Exception as e:
                logger.error("Failed to parse analysis response", extra={"error": str(e)})
                return {
                    "success": False,
                    "confidence": 0.0,
                    "actual_outcome": "Failed to analyze",
                    "matches_expected": False,
                    "ui_changes": [],
                    "recommendations": []
                }
        
        return {
            "success": result.execution_success,
            "confidence": 0.5,
            "actual_outcome": "No screenshot comparison available",
            "matches_expected": False,
            "ui_changes": [],
            "recommendations": []
        }
    
    def _create_highlighted_screenshot(
        self,
        screenshot: bytes,
        grid_cell: str
    ) -> bytes:
        """Create a screenshot with the selected grid cell highlighted."""
        # Load image
        img = Image.open(BytesIO(screenshot))
        width, height = img.size
        
        # Initialize grid if needed
        if self.grid_overlay.viewport_width != width:
            self.grid_overlay.initialize(width, height)
        
        # Get cell bounds
        x, y, cell_width, cell_height = self.grid_overlay.get_cell_bounds(grid_cell)
        
        # Create overlay with grid
        overlay_img = Image.open(BytesIO(self.grid_overlay.create_overlay_image(screenshot)))
        
        # Draw highlight on the selected cell
        draw = ImageDraw.Draw(overlay_img, "RGBA")
        
        # Draw a semi-transparent red rectangle
        draw.rectangle(
            [x, y, x + cell_width, y + cell_height],
            fill=(255, 0, 0, 100),
            outline=(255, 0, 0, 255),
            width=3
        )
        
        # Add label
        draw.text(
            (x + 5, y + 5),
            f"SELECTED: {grid_cell}",
            fill=(255, 255, 255, 255)
        )
        
        # Convert back to bytes
        output = BytesIO()
        overlay_img.save(output, format="PNG")
        return output.getvalue()