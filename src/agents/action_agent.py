"""
Action Agent implementation.

Refactored to own the complete action execution lifecycle:
1. Validates actions before attempting
2. Analyzes screenshots and converts visual instructions into precise grid coordinates
3. Executes browser actions
4. Captures comprehensive results for debugging
"""

import asyncio
import base64
import json
import traceback
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, Optional, Tuple, Any

from PIL import Image, ImageDraw

from src.agents.base_agent import BaseAgent
from src.browser.driver import BrowserDriver
from src.config.agent_prompts import ACTION_AGENT_SYSTEM_PROMPT
from src.config.settings import get_settings
from src.core.types import ActionInstruction, GridAction, GridCoordinate, TestStep
from src.core.enhanced_types import (
    EnhancedActionResult, ValidationResult, CoordinateResult,
    ExecutionResult, BrowserState, AIAnalysis
)
from src.grid.overlay import GridOverlay
from src.grid.refinement import GridRefinement
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class ActionAgent(BaseAgent):
    """
    Refactored AI agent that owns the complete action execution lifecycle.
    
    This agent:
    1. Validates if an action makes sense in the current context
    2. Analyzes screenshots and determines precise grid coordinates
    3. Executes the browser action
    4. Captures comprehensive results for debugging
    
    The refactored architecture gives Action Agent full responsibility for
    action execution, improving error context and debugging capabilities.
    """
    
    def __init__(
        self,
        name: str = "ActionAgent",
        browser_driver: Optional[BrowserDriver] = None,
        **kwargs
    ):
        """
        Initialize the Action Agent.
        
        Args:
            name: Agent name
            browser_driver: Browser driver for action execution
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(name=name, **kwargs)
        self.system_prompt = ACTION_AGENT_SYSTEM_PROMPT
        self.browser_driver = browser_driver
        
        # Initialize grid components
        settings = get_settings()
        self.grid_overlay = GridOverlay(grid_size=settings.grid_size)
        self.grid_refinement = GridRefinement(base_grid=self.grid_overlay)
        
        # Configuration
        self.confidence_threshold = settings.grid_confidence_threshold
        self.refinement_enabled = settings.grid_refinement_enabled
    
    async def execute_action(
        self,
        test_step: TestStep,
        test_context: Dict[str, Any],
        screenshot: Optional[bytes] = None
    ) -> EnhancedActionResult:
        """
        Execute a complete action with validation, coordination, and execution.
        
        This is the new main entry point that owns the full action lifecycle.
        
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
            test_step_id=test_step.step_id,
            test_step=test_step,
            test_context=test_context,
            validation=ValidationResult(
                valid=False,
                confidence=0.0,
                reasoning="Not yet validated"
            )
        )
        
        try:
            # Capture initial browser state
            if self.browser_driver:
                viewport_size = await self.browser_driver.get_viewport_size()
                result.browser_state_before = BrowserState(
                    url=await self.browser_driver.get_page_url(),
                    title=await self.browser_driver.get_page_title(),
                    viewport_size=viewport_size,
                    screenshot=screenshot or await self.browser_driver.screenshot()
                )
                
                # Use provided screenshot or capture new one
                if not screenshot:
                    screenshot = result.browser_state_before.screenshot
            
            # Phase 1: Validation
            validation_result = await self._validate_action(
                test_step.action_instruction,
                screenshot,
                test_context
            )
            result.validation = validation_result
            
            if not result.validation.valid:
                logger.warning("Action validation failed", extra={
                    "reasoning": result.validation.reasoning,
                    "confidence": result.validation.confidence
                })
                result.failure_phase = "validation"
                result.timestamp_end = datetime.now(timezone.utc)
                return result
            
            # Phase 2: Coordinate Determination
            grid_action = await self.determine_action(screenshot, test_step.action_instruction)
            
            # Convert GridAction to CoordinateResult
            self.grid_overlay.initialize(viewport_size[0], viewport_size[1])
            x, y = self.grid_overlay.coordinate_to_pixels(grid_action.coordinate)
            result.coordinates = CoordinateResult(
                grid_cell=grid_action.coordinate.cell,
                grid_coordinates=(x, y),
                offset_x=grid_action.coordinate.offset_x,
                offset_y=grid_action.coordinate.offset_y,
                confidence=grid_action.coordinate.confidence,
                reasoning=getattr(grid_action.coordinate, 'reasoning', ''),
                refined=grid_action.coordinate.refined
            )
            
            # Create grid overlay screenshots
            result.grid_screenshot_before = self.grid_overlay.create_overlay_image(screenshot)
            result.grid_screenshot_highlighted = self._create_highlighted_screenshot(
                screenshot,
                result.coordinates.grid_cell
            )
            
            # Phase 3: Action Execution
            if self.browser_driver and result.coordinates.confidence >= 0.7:
                execution_start = asyncio.get_event_loop().time()
                
                try:
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
                    result.browser_state_after = BrowserState(
                        url=await self.browser_driver.get_page_url(),
                        title=await self.browser_driver.get_page_title(),
                        viewport_size=viewport_size,
                        screenshot=await self.browser_driver.screenshot()
                    )
                    
                    result.execution = ExecutionResult(
                        success=True,
                        execution_time_ms=(asyncio.get_event_loop().time() - execution_start) * 1000
                    )
                    
                except Exception as e:
                    result.execution = ExecutionResult(
                        success=False,
                        execution_time_ms=(asyncio.get_event_loop().time() - execution_start) * 1000,
                        error_message=str(e),
                        error_traceback=traceback.format_exc()
                    )
                    result.failure_phase = "execution"
                    logger.error("Action execution failed", extra={
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
            else:
                # Low confidence, skip execution
                result.execution = ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message=f"Coordinate confidence too low: {result.coordinates.confidence}"
                )
                result.failure_phase = "coordinates"
            
            # Phase 4: Result Analysis
            if result.execution and result.execution.success and result.browser_state_after:
                analysis = await self._analyze_result(
                    test_step.action_instruction,
                    result
                )
                result.ai_analysis = analysis
            
            # Set overall success
            result.overall_success = (
                result.validation.valid and
                result.coordinates is not None and
                result.execution is not None and
                result.execution.success
            )
            
            result.timestamp_end = datetime.now(timezone.utc)
            return result
            
        except Exception as e:
            logger.error("Unexpected error in action execution", extra={
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            if not result.execution:
                result.execution = ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message=str(e),
                    error_traceback=traceback.format_exc()
                )
            result.failure_phase = "unknown"
            result.timestamp_end = datetime.now(timezone.utc)
            return result
    
    async def determine_action(
        self, screenshot: bytes, instruction: ActionInstruction
    ) -> GridAction:
        """
        Determine grid coordinates for an action from a screenshot.
        
        This method is maintained for backward compatibility but now focuses
        only on coordinate determination, not execution.
        
        Args:
            screenshot: Screenshot of current state
            instruction: Action instruction to execute
            
        Returns:
            Grid-based action with coordinates
        """
        logger.info("Determining action coordinates", extra={
            "action_type": instruction.action_type.value,
            "target": instruction.target,
            "description": instruction.description
        })
        
        # Create screenshot with grid overlay for analysis
        overlay_image = self._create_overlay_image(screenshot)
        
        # Analyze the screenshot to find coordinates
        initial_coords = await self._analyze_screenshot(
            overlay_image, 
            instruction
        )
        
        # Check if refinement is needed
        if (self.refinement_enabled and 
            initial_coords.confidence < self.confidence_threshold):
            logger.info("Confidence below threshold, applying refinement", extra={
                "initial_confidence": initial_coords.confidence,
                "threshold": self.confidence_threshold,
                "cell": initial_coords.cell
            })
            
            # Apply adaptive refinement
            refined_coords = await self._apply_refinement(
                screenshot,
                initial_coords,
                instruction
            )
            
            # Create action with refined coordinates
            action = GridAction(
                instruction=instruction,
                coordinate=refined_coords,
                screenshot_before=None  # Will be set by browser driver
            )
        else:
            # Use initial coordinates
            action = GridAction(
                instruction=instruction,
                coordinate=initial_coords,
                screenshot_before=None
            )
        
        logger.info("Action coordinates determined", extra={
            "cell": action.coordinate.cell,
            "offset_x": action.coordinate.offset_x,
            "offset_y": action.coordinate.offset_y,
            "confidence": action.coordinate.confidence,
            "refined": action.coordinate.refined
        })
        
        return action
    
    async def refine_coordinates(
        self, cropped_region: bytes, initial_coords: GridCoordinate
    ) -> GridCoordinate:
        """
        Refine grid coordinates using adaptive refinement.
        
        Args:
            cropped_region: Cropped screenshot region
            initial_coords: Initial grid coordinates
            
        Returns:
            Refined grid coordinates with higher precision
        """
        return await self._apply_refinement_to_region(
            cropped_region,
            initial_coords
        )
    
    def _create_overlay_image(self, screenshot: bytes) -> bytes:
        """Create screenshot with grid overlay for AI analysis."""
        # Check if grid needs initialization
        image = Image.open(BytesIO(screenshot))
        width, height = image.size
        
        if self.grid_overlay.viewport_width == 0:
            self.grid_overlay.initialize(width, height)
        
        # Create overlay image (GridOverlay expects bytes)
        return self.grid_overlay.create_overlay_image(screenshot)
    
    async def _analyze_screenshot(
        self, overlay_image: bytes, instruction: ActionInstruction
    ) -> GridCoordinate:
        """Analyze screenshot with AI to find target coordinates."""
        # Prepare the analysis prompt
        prompt = self._build_analysis_prompt(instruction)
        
        # Convert image to base64 for AI
        base64_image = base64.b64encode(overlay_image).decode('utf-8')
        
        # Build messages
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
        
        # Call AI for analysis
        response = await self.call_openai(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for precision
        )
        
        # Parse response
        return self._parse_coordinate_response(response)
    
    def _build_analysis_prompt(self, instruction: ActionInstruction) -> str:
        """Build the prompt for screenshot analysis."""
        prompt = f"""Analyze this screenshot with a grid overlay to locate the target element.

Action Type: {instruction.action_type.value}
Target: {instruction.target or 'Not specified'}
Description: {instruction.description}

The screenshot has a {self.grid_overlay.grid_size}x{self.grid_overlay.grid_size} grid overlay.
Grid cells are labeled with columns (A-Z, AA-AZ, etc.) and rows (1-{self.grid_overlay.grid_size}).

Please provide the location in the following JSON format:
{{
    "cell": "Grid cell identifier (e.g., 'M23')",
    "offset_x": 0.5,  // X offset within cell (0.0=left edge, 1.0=right edge)
    "offset_y": 0.5,  // Y offset within cell (0.0=top edge, 1.0=bottom edge)
    "confidence": 0.9,  // Confidence score (0.0-1.0)
    "reasoning": "Brief explanation of why this location was chosen"
}}

Guidelines:
- Look for the exact element described in the target/description
- Consider button boundaries and clickable areas
- For text input fields, aim for the center
- For buttons, aim for the center of the clickable area
- If the element spans multiple cells, choose the most central one
- If you cannot find the element with high confidence, still provide your best guess"""
        
        return prompt
    
    def _parse_coordinate_response(self, response: Dict) -> GridCoordinate:
        """Parse AI response into GridCoordinate."""
        try:
            # Check if response is actually a coroutine (testing issue)
            if asyncio.iscoroutine(response):
                logger.error("Response is a coroutine, not a dict - likely test mock issue")
                raise ValueError("Response is a coroutine")
                
            content = response.get("content", {})
            
            # Handle string content (JSON)
            if isinstance(content, str):
                content = json.loads(content)
            
            # Extract coordinate data
            cell = content.get("cell", "A1")
            offset_x = float(content.get("offset_x", 0.5))
            offset_y = float(content.get("offset_y", 0.5))
            confidence = float(content.get("confidence", 0.5))
            
            # Log reasoning if provided
            reasoning = content.get("reasoning", "")
            if reasoning:
                logger.debug("AI reasoning for coordinate selection", extra={
                    "reasoning": reasoning,
                    "cell": cell,
                    "confidence": confidence
                })
            
            coord = GridCoordinate(
                cell=cell,
                offset_x=max(0.0, min(1.0, offset_x)),  # Clamp to valid range
                offset_y=max(0.0, min(1.0, offset_y)),
                confidence=max(0.0, min(1.0, confidence)),
                refined=False
            )
            
            # Store reasoning for later use
            setattr(coord, 'reasoning', reasoning)
            
            return coord
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse coordinate response", extra={
                "error": str(e),
                "response": response,
                "error_type": type(e).__name__
            })
            # Return default coordinate with low confidence
            return GridCoordinate(
                cell="A1",
                offset_x=0.5,
                offset_y=0.5,
                confidence=0.1,
                refined=False
            )
    
    async def _apply_refinement(
        self, screenshot: bytes, initial_coords: GridCoordinate,
        instruction: ActionInstruction
    ) -> GridCoordinate:
        """Apply adaptive refinement to improve coordinate precision."""
        # Get the refined coordinates from GridRefinement
        refined_coords = self.grid_refinement.refine_coordinate(
            screenshot,
            initial_coords,
            instruction.target or instruction.description
        )
        
        # If refinement improved confidence, return the refined coordinates
        # Otherwise perform additional AI-based refinement
        if refined_coords.confidence > initial_coords.confidence:
            return refined_coords
        
        # GridRefinement didn't improve confidence, try our own AI refinement
        # Get the cropped region for detailed analysis
        x, y, width, height = self.grid_overlay.get_refinement_region(initial_coords.cell)
        image = Image.open(BytesIO(screenshot))
        cropped = image.crop((x, y, x + width, y + height))
        
        # Convert cropped region to bytes
        buffer = BytesIO()
        cropped.save(buffer, format='PNG')
        cropped_bytes = buffer.getvalue()
        
        return await self._apply_refinement_to_region(
            cropped_bytes,
            initial_coords
        )
    
    async def _apply_refinement_to_region(
        self, cropped_region: bytes, initial_coords: GridCoordinate
    ) -> GridCoordinate:
        """Apply refinement to a specific region."""
        # This method can be called directly for focused refinement
        prompt = f"""This is a zoomed-in view of grid cell {initial_coords.cell} and its surrounding area.
The image shows a 3x3 grid of the original cells, now divided into a finer 9x9 grid.

Please identify the precise location within this refined grid.
The center cell (positions 4-6 horizontally, 4-6 vertically) corresponds to the original cell {initial_coords.cell}.

Provide the refined position in JSON format:
{{
    "refined_x": 5,  // X position in 9x9 grid (1-9)
    "refined_y": 5,  // Y position in 9x9 grid (1-9)  
    "confidence": 0.95,  // Updated confidence
    "reasoning": "Explanation of the refined position"
}}"""
        
        # Convert image to base64
        base64_image = base64.b64encode(cropped_region).decode('utf-8')
        
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
            temperature=0.2  # Even lower temperature for refinement
        )
        
        # Parse refinement response
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
            
            refined_x = int(content.get("refined_x", 5))
            refined_y = int(content.get("refined_y", 5))
            confidence = float(content.get("confidence", 0.8))
            
            # Convert 9x9 position back to offset within original cell
            # Center cell (5,5) = (0.5, 0.5) offset
            offset_x = (refined_x - 1) / 9.0
            offset_y = (refined_y - 1) / 9.0
            
            return GridCoordinate(
                cell=initial_coords.cell,
                offset_x=max(0.0, min(1.0, offset_x)),
                offset_y=max(0.0, min(1.0, offset_y)),
                confidence=max(0.0, min(1.0, confidence)),
                refined=True
            )
            
        except Exception as e:
            logger.error("Failed to parse refinement response", extra={
                "error": str(e)
            })
            # Return original with refined flag
            return GridCoordinate(
                cell=initial_coords.cell,
                offset_x=initial_coords.offset_x,
                offset_y=initial_coords.offset_y,
                confidence=initial_coords.confidence,
                refined=True
            )
    
    async def _validate_action(
        self,
        instruction: ActionInstruction,
        screenshot: bytes,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate if the action makes sense in current context.
        
        Args:
            instruction: Action instruction to validate
            screenshot: Current screenshot
            context: Test execution context
            
        Returns:
            Validation result with reasoning
        """
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
            
            return ValidationResult(
                valid=content.get("valid", False),
                confidence=float(content.get("confidence", 0.0)),
                reasoning=content.get("reasoning", "No reasoning provided"),
                concerns=content.get("concerns", []),
                suggestions=content.get("suggestions", [])
            )
        except Exception as e:
            logger.error("Failed to parse validation response", extra={"error": str(e)})
            return ValidationResult(
                valid=False,
                confidence=0.0,
                reasoning=f"Failed to validate: {str(e)}",
                concerns=["Validation parsing failed"],
                suggestions=[]
            )
    
    def _create_highlighted_screenshot(
        self,
        screenshot: bytes,
        grid_cell: str
    ) -> bytes:
        """
        Create a screenshot with the selected grid cell highlighted.
        
        Args:
            screenshot: Original screenshot
            grid_cell: Grid cell to highlight
            
        Returns:
            Screenshot with highlighted cell
        """
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
    
    async def _analyze_result(
        self,
        instruction: ActionInstruction,
        result: EnhancedActionResult
    ) -> AIAnalysis:
        """
        Analyze the result of the action execution.
        
        Args:
            instruction: Original action instruction
            result: Execution result with before/after states
            
        Returns:
            AI analysis of the action result
        """
        prompt = f"""Analyze the result of this action.

Action Performed:
- Type: {instruction.action_type.value}
- Target: {instruction.target or instruction.description}
- Expected: {instruction.expected_outcome}

Execution Details:
- Grid Cell: {result.coordinates.grid_cell}
- Execution Time: {result.execution.execution_time_ms}ms
- URL Change: {result.browser_state_before.url} → {result.browser_state_after.url}

Compare the before and after screenshots and provide analysis in JSON format:
{{
    "success": true/false,
    "confidence": 0.0-1.0,
    "actual_outcome": "What actually happened",
    "matches_expected": true/false,
    "ui_changes": ["List of observed UI changes"],
    "recommendations": ["Any recommendations for next steps"],
    "anomalies": ["Any unexpected behaviors detected"]
}}"""

        # Create comparison with before and after screenshots
        if result.browser_state_before.screenshot and result.browser_state_after.screenshot:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(result.browser_state_before.screenshot).decode('utf-8')}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(result.browser_state_after.screenshot).decode('utf-8')}",
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
                
                return AIAnalysis(
                    success=content.get("success", False),
                    confidence=float(content.get("confidence", 0.0)),
                    actual_outcome=content.get("actual_outcome", "Unknown"),
                    matches_expected=content.get("matches_expected", False),
                    ui_changes=content.get("ui_changes", []),
                    recommendations=content.get("recommendations", []),
                    anomalies=content.get("anomalies", [])
                )
            except Exception as e:
                logger.error("Failed to parse analysis response", extra={"error": str(e)})
        
        return AIAnalysis(
            success=result.execution.success,
            confidence=0.5,
            actual_outcome="No screenshot comparison available",
            matches_expected=False,
            ui_changes=[],
            recommendations=[],
            anomalies=[]
        )