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
        Execute a complete action with multi-step workflow support.
        
        Routes to different workflows based on action type.
        
        Args:
            test_step: The test step to execute
            test_context: Context about the test plan and previous steps
            screenshot: Optional pre-captured screenshot
            
        Returns:
            Comprehensive action result with debugging information
        """
        logger.info("Executing action with multi-step workflow", extra={
            "step_number": test_step.step_number,
            "action_type": test_step.action_instruction.action_type.value,
            "description": test_step.description
        })
        
        action_type = test_step.action_instruction.action_type.value
        
        # Route to appropriate workflow based on action type
        if action_type == "navigate":
            return await self._execute_navigate_workflow(test_step, test_context)
        elif action_type == "click":
            # Phase 8c: Check if this is a dropdown/select action
            target_lower = test_step.action_instruction.target.lower()
            desc_lower = test_step.description.lower()
            if ("dropdown" in target_lower or "dropdown" in desc_lower or
                "select" in target_lower or "select" in desc_lower or
                "option" in target_lower or "option" in desc_lower):
                return await self._execute_dropdown_workflow(test_step, test_context, screenshot)
            else:
                return await self._execute_click_workflow(test_step, test_context, screenshot)
        elif action_type == "type":
            return await self._execute_type_workflow(test_step, test_context, screenshot)
        elif action_type == "assert":
            return await self._execute_assert_workflow(test_step, test_context, screenshot)
        else:
            # Unknown action type
            logger.warning(f"Unknown action type: {action_type}")
            return EnhancedActionResult(
                test_step_id=test_step.step_id,
                test_step=test_step,
                test_context=test_context,
                overall_success=False,
                execution=ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message=f"Unknown action type: {action_type}"
                ),
                failure_phase="routing",
                timestamp_end=datetime.now(timezone.utc)
            )
    
    async def _execute_navigate_workflow(
        self, test_step: TestStep, test_context: Dict[str, Any]
    ) -> EnhancedActionResult:
        """Execute navigation action workflow."""
        logger.info("Executing navigation workflow", extra={
            "target": test_step.action_instruction.target,
            "value": test_step.action_instruction.value
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
            ),
            timestamp_end=datetime.now(timezone.utc)
        )
        
        try:
            # Phase 8b: Enhanced URL extraction from multiple sources
            url = test_step.action_instruction.value
            
            # If URL not in value, try to get from test context
            if not url or "URL" in url:
                # Check if test_context has scenario info with URL
                if "test_scenario" in test_context:
                    scenario = test_context["test_scenario"]
                    if isinstance(scenario, dict) and "url" in scenario:
                        url = scenario["url"]
                        logger.info(f"Found URL in test_scenario: {url}")
                elif "url" in test_context:
                    url = test_context["url"]
                    logger.info(f"Found URL in test_context: {url}")
                
                # Still no URL? Try to infer from description
                if not url or "URL" in url:
                    desc_lower = test_step.description.lower()
                    target_lower = test_step.action_instruction.target.lower()
                    
                    if "wikipedia" in desc_lower or "wikipedia" in target_lower:
                        url = "https://en.wikipedia.org"
                        logger.info("Inferred Wikipedia URL from description")
                    else:
                        result.overall_success = False
                        result.execution = ExecutionResult(
                            success=False,
                            execution_time_ms=0.0,
                            error_message="No valid URL found in instruction value, test context, or description"
                        )
                        result.failure_phase = "url_extraction"
                        return result
            
            # Capture before state
            if self.browser_driver:
                result.browser_state_before = BrowserState(
                    url=await self.browser_driver.get_page_url(),
                    title=await self.browser_driver.get_page_title(),
                    viewport_size=await self.browser_driver.get_viewport_size(),
                    screenshot=await self.browser_driver.screenshot()
                )
            
            # Navigate
            execution_start = asyncio.get_event_loop().time()
            if not self.browser_driver:
                raise RuntimeError("Browser driver not initialized")
            await self.browser_driver.navigate(url)
            
            # Wait for page load
            await asyncio.sleep(2)
            
            # Capture after state
            screenshot_after = await self.browser_driver.screenshot()
            result.browser_state_after = BrowserState(
                url=await self.browser_driver.get_page_url(),
                title=await self.browser_driver.get_page_title(),
                viewport_size=await self.browser_driver.get_viewport_size(),
                screenshot=screenshot_after
            )
            
            # Phase 8b: Enhanced visual validation
            validation_prompt = f"""
I navigated to: {url}

Please analyze the screenshot and answer:

1. Did the navigation succeed? (Check for error pages, blank pages, connection errors)
2. What page am I looking at? Be specific about the content you see.
3. Does it match the expected outcome: "{test_step.action_instruction.expected_outcome}"?

Common error patterns to check for:
- 404 or "Page not found" errors
- Blank white pages
- Connection error messages
- Certificate warnings
- "Site can't be reached" messages
- Wrong domain/website

Expected outcome detail: {test_step.action_instruction.expected_outcome}

Respond in this EXACT format:
SUCCESS: true/false
PAGE_DESCRIPTION: <detailed description of what you see>
MATCHES_EXPECTED: true/false
ERROR_TYPE: none/404/blank/connection/wrong_site/other
REASON: <explanation of any issues>
"""
            
            # Use call_openai directly with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": validation_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(screenshot_after).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.call_openai(messages=messages, temperature=0.3)
            validation_response = response.get("content", "")
            
            # Phase 8b: Enhanced parsing of validation response
            success = "SUCCESS: true" in validation_response
            matches = "MATCHES_EXPECTED: true" in validation_response
            
            # Extract detailed information
            page_desc = ""
            error_type = "none"
            reason = ""
            
            if "PAGE_DESCRIPTION:" in validation_response:
                page_desc = validation_response.split("PAGE_DESCRIPTION:")[1].split("\n")[0].strip()
            
            if "ERROR_TYPE:" in validation_response:
                error_type = validation_response.split("ERROR_TYPE:")[1].split("\n")[0].strip()
            
            if "REASON:" in validation_response:
                reason = validation_response.split("REASON:")[1].strip()
            
            # Create detailed validation result
            validation_confidence = 1.0 if success and matches else 0.0
            validation_reasoning = f"{page_desc}"
            if not matches:
                validation_reasoning += f" | Error type: {error_type}"
                if reason:
                    validation_reasoning += f" | {reason}"
            
            # Set validation result
            result.validation = ValidationResult(
                valid=success and matches,
                confidence=validation_confidence,
                reasoning=validation_reasoning,
                concerns=[] if matches else [f"Navigation may have failed: {error_type}"],
                suggestions=[] if matches else ["Check URL validity", "Verify expected outcome description"]
            )
            
            # Set execution result with detailed error info
            error_msg = None
            if not success:
                error_msg = f"Navigation failed - {error_type}: {page_desc}"
            elif not matches:
                error_msg = f"Page loaded but doesn't match expected: {reason}"
            
            result.execution = ExecutionResult(
                success=success and matches,
                execution_time_ms=(asyncio.get_event_loop().time() - execution_start) * 1000,
                error_message=error_msg
            )
            
            # Create comprehensive AI analysis
            ui_changes = []
            anomalies = []
            recommendations = []
            
            if success:
                ui_changes.append(f"Navigated to {url}")
                ui_changes.append(f"Page loaded: {page_desc}")
            
            if not matches:
                anomalies.append(f"Expected: {test_step.action_instruction.expected_outcome}")
                anomalies.append(f"Actual: {page_desc}")
                
                if error_type == "404":
                    recommendations.append("Verify the URL is correct")
                    recommendations.append("Check if the page exists")
                elif error_type == "blank":
                    recommendations.append("Wait longer for page load")
                    recommendations.append("Check for JavaScript errors")
                elif error_type == "connection":
                    recommendations.append("Check internet connection")
                    recommendations.append("Verify the site is accessible")
                elif error_type == "wrong_site":
                    recommendations.append("Verify the URL in test scenario")
                    recommendations.append("Check for redirects")
            
            result.overall_success = success and matches
            result.ai_analysis = AIAnalysis(
                success=success and matches,
                confidence=validation_confidence,
                actual_outcome=page_desc,
                matches_expected=matches,
                ui_changes=ui_changes,
                recommendations=recommendations,
                anomalies=anomalies
            )
            
            if not result.overall_success:
                result.failure_phase = "validation"
            
        except Exception as e:
            logger.error(f"Navigation workflow failed: {str(e)}")
            result.overall_success = False
            result.execution = ExecutionResult(
                success=False,
                execution_time_ms=0.0,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
            result.failure_phase = "execution"
        
        return result
    
    async def _execute_click_workflow(
        self, test_step: TestStep, test_context: Dict[str, Any], screenshot: Optional[bytes]
    ) -> EnhancedActionResult:
        """Execute click action workflow with validation."""
        logger.info("Executing click workflow", extra={
            "target": test_step.action_instruction.target,
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
            ),
            timestamp_end=datetime.now(timezone.utc)
        )
        
        try:
            # Step 1: Capture initial state
            if not screenshot and self.browser_driver:
                screenshot = await self.browser_driver.screenshot()
            
            # Get viewport for grid initialization
            viewport_size = await self.browser_driver.get_viewport_size()
            self.grid_overlay.initialize(viewport_size[0], viewport_size[1])
            
            result.browser_state_before = BrowserState(
                url=await self.browser_driver.get_page_url(),
                title=await self.browser_driver.get_page_title(),
                viewport_size=viewport_size,
                screenshot=screenshot
            )
            
            # Step 2: Target Identification with AI
            click_prompt = f"""
I need to click on an element in this screenshot.

Target: {test_step.action_instruction.target}
Description: {test_step.description}
Expected outcome: {test_step.action_instruction.expected_outcome}

Please analyze the screenshot with the grid overlay and:
1. Find the clickable target element
2. Check if it's visible and not obscured
3. Identify the element type if possible (button, link, icon, etc.)
4. Provide the grid coordinates

Respond in this format:
TARGET_FOUND: true/false
VISIBLE: true/false
CLICKABLE: true/false
ELEMENT_TYPE: button/link/icon/form_element/other
GRID_CELL: <cell identifier>
OFFSET_X: <0.0-1.0>
OFFSET_Y: <0.0-1.0>
CONFIDENCE: <0.0-1.0>
REASONING: <why you selected this location>
BLOCKED_BY: <if blocked, what's blocking it>
"""
            
            # Create grid overlay screenshot for analysis
            grid_screenshot = self.grid_overlay.create_overlay_image(screenshot)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": click_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(grid_screenshot).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.call_openai(messages=messages, temperature=0.3)
            click_response = response.get("content", "")
            
            # Parse response
            target_found = "TARGET_FOUND: true" in click_response
            visible = "VISIBLE: true" in click_response
            clickable = "CLICKABLE: true" in click_response
            
            if not target_found:
                result.overall_success = False
                result.execution = ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message="Could not locate clickable element matching instruction"
                )
                result.failure_phase = "identification"
                return result
            
            if not visible or not clickable:
                blocked_by = ""
                if "BLOCKED_BY:" in click_response:
                    blocked_by = click_response.split("BLOCKED_BY:")[1].split("\n")[0].strip()
                result.overall_success = False
                result.execution = ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message=f"Target element is not clickable: {blocked_by or 'not visible'}"
                )
                result.failure_phase = "validation"
                return result
            
            # Extract coordinates and element type
            grid_cell = ""
            offset_x = 0.5
            offset_y = 0.5
            element_type = "other"
            confidence = 0.0
            reasoning = ""
            
            if "GRID_CELL:" in click_response:
                grid_cell = click_response.split("GRID_CELL:")[1].split("\n")[0].strip()
            if "OFFSET_X:" in click_response:
                offset_x = float(click_response.split("OFFSET_X:")[1].split("\n")[0].strip())
            if "OFFSET_Y:" in click_response:
                offset_y = float(click_response.split("OFFSET_Y:")[1].split("\n")[0].strip())
            if "ELEMENT_TYPE:" in click_response:
                element_type = click_response.split("ELEMENT_TYPE:")[1].split("\n")[0].strip()
            if "CONFIDENCE:" in click_response:
                confidence = float(click_response.split("CONFIDENCE:")[1].split("\n")[0].strip())
            if "REASONING:" in click_response:
                reasoning = click_response.split("REASONING:")[1].split("\n")[0].strip()
            
            # Create coordinate result
            # First create a GridCoordinate object
            grid_coord = GridCoordinate(
                cell=grid_cell,
                offset_x=offset_x,
                offset_y=offset_y,
                confidence=confidence,
                refined=False
            )
            # Convert to pixel coordinates
            pixel_x, pixel_y = self.grid_overlay.coordinate_to_pixels(grid_coord)
            
            result.coordinates = CoordinateResult(
                grid_cell=grid_cell,
                grid_coordinates=(pixel_x, pixel_y),
                offset_x=offset_x,
                offset_y=offset_y,
                confidence=confidence,
                reasoning=reasoning,
                refined=False
            )
            
            # Create grid screenshots for debugging
            result.grid_screenshot_before = grid_screenshot
            result.grid_screenshot_highlighted = self._create_highlighted_screenshot(
                screenshot, grid_cell
            )
            
            # Step 3: Click Execution
            execution_start = asyncio.get_event_loop().time()
            
            logger.info("Executing click", extra={
                "coordinates": coordinates,
                "element_type": element_type,
                "confidence": confidence
            })
            
            await self.browser_driver.click(coordinates[0], coordinates[1])
            
            # Wait based on element type (or use generic 1s as discussed)
            wait_time = 1000  # Default 1 second
            await self.browser_driver.wait(wait_time)
            
            # Step 4: Capture post-click state
            screenshot_after = await self.browser_driver.screenshot()
            result.browser_state_after = BrowserState(
                url=await self.browser_driver.get_page_url(),
                title=await self.browser_driver.get_page_title(),
                viewport_size=viewport_size,
                screenshot=screenshot_after
            )
            
            execution_time = (asyncio.get_event_loop().time() - execution_start) * 1000
            
            # Step 5: Click Validation with AI
            validation_prompt = f"""
I just clicked on the element at grid cell {grid_cell}.

Target was: {test_step.action_instruction.target}
Expected outcome: {test_step.action_instruction.expected_outcome}

Please compare the before and after screenshots and tell me:
1. What changed after the click?
2. Did the expected outcome occur?
3. Were there any unexpected changes?

Respond in this format:
CHANGES_DETECTED: true/false
EXPECTED_OUTCOME_MET: true/false
URL_CHANGED: true/false
NEW_URL: <if changed>
UI_CHANGES: <list what changed>
UNEXPECTED_CHANGES: <any unexpected behaviors>
CONFIDENCE: <0.0-1.0>
"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": validation_prompt},
                        {"type": "text", "text": "BEFORE screenshot:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}"
                            }
                        },
                        {"type": "text", "text": "AFTER screenshot:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(screenshot_after).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ]
            
            validation_response = await self.call_openai(messages=messages, temperature=0.3)
            validation_content = validation_response.get("content", "")
            
            # Parse validation
            changes_detected = "CHANGES_DETECTED: true" in validation_content
            expected_met = "EXPECTED_OUTCOME_MET: true" in validation_content
            url_changed = "URL_CHANGED: true" in validation_content
            
            ui_changes = []
            unexpected_changes = []
            validation_confidence = 0.0
            
            if "UI_CHANGES:" in validation_content:
                ui_changes_text = validation_content.split("UI_CHANGES:")[1].split("\n")[0].strip()
                ui_changes = [change.strip() for change in ui_changes_text.split(",") if change.strip()]
            
            if "UNEXPECTED_CHANGES:" in validation_content:
                unexpected_text = validation_content.split("UNEXPECTED_CHANGES:")[1].split("\n")[0].strip()
                if unexpected_text and unexpected_text.lower() not in ["none", "n/a"]:
                    unexpected_changes = [unexpected_text]
            
            if "CONFIDENCE:" in validation_content:
                validation_confidence = float(validation_content.split("CONFIDENCE:")[1].split("\n")[0].strip())
            
            # Determine success
            success = changes_detected and expected_met
            
            # Build error message if needed
            error_message = None
            if not changes_detected:
                error_message = "Click had no effect - no changes detected"
            elif not expected_met:
                error_message = f"Click executed but unexpected result: {', '.join(ui_changes)}"
            
            # Set results
            result.validation = ValidationResult(
                valid=True,  # Click was valid to attempt
                confidence=confidence,
                reasoning=f"Clicked {element_type} at {grid_cell}: {reasoning}"
            )
            
            result.execution = ExecutionResult(
                success=success,
                execution_time_ms=execution_time,
                error_message=error_message
            )
            
            result.ai_analysis = AIAnalysis(
                success=success,
                confidence=validation_confidence,
                actual_outcome=f"Clicked at {grid_cell}. Changes: {', '.join(ui_changes) if ui_changes else 'No visible changes'}",
                matches_expected=expected_met,
                ui_changes=ui_changes,
                recommendations=[] if success else ["Verify the correct element was clicked", "Check if page needs more time to load"],
                anomalies=unexpected_changes
            )
            
            result.overall_success = success
            if not success:
                result.failure_phase = "validation" if changes_detected else "execution"
            
        except Exception as e:
            logger.error(f"Click workflow failed: {str(e)}")
            result.overall_success = False
            result.execution = ExecutionResult(
                success=False,
                execution_time_ms=0.0,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
            result.failure_phase = "execution"
        
        return result
    
    async def _execute_type_workflow(
        self, test_step: TestStep, test_context: Dict[str, Any], screenshot: Optional[bytes]
    ) -> EnhancedActionResult:
        """Execute type action workflow with focus handling."""
        logger.info("Executing type workflow", extra={
            "target": test_step.action_instruction.target,
            "value": test_step.action_instruction.value
        })
        
        # For now, use the existing flow
        # This will be enhanced in future phases
        return await self._execute_legacy_workflow(test_step, test_context, screenshot)
    
    async def _execute_assert_workflow(
        self, test_step: TestStep, test_context: Dict[str, Any], screenshot: Optional[bytes]
    ) -> EnhancedActionResult:
        """Execute assert action workflow."""
        logger.info("Executing assert workflow", extra={
            "target": test_step.action_instruction.target
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
            ),
            timestamp_end=datetime.now(timezone.utc)
        )
        
        try:
            # Capture current state
            if not screenshot and self.browser_driver:
                screenshot = await self.browser_driver.screenshot()
            
            result.browser_state_before = BrowserState(
                url=await self.browser_driver.get_page_url(),
                title=await self.browser_driver.get_page_title(),
                viewport_size=await self.browser_driver.get_viewport_size(),
                screenshot=screenshot
            )
            
            # Ask AI to validate assertion
            assert_prompt = f"""
Please analyze this screenshot and verify:

Target: {test_step.action_instruction.target}
Description: {test_step.action_instruction.description}
Expected: {test_step.action_instruction.expected_outcome}

Questions:
1. Can you see what is described?
2. Does it match the expected outcome?

Respond in this format:
VISIBLE: true/false
MATCHES_EXPECTED: true/false
WHAT_I_SEE: <description>
REASON: <explanation if it doesn't match>
"""
            
            # Use call_openai directly with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": assert_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.call_openai(messages=messages, temperature=0.3)
            validation_response = response.get("content", "")
            
            # Parse validation
            visible = "VISIBLE: true" in validation_response
            matches = "MATCHES_EXPECTED: true" in validation_response
            
            # Extract what AI sees
            what_seen = ""
            if "WHAT_I_SEE:" in validation_response:
                what_seen = validation_response.split("WHAT_I_SEE:")[1].split("\n")[0].strip()
            
            # Set results
            result.validation = ValidationResult(
                valid=visible and matches,
                confidence=1.0 if visible and matches else 0.0,
                reasoning=what_seen
            )
            
            result.execution = ExecutionResult(
                success=visible and matches,
                execution_time_ms=0.0  # No action executed, just validation
            )
            
            result.overall_success = visible and matches
            result.ai_analysis = AIAnalysis(
                success=visible and matches,
                confidence=1.0 if visible and matches else 0.0,
                actual_outcome=what_seen,
                matches_expected=matches,
                ui_changes=[],
                recommendations=[],
                anomalies=[] if visible and matches else [f"Expected outcome not visible: {what_seen}"]
            )
            
            if not result.overall_success:
                result.failure_phase = "assertion"
            
        except Exception as e:
            logger.error(f"Assert workflow failed: {str(e)}")
            result.overall_success = False
            result.execution = ExecutionResult(
                success=False,
                execution_time_ms=0.0,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
            result.failure_phase = "execution"
        
        return result
    
    async def _execute_dropdown_workflow(
        self, test_step: TestStep, test_context: Dict[str, Any], screenshot: Optional[bytes]
    ) -> EnhancedActionResult:
        """Execute dropdown/select action workflow with multi-step interaction."""
        logger.info("Executing dropdown workflow", extra={
            "target": test_step.action_instruction.target,
            "value": test_step.action_instruction.value
        })
        
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
            if not screenshot:
                screenshot = await self.browser_driver.screenshot()
            
            # Step 1: Identify dropdown element
            dropdown_prompt = f"""
I need to interact with a dropdown/select element.

Target: {test_step.action_instruction.target}
Description: {test_step.action_instruction.description}
Value to select: {test_step.action_instruction.value}

Please analyze the screenshot and:
1. Identify the dropdown element
2. Check if it's currently open or closed
3. Provide grid coordinates to click on the dropdown
4. Check for any blockers (overlays, animations)

Respond in this format:
DROPDOWN_FOUND: true/false
DROPDOWN_STATE: open/closed
GRID_CELL: <cell identifier>
OFFSET_X: <0.0-1.0>
OFFSET_Y: <0.0-1.0>
BLOCKED: true/false
BLOCKER_REASON: <if blocked, explain why>
CONFIDENCE: <0.0-1.0>
"""
            
            # Get dropdown location
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": dropdown_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.call_openai(messages=messages, temperature=0.3)
            dropdown_response = response.get("content", "")
            
            # Parse response
            dropdown_found = "DROPDOWN_FOUND: true" in dropdown_response
            is_open = "DROPDOWN_STATE: open" in dropdown_response
            blocked = "BLOCKED: true" in dropdown_response
            
            if not dropdown_found:
                result.overall_success = False
                result.execution = ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message="Unable to locate dropdown element"
                )
                result.failure_phase = "identification"
                return result
            
            if blocked:
                blocker_reason = ""
                if "BLOCKER_REASON:" in dropdown_response:
                    blocker_reason = dropdown_response.split("BLOCKER_REASON:")[1].split("\n")[0].strip()
                result.overall_success = False
                result.execution = ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message=f"Dropdown is blocked: {blocker_reason}"
                )
                result.failure_phase = "validation"
                return result
            
            # Extract coordinates
            grid_cell = ""
            offset_x = 0.5
            offset_y = 0.5
            
            if "GRID_CELL:" in dropdown_response:
                grid_cell = dropdown_response.split("GRID_CELL:")[1].split("\n")[0].strip()
            if "OFFSET_X:" in dropdown_response:
                offset_x = float(dropdown_response.split("OFFSET_X:")[1].split("\n")[0].strip())
            if "OFFSET_Y:" in dropdown_response:
                offset_y = float(dropdown_response.split("OFFSET_Y:")[1].split("\n")[0].strip())
            
            # Step 2: Open dropdown if needed
            if not is_open:
                logger.info("Opening dropdown", extra={"grid_cell": grid_cell})
                
                # Click to open dropdown
                grid_coord = GridCoordinate(
                    cell=grid_cell,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    confidence=0.9,
                    refined=False
                )
                pixel_x, pixel_y = self.grid_overlay.coordinate_to_pixels(grid_coord)
                await self.browser_driver.click(pixel_x, pixel_y)
                await asyncio.sleep(0.5)  # Wait for dropdown to open
                
                # Take new screenshot
                screenshot_after_open = await self.browser_driver.screenshot()
                
                # Verify dropdown opened
                verify_prompt = "Is the dropdown now open? Respond with DROPDOWN_OPEN: true/false"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": verify_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64.b64encode(screenshot_after_open).decode('utf-8')}"
                                }
                            }
                        ]
                    }
                ]
                
                verify_response = await self.call_openai(messages=messages, temperature=0.3)
                if "DROPDOWN_OPEN: false" in verify_response.get("content", ""):
                    # Try one more time with a slight delay
                    await asyncio.sleep(0.5)
                    await self.browser_driver.click(pixel_x, pixel_y)
                    await asyncio.sleep(0.5)
                    screenshot_after_open = await self.browser_driver.screenshot()
            else:
                screenshot_after_open = screenshot
            
            # Step 3: Find and select target option
            target_value = test_step.action_instruction.value
            option_prompt = f"""
The dropdown is now open. I need to select: "{target_value}"

Please analyze the screenshot and:
1. Find the option that matches "{target_value}" (exact or close match)
2. If not visible, check if scrolling is needed
3. Provide grid coordinates for the option

Respond in this format:
OPTION_FOUND: true/false
OPTION_VISIBLE: true/false
NEEDS_SCROLL: true/false
SCROLL_DIRECTION: up/down/none
GRID_CELL: <cell identifier>
OFFSET_X: <0.0-1.0>
OFFSET_Y: <0.0-1.0>
CONFIDENCE: <0.0-1.0>
MATCH_TYPE: exact/partial/none
"""
            
            # Search for option
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": option_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(screenshot_after_open).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ]
            
            option_response = await self.call_openai(messages=messages, temperature=0.3)
            option_content = option_response.get("content", "")
            
            option_found = "OPTION_FOUND: true" in option_content
            option_visible = "OPTION_VISIBLE: true" in option_content
            needs_scroll = "NEEDS_SCROLL: true" in option_content
            
            # Handle scrolling if needed
            if needs_scroll and not option_visible:
                # For MVP, we'll try basic scrolling
                scroll_direction = "down"
                if "SCROLL_DIRECTION: up" in option_content:
                    scroll_direction = "up"
                
                # Find scroll area (simplified - click and drag in center of dropdown)
                if scroll_direction == "down":
                    # Drag from middle to top of dropdown area
                    start_y = pixel_y + 50
                    end_y = pixel_y - 50
                else:
                    # Drag from middle to bottom
                    start_y = pixel_y - 50
                    end_y = pixel_y + 50
                
                # Simple drag to scroll
                await self.browser_driver.click(pixel_x, start_y)
                await asyncio.sleep(0.1)
                # Note: We need drag support in browser driver for this
                # For now, we'll just try to find the option in the visible area
                logger.warning("Dropdown scrolling not fully implemented - drag support needed")
            
            if not option_found:
                result.overall_success = False
                result.execution = ExecutionResult(
                    success=False,
                    execution_time_ms=0.0,
                    error_message=f"Could not find option '{target_value}' in dropdown"
                )
                result.failure_phase = "option_search"
                return result
            
            # Extract option coordinates
            option_cell = ""
            option_x = 0.5
            option_y = 0.5
            
            if "GRID_CELL:" in option_content:
                parts = option_content.split("GRID_CELL:")
                if len(parts) > 1:
                    option_cell = parts[1].split("\n")[0].strip()
            if "OFFSET_X:" in option_content:
                parts = option_content.split("OFFSET_X:")
                if len(parts) > 1:
                    option_x = float(parts[1].split("\n")[0].strip())
            if "OFFSET_Y:" in option_content:
                parts = option_content.split("OFFSET_Y:")
                if len(parts) > 1:
                    option_y = float(parts[1].split("\n")[0].strip())
            
            # Step 4: Click on the option
            option_grid_coord = GridCoordinate(
                cell=option_cell,
                offset_x=option_x,
                offset_y=option_y,
                confidence=0.9,
                refined=False
            )
            option_coords = self.grid_overlay.coordinate_to_pixels(option_grid_coord)
            
            execution_start = asyncio.get_event_loop().time()
            await self.browser_driver.click(option_coords[0], option_coords[1])
            await asyncio.sleep(0.5)  # Wait for selection
            
            # Step 5: Validate selection
            screenshot_after = await self.browser_driver.screenshot()
            
            validate_prompt = f"""
I just selected "{target_value}" from a dropdown.

Please verify:
1. Is the dropdown now closed?
2. Does the dropdown show the selected value "{target_value}"?
3. Does it match the expected outcome: "{test_step.action_instruction.expected_outcome}"?

Respond in this format:
DROPDOWN_CLOSED: true/false
SELECTED_VALUE: <what the dropdown shows>
MATCHES_TARGET: true/false
MATCHES_EXPECTED: true/false
CONFIDENCE: <0.0-1.0>
"""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": validate_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(screenshot_after).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ]
            
            validate_response = await self.call_openai(messages=messages, temperature=0.3)
            validate_content = validate_response.get("content", "")
            
            # Parse validation
            dropdown_closed = "DROPDOWN_CLOSED: true" in validate_content
            matches_target = "MATCHES_TARGET: true" in validate_content
            matches_expected = "MATCHES_EXPECTED: true" in validate_content
            
            selected_value = ""
            if "SELECTED_VALUE:" in validate_content:
                selected_value = validate_content.split("SELECTED_VALUE:")[1].split("\n")[0].strip()
            
            # Set results
            success = matches_target and matches_expected
            
            result.coordinates = CoordinateResult(
                grid_cell=option_cell,
                grid_coordinates=(option_coords[0], option_coords[1]),
                offset_x=option_x,
                offset_y=option_y,
                confidence=0.9 if success else 0.5,
                reasoning=f"Selected '{selected_value}' from dropdown"
            )
            
            result.validation = ValidationResult(
                valid=success,
                confidence=0.9 if success else 0.5,
                reasoning=f"Dropdown selection: {selected_value}",
                concerns=[] if success else [f"Selected value '{selected_value}' may not match target '{target_value}'"]
            )
            
            result.execution = ExecutionResult(
                success=success,
                execution_time_ms=(asyncio.get_event_loop().time() - execution_start) * 1000,
                error_message=None if success else f"Selection mismatch: expected '{target_value}', got '{selected_value}'"
            )
            
            result.browser_state_before = BrowserState(
                url=await self.browser_driver.get_page_url(),
                title=await self.browser_driver.get_page_title(),
                viewport_size=await self.browser_driver.get_viewport_size(),
                screenshot=screenshot
            )
            
            result.browser_state_after = BrowserState(
                url=await self.browser_driver.get_page_url(),
                title=await self.browser_driver.get_page_title(),
                viewport_size=await self.browser_driver.get_viewport_size(),
                screenshot=screenshot_after
            )
            
            result.ai_analysis = AIAnalysis(
                success=success,
                confidence=0.9 if success else 0.5,
                actual_outcome=f"Selected '{selected_value}' from dropdown",
                matches_expected=matches_expected,
                ui_changes=[f"Dropdown value changed to: {selected_value}"],
                recommendations=[] if success else ["Verify the target value matches available options", "Check for case sensitivity"],
                anomalies=[] if success else [f"Selection mismatch: '{selected_value}' vs '{target_value}'"]
            )
            
            result.overall_success = success
            if not success:
                result.failure_phase = "validation"
            
        except Exception as e:
            logger.error(f"Dropdown workflow failed: {str(e)}")
            result.overall_success = False
            result.execution = ExecutionResult(
                success=False,
                execution_time_ms=0.0,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
            result.failure_phase = "execution"
        
        return result
    
    async def _execute_legacy_workflow(
        self, test_step: TestStep, test_context: Dict[str, Any], screenshot: Optional[bytes]
    ) -> EnhancedActionResult:
        """Execute the legacy workflow for click/type actions."""
        # This is the existing execute_action logic that was replaced
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
                        await self._execute_click_with_focus(x, y)
                    elif test_step.action_instruction.action_type.value == "type":
                        # Enhanced typing with focus validation
                        if test_step.action_instruction.value:
                            success = await self._execute_type_with_focus(
                                x, y, 
                                test_step.action_instruction.value,
                                test_step.action_instruction.target
                            )
                            if not success:
                                raise Exception("Failed to type text - element not focusable")
                    
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
        
        # Add special validation for type actions
        if instruction.action_type.value == "type":
            prompt += """
5. IMPORTANT FOR TYPING: Is the target element a focusable text input field?
   - Look for: <input>, <textarea>, or contenteditable elements
   - Check if it appears to be an interactive text field
   - Verify it's not disabled or read-only"""

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
    
    async def _execute_click_with_focus(self, x: int, y: int) -> None:
        """
        Execute a click with enhanced focus handling.
        
        This method implements multiple click strategies to ensure proper focus.
        """
        logger.debug("Executing enhanced click with focus", extra={"x": x, "y": y})
        
        # Strategy 1: Single click
        await self.browser_driver.click(x, y)
        await self.browser_driver.wait(100)
        
        # Strategy 2: Double-click for stubborn elements
        # Some elements require double-click to properly focus
        # We'll use this selectively based on validation
    
    async def _execute_type_with_focus(
        self, 
        x: int, 
        y: int, 
        text: str,
        target_description: Optional[str] = None
    ) -> bool:
        """
        Execute typing with enhanced focus validation and retry strategies.
        
        Args:
            x: X coordinate to click
            y: Y coordinate to click
            text: Text to type
            target_description: Description of the target element
            
        Returns:
            True if typing was successful, False otherwise
        """
        logger.info("Executing enhanced typing with focus validation", extra={
            "x": x,
            "y": y,
            "text_length": len(text),
            "target": target_description
        })
        
        # Strategy 1: Single click and type
        await self.browser_driver.click(x, y)
        await self.browser_driver.wait(200)
        
        # Validate focus by checking if we can type
        focus_validated = await self._validate_focus_for_typing(x, y, target_description)
        
        if focus_validated:
            await self.browser_driver.type_text(text)
            await self.browser_driver.wait(500)
            return True
        
        # Strategy 2: Double-click to focus
        logger.warning("Single click didn't achieve focus, trying double-click")
        await self.browser_driver.click(x, y)
        await self.browser_driver.wait(50)
        await self.browser_driver.click(x, y)
        await self.browser_driver.wait(200)
        
        focus_validated = await self._validate_focus_for_typing(x, y, target_description)
        
        if focus_validated:
            await self.browser_driver.type_text(text)
            await self.browser_driver.wait(500)
            return True
        
        # Strategy 3: Click and wait longer
        logger.warning("Double-click didn't achieve focus, trying click with longer wait")
        await self.browser_driver.click(x, y)
        await self.browser_driver.wait(1000)  # Wait longer for JavaScript to load
        
        focus_validated = await self._validate_focus_for_typing(x, y, target_description)
        
        if focus_validated:
            await self.browser_driver.type_text(text)
            await self.browser_driver.wait(500)
            return True
        
        logger.error("Failed to achieve focus after multiple strategies")
        return False
    
    async def _validate_focus_for_typing(
        self, 
        x: int, 
        y: int,
        target_description: Optional[str] = None
    ) -> bool:
        """
        Validate that an element is focused and ready for typing.
        
        This method uses AI to analyze the current state and determine if
        the target element is properly focused.
        """
        # Take a screenshot to analyze current state
        screenshot = await self.browser_driver.screenshot()
        
        # Create a highlighted screenshot showing where we clicked
        highlighted_screenshot = self._create_click_highlight_screenshot(screenshot, x, y)
        
        prompt = f"""Analyze this screenshot to determine if a text input element is currently focused and ready for typing.

I just clicked at the marked location (red circle). {f'The target is: {target_description}' if target_description else ''}

Please examine:
1. Is there a visible text cursor (blinking line) in an input field?
2. Is there a focus outline/border around an input element?
3. Is the clicked area an actual text input field (input, textarea, contenteditable)?
4. Are there any visual indicators that the element is ready to receive text?

Respond in JSON format:
{{
    "is_focused": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "element_type": "input/textarea/contenteditable/other/none",
    "visual_indicators": ["list of indicators seen"]
}}"""

        try:
            # Convert screenshot to base64
            base64_image = base64.b64encode(highlighted_screenshot).decode('utf-8')
            
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
                temperature=0.1
            )
            
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
            
            is_focused = content.get("is_focused", False)
            confidence = float(content.get("confidence", 0.0))
            reasoning = content.get("reasoning", "")
            
            logger.debug("Focus validation result", extra={
                "is_focused": is_focused,
                "confidence": confidence,
                "reasoning": reasoning,
                "element_type": content.get("element_type", "unknown")
            })
            
            # Consider it focused if confidence is high enough
            return is_focused and confidence >= 0.7
            
        except Exception as e:
            logger.error("Failed to validate focus", extra={"error": str(e)})
            # Assume focused to allow typing to proceed
            return True
    
    def _create_click_highlight_screenshot(self, screenshot: bytes, x: int, y: int) -> bytes:
        """Create a screenshot with a highlight showing where we clicked."""
        from PIL import Image, ImageDraw
        import io
        
        # Open the screenshot
        image = Image.open(io.BytesIO(screenshot))
        draw = ImageDraw.Draw(image)
        
        # Draw a red circle at the click location
        radius = 15
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline="red",
            width=3
        )
        
        # Draw a crosshair
        draw.line([x - radius, y, x + radius, y], fill="red", width=2)
        draw.line([x, y - radius, x, y + radius], fill="red", width=2)
        
        # Convert back to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()