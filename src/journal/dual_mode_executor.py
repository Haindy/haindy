"""
Dual-mode executor for scripted and visual test execution.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Tuple

from playwright.async_api import Page, Error as PlaywrightError

from src.browser.driver import BrowserDriver
from src.core.types import ActionType, TestStep
from src.journal.models import ActionRecord, ExecutionMode, JournalActionResult, ScriptedCommand
from src.journal.script_recorder import ScriptRecorder
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class DualModeExecutor:
    """
    Executes test actions in dual mode: scripted (fast) or visual (fallback).
    
    Attempts scripted execution first for speed, falls back to visual
    AI-driven execution when scripts fail.
    """
    
    def __init__(
        self,
        browser_driver: BrowserDriver,
        visual_executor = None  # Will be ActionAgent
    ):
        """
        Initialize the dual-mode executor.
        
        Args:
            browser_driver: Browser driver for scripted execution
            visual_executor: Visual execution agent (ActionAgent)
        """
        self.browser_driver = browser_driver
        self.visual_executor = visual_executor
        self.script_recorder = ScriptRecorder()
        
        # Execution statistics
        self.stats = {
            "scripted_attempts": 0,
            "scripted_successes": 0,
            "visual_fallbacks": 0,
            "total_time_saved_ms": 0
        }
    
    async def execute_action(
        self,
        step: TestStep,
        pattern: Optional[ActionRecord] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[JournalActionResult, ExecutionMode, int]:
        """
        Execute an action in dual mode.
        
        Args:
            step: Test step to execute
            pattern: Matched pattern if available
            context: Execution context
            
        Returns:
            Tuple of (action result, execution mode used, execution time)
        """
        start_time = time.time()
        
        # Try scripted execution first if pattern available
        if pattern and pattern.playwright_command:
            logger.debug(f"Attempting scripted execution for: {step.description}")
            
            result, success = await self._try_scripted_execution(
                pattern,
                context or {}
            )
            
            if success:
                execution_time = int((time.time() - start_time) * 1000)
                self.stats["scripted_successes"] += 1
                
                # Calculate time saved vs visual execution
                avg_visual_time = 2000  # Assume 2s for visual execution
                self.stats["total_time_saved_ms"] += max(0, avg_visual_time - execution_time)
                
                logger.info(f"Scripted execution successful in {execution_time}ms")
                return result, ExecutionMode.SCRIPTED, execution_time
            else:
                logger.warning("Scripted execution failed, falling back to visual")
                self.stats["visual_fallbacks"] += 1
        
        # Fall back to visual execution
        if self.visual_executor:
            logger.debug(f"Using visual execution for: {step.description}")
            
            # Take screenshot for visual analysis
            screenshot = await self.browser_driver.take_screenshot()
            
            # Get current URL
            page = await self._get_page()
            current_url = page.url if page else None
            
            # Execute visually
            result = await self.visual_executor.determine_action(
                screenshot=screenshot,
                instruction=step.action_instruction.description,
                current_url=current_url
            )
            
            # Perform the action
            if result.success and result.action:
                await self._perform_visual_action(result)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Record the action for future scripted execution
            if result.success:
                element_info = await self._extract_element_info(result)
                scripted_cmd = self.script_recorder.record_action(
                    step.action_instruction.action_type,
                    result,
                    element_info
                )
                
                if scripted_cmd:
                    result.playwright_command = scripted_cmd.command
                    result.selectors = {f"selector_{i}": sel for i, sel in enumerate(scripted_cmd.selectors)}
            
            return result, ExecutionMode.VISUAL, execution_time
        else:
            # No visual executor available
            logger.error("No visual executor available for fallback")
            return JournalActionResult(
                success=False,
                action=None,
                confidence=0.0,
                error_message="No execution method available"
            ), ExecutionMode.VISUAL, int((time.time() - start_time) * 1000)
    
    async def _try_scripted_execution(
        self,
        pattern: ActionRecord,
        context: Dict[str, Any]
    ) -> Tuple[JournalActionResult, bool]:
        """
        Try to execute a scripted command.
        
        Args:
            pattern: Action pattern with scripted command
            context: Execution context
            
        Returns:
            Tuple of (action result, success)
        """
        self.stats["scripted_attempts"] += 1
        
        page = await self._get_page()
        if not page:
            return JournalActionResult(
                success=False,
                error_message="No page available"
            ), False
        
        try:
            # Try primary command
            await self._execute_playwright_command(pattern.playwright_command, page)
            
            # Wait for any navigation or DOM updates
            await page.wait_for_load_state("networkidle", timeout=5000)
            
            return JournalActionResult(
                success=True,
                action=pattern.pattern_type,
                confidence=1.0,
                playwright_command=pattern.playwright_command,
                selectors=pattern.selectors
            ), True
            
        except PlaywrightError as e:
            logger.debug(f"Primary command failed: {e}")
            
            # Try fallback commands
            for fallback_cmd in pattern.fallback_commands:
                try:
                    await self._execute_playwright_command(fallback_cmd, page)
                    await page.wait_for_load_state("networkidle", timeout=5000)
                    
                    return JournalActionResult(
                        success=True,
                        action=pattern.pattern_type,
                        confidence=0.9,
                        playwright_command=fallback_cmd,
                        selectors=pattern.selectors
                    ), True
                    
                except PlaywrightError:
                    continue
            
            # All commands failed
            return JournalActionResult(
                success=False,
                error_message=f"Scripted execution failed: {str(e)}"
            ), False
    
    async def _execute_playwright_command(self, command: str, page: Page) -> None:
        """
        Execute a Playwright command string.
        
        Args:
            command: Command to execute
            page: Playwright page object
        """
        # Create a safe execution context
        exec_context = {
            "page": page,
            "asyncio": asyncio
        }
        
        # Execute the command
        # Note: In production, this should use a safer execution method
        exec(f"async def _cmd():\n    {command}\nasyncio.create_task(_cmd())", exec_context)
        await exec_context['_cmd']()
    
    async def _perform_visual_action(self, result: JournalActionResult) -> None:
        """
        Perform the action determined by visual analysis.
        
        Args:
            result: Action result from visual analysis
        """
        if result.action == ActionType.CLICK and result.coordinates:
            await self.browser_driver.click(
                result.coordinates[0],
                result.coordinates[1]
            )
        elif result.action == ActionType.TYPE and result.input_text:
            # Click first to focus
            if result.coordinates:
                await self.browser_driver.click(
                    result.coordinates[0],
                    result.coordinates[1]
                )
            # Then type
            page = await self._get_page()
            if page:
                await page.keyboard.type(result.input_text)
        elif result.action == ActionType.SCROLL:
            await self.browser_driver.scroll(
                direction="down",
                amount=300
            )
        elif result.action == ActionType.WAIT:
            await asyncio.sleep(1)
    
    async def _extract_element_info(self, result: JournalActionResult) -> Optional[Dict[str, Any]]:
        """
        Extract element information for script recording.
        
        Args:
            result: Action result
            
        Returns:
            Element information if available
        """
        if not result.coordinates:
            return None
        
        page = await self._get_page()
        if not page:
            return None
        
        try:
            # Get element at coordinates
            x, y = result.coordinates
            element = await page.evaluate(f'''
                (function() {{
                    const elem = document.elementFromPoint({x}, {y});
                    if (!elem) return null;
                    
                    return {{
                        tag: elem.tagName.toLowerCase(),
                        id: elem.id || null,
                        class: elem.className || null,
                        text: elem.textContent?.trim() || null,
                        role: elem.getAttribute('role') || null,
                        'aria-label': elem.getAttribute('aria-label') || null,
                        'data-testid': elem.getAttribute('data-testid') || null,
                        href: elem.href || null,
                        type: elem.type || null
                    }};
                }})()
            ''')
            
            return element
            
        except Exception as e:
            logger.error(f"Failed to extract element info: {e}")
            return None
    
    async def _get_page(self) -> Optional[Page]:
        """Get current Playwright page."""
        # This would be implemented based on browser driver internals
        # For now, return None as placeholder
        return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_attempts = self.stats["scripted_attempts"]
        
        return {
            "total_attempts": total_attempts,
            "scripted_success_rate": (
                self.stats["scripted_successes"] / total_attempts 
                if total_attempts > 0 else 0
            ),
            "visual_fallback_rate": (
                self.stats["visual_fallbacks"] / total_attempts
                if total_attempts > 0 else 0
            ),
            "time_saved_seconds": self.stats["total_time_saved_ms"] / 1000,
            "avg_time_saved_per_action_ms": (
                self.stats["total_time_saved_ms"] / self.stats["scripted_successes"]
                if self.stats["scripted_successes"] > 0 else 0
            )
        }