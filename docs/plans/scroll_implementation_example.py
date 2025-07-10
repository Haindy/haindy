# Scroll to Element Implementation Example
# This shows how iterative scrolling would be integrated into ActionAgent

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
import asyncio

class ScrollDirection(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

class VisibilityStatus(Enum):
    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_VISIBLE = "partially_visible"  
    NOT_VISIBLE = "not_visible"

@dataclass
class VisibilityResult:
    status: VisibilityStatus
    coordinates: Optional[str] = None  # Grid coordinates like "M23"
    confidence: float = 0.0
    visible_percentage: Optional[int] = None  # For partial visibility
    suggested_direction: Optional[ScrollDirection] = None
    direction_confidence: float = 0.0
    ai_notes: str = ""  # Additional context from AI

@dataclass
class ScrollAction:
    direction: ScrollDirection
    distance: int  # pixels
    is_correction: bool = False
    executed_at: Optional[float] = None

@dataclass
class ScrollState:
    target_element: str
    attempts: int = 0
    max_attempts: int = 15
    scroll_history: List[ScrollAction] = None
    last_direction: Optional[ScrollDirection] = None
    overshoot_detected: bool = False
    element_partially_visible: bool = False
    last_screenshot_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.scroll_history is None:
            self.scroll_history = []

class ActionAgent:
    """Extended ActionAgent with scroll capabilities"""
    
    async def _scroll_to_element_workflow(self, screenshot: bytes, instruction: str):
        """
        Main entry point for scroll-to-element action.
        Iteratively scrolls until element is found or max attempts reached.
        """
        
        state = ScrollState(target_element=instruction)
        
        self.logger.info(
            "Starting scroll-to-element workflow",
            extra={
                "target": instruction,
                "max_attempts": state.max_attempts
            }
        )
        
        while state.attempts < state.max_attempts:
            state.attempts += 1
            
            # Check current visibility
            visibility = await self._check_element_visibility(screenshot, instruction, state)
            
            self.logger.info(
                f"Visibility check attempt {state.attempts}",
                extra={
                    "status": visibility.status.value,
                    "confidence": visibility.confidence,
                    "coordinates": visibility.coordinates
                }
            )
            
            # Success case - element fully visible
            if visibility.status == VisibilityStatus.FULLY_VISIBLE and visibility.confidence > 80:
                return {
                    "action": "scroll_to_element",
                    "success": True,
                    "coordinates": visibility.coordinates,
                    "confidence": visibility.confidence,
                    "attempts": state.attempts,
                    "total_scroll_distance": sum(abs(s.distance) for s in state.scroll_history)
                }
            
            # Plan next scroll action
            scroll_action = await self._plan_scroll_action(state, visibility)
            
            if not scroll_action:
                # AI couldn't determine scroll direction
                return {
                    "action": "scroll_to_element", 
                    "success": False,
                    "error": "Could not determine scroll direction",
                    "attempts": state.attempts
                }
            
            # Execute scroll
            await self._execute_scroll(scroll_action)
            state.scroll_history.append(scroll_action)
            state.last_direction = scroll_action.direction
            
            # Wait for scroll animation and any dynamic content
            await asyncio.sleep(0.8)
            
            # Capture new screenshot
            screenshot = await self.browser_driver.screenshot()
            
            # Check if we're making progress
            if not await self._is_making_progress(state, screenshot):
                self.logger.warning("No progress detected, adjusting strategy")
                # Could implement alternative strategies here
        
        # Max attempts reached
        return {
            "action": "scroll_to_element",
            "success": False,
            "error": f"Element not found after {state.max_attempts} attempts",
            "scroll_history": [
                {
                    "direction": s.direction.value,
                    "distance": s.distance,
                    "is_correction": s.is_correction
                } 
                for s in state.scroll_history
            ]
        }
    
    async def _check_element_visibility(
        self, 
        screenshot: bytes, 
        target: str, 
        state: ScrollState
    ) -> VisibilityResult:
        """
        Use AI to check if target element is visible in screenshot.
        Handles full, partial, and not visible cases.
        """
        
        # Craft context-aware prompt
        context = self._build_visibility_context(state)
        
        prompt = f"""
        Analyze this screenshot with a 60x60 grid overlay.
        
        Target element: "{target}"
        
        Determine the visibility of the target element:
        
        1. If FULLY VISIBLE (entire element is visible and clickable):
           - Provide exact grid coordinates (e.g., "M23")
           - Confidence score (0-100)
           
        2. If PARTIALLY VISIBLE (only part is visible):
           - Indicate visible portion (top/bottom/left/right edge)
           - Estimate percentage visible
           - Suggest scroll direction to reveal fully
           
        3. If NOT VISIBLE:
           - Based on current page content, suggest scroll direction
           - Consider these patterns:
             * Headers/navigation → scroll UP
             * Footers/submit buttons → scroll DOWN  
             * Next/continue buttons → usually DOWN
             * Previous/back → usually UP
           - Provide confidence in direction (0-100)
        
        {context}
        
        Respond in this format:
        STATUS: [FULLY_VISIBLE|PARTIALLY_VISIBLE|NOT_VISIBLE]
        COORDINATES: [grid coords if visible, e.g. M23]
        CONFIDENCE: [0-100]
        VISIBLE_PERCENT: [if partial, e.g. 30]
        DIRECTION: [UP|DOWN|LEFT|RIGHT if not fully visible]
        DIRECTION_CONFIDENCE: [0-100]
        NOTES: [any relevant observations]
        """
        
        ai_response = await self._call_ai_with_vision(screenshot, prompt)
        return self._parse_visibility_response(ai_response)
    
    def _build_visibility_context(self, state: ScrollState) -> str:
        """Build context from scroll history for better AI decisions."""
        
        if not state.scroll_history:
            return ""
        
        context_parts = ["Previous scroll attempts:"]
        
        # Summarize recent history
        recent_scrolls = state.scroll_history[-3:]  # Last 3 scrolls
        for i, scroll in enumerate(recent_scrolls):
            context_parts.append(
                f"- Scrolled {scroll.direction.value} {scroll.distance}px"
            )
        
        # Add warnings about patterns
        if self._is_oscillating(state.scroll_history):
            context_parts.append(
                "WARNING: Oscillating pattern detected - element might be in the middle"
            )
        
        return "\n".join(context_parts)
    
    async def _plan_scroll_action(
        self, 
        state: ScrollState, 
        visibility: VisibilityResult
    ) -> Optional[ScrollAction]:
        """
        Plan the next scroll action based on current visibility and history.
        Implements intelligent scroll distance calculation.
        """
        
        # Handle overshoot correction
        if self._detect_overshoot(state, visibility):
            self.logger.info("Overshoot detected, planning correction")
            return self._create_correction_scroll(state)
        
        # Handle partial visibility - fine-tune positioning  
        if visibility.status == VisibilityStatus.PARTIALLY_VISIBLE:
            return self._create_fine_tune_scroll(visibility)
        
        # Handle not visible - calculate smart scroll distance
        if visibility.status == VisibilityStatus.NOT_VISIBLE:
            if not visibility.suggested_direction:
                return None
                
            distance = self._calculate_scroll_distance(state, visibility)
            
            return ScrollAction(
                direction=visibility.suggested_direction,
                distance=distance,
                is_correction=False
            )
        
        return None
    
    def _calculate_scroll_distance(
        self, 
        state: ScrollState, 
        visibility: VisibilityResult
    ) -> int:
        """
        Calculate optimal scroll distance based on confidence and history.
        Implements convergence to avoid overshooting.
        """
        
        # Base distances for different confidence levels
        if visibility.direction_confidence > 90:
            base_distance = 600  # Very confident - bigger jumps
        elif visibility.direction_confidence > 70:
            base_distance = 400  # Confident - medium jumps
        else:
            base_distance = 200  # Less confident - smaller jumps
        
        # Reduce distance as attempts increase (convergence)
        attempt_factor = max(0.3, 1.0 - (state.attempts * 0.1))
        
        # Further reduce if we've been oscillating
        if self._is_oscillating(state.scroll_history):
            attempt_factor *= 0.5
            self.logger.debug("Oscillation detected, reducing scroll distance")
        
        final_distance = int(base_distance * attempt_factor)
        
        # Ensure minimum scroll distance
        return max(100, final_distance)
    
    def _create_fine_tune_scroll(self, visibility: VisibilityResult) -> ScrollAction:
        """Create small scroll action for fine-tuning when element is partially visible."""
        
        # Small distances for fine-tuning
        if visibility.visible_percentage and visibility.visible_percentage > 70:
            distance = 50  # Very small adjustment
        elif visibility.visible_percentage and visibility.visible_percentage > 40:
            distance = 100  # Small adjustment
        else:
            distance = 200  # Medium adjustment
        
        return ScrollAction(
            direction=visibility.suggested_direction,
            distance=distance,
            is_correction=False
        )
    
    def _detect_overshoot(self, state: ScrollState, visibility: VisibilityResult) -> bool:
        """Detect if we've scrolled past the target element."""
        
        if state.attempts < 2:
            return False
        
        # Was partially visible, now not visible = likely overshot
        if state.element_partially_visible and visibility.status == VisibilityStatus.NOT_VISIBLE:
            return True
        
        # Direction reversal with high confidence = likely overshot
        if (state.last_direction and 
            visibility.suggested_direction and
            self._is_opposite_direction(state.last_direction, visibility.suggested_direction) and
            visibility.direction_confidence > 80):
            return True
        
        return False
    
    def _create_correction_scroll(self, state: ScrollState) -> ScrollAction:
        """Create a corrective scroll action after overshoot."""
        
        last_scroll = state.scroll_history[-1]
        opposite_dir = self._get_opposite_direction(last_scroll.direction)
        
        # Scroll back partial distance of last scroll
        correction_distance = last_scroll.distance // 3
        
        return ScrollAction(
            direction=opposite_dir,
            distance=max(50, correction_distance),  # Minimum 50px
            is_correction=True
        )
    
    def _is_oscillating(self, history: List[ScrollAction], window: int = 4) -> bool:
        """Check if scroll history shows oscillating pattern."""
        
        if len(history) < window:
            return False
        
        recent = history[-window:]
        directions = [s.direction for s in recent]
        
        # Check for alternating directions
        changes = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1]:
                changes += 1
        
        return changes >= window - 1
    
    def _is_opposite_direction(self, dir1: ScrollDirection, dir2: ScrollDirection) -> bool:
        """Check if two directions are opposite."""
        opposites = {
            ScrollDirection.UP: ScrollDirection.DOWN,
            ScrollDirection.DOWN: ScrollDirection.UP,
            ScrollDirection.LEFT: ScrollDirection.RIGHT,
            ScrollDirection.RIGHT: ScrollDirection.LEFT
        }
        return opposites.get(dir1) == dir2
    
    def _get_opposite_direction(self, direction: ScrollDirection) -> ScrollDirection:
        """Get the opposite scroll direction."""
        opposites = {
            ScrollDirection.UP: ScrollDirection.DOWN,
            ScrollDirection.DOWN: ScrollDirection.UP,
            ScrollDirection.LEFT: ScrollDirection.RIGHT,
            ScrollDirection.RIGHT: ScrollDirection.LEFT
        }
        return opposites[direction]
    
    async def _execute_scroll(self, action: ScrollAction):
        """Execute the actual scroll action in the browser."""
        
        x, y = 0, 0
        
        if action.direction == ScrollDirection.DOWN:
            y = action.distance
        elif action.direction == ScrollDirection.UP:
            y = -action.distance
        elif action.direction == ScrollDirection.RIGHT:
            x = action.distance
        elif action.direction == ScrollDirection.LEFT:
            x = -action.distance
        
        self.logger.debug(
            f"Executing scroll: {action.direction.value} by {action.distance}px",
            extra={"x": x, "y": y, "is_correction": action.is_correction}
        )
        
        # Use smooth scrolling for better UX
        await self.browser_driver.page.evaluate(
            f"""
            window.scrollBy({{
                left: {x},
                top: {y},
                behavior: 'smooth'
            }});
            """
        )
        
        # Record execution time
        action.executed_at = asyncio.get_event_loop().time()
    
    async def _is_making_progress(self, state: ScrollState, screenshot: bytes) -> bool:
        """Check if scrolling is having an effect (not stuck)."""
        
        # Simple hash comparison to detect identical screenshots
        current_hash = self._hash_screenshot(screenshot)
        
        if state.last_screenshot_hash == current_hash:
            # Screenshot hasn't changed - might be at page boundary
            return False
        
        state.last_screenshot_hash = current_hash
        return True
    
    def _parse_visibility_response(self, ai_response: str) -> VisibilityResult:
        """Parse AI response into VisibilityResult object."""
        
        # Parse the structured response
        lines = ai_response.strip().split('\n')
        result = VisibilityResult(status=VisibilityStatus.NOT_VISIBLE)
        
        for line in lines:
            if line.startswith('STATUS:'):
                status_str = line.split(':', 1)[1].strip()
                result.status = VisibilityStatus[status_str]
            elif line.startswith('COORDINATES:'):
                coords = line.split(':', 1)[1].strip()
                if coords and coords != 'None':
                    result.coordinates = coords
            elif line.startswith('CONFIDENCE:'):
                result.confidence = float(line.split(':', 1)[1].strip())
            elif line.startswith('VISIBLE_PERCENT:'):
                pct = line.split(':', 1)[1].strip()
                if pct and pct != 'None':
                    result.visible_percentage = int(pct)
            elif line.startswith('DIRECTION:'):
                dir_str = line.split(':', 1)[1].strip()
                if dir_str in [d.name for d in ScrollDirection]:
                    result.suggested_direction = ScrollDirection[dir_str]
            elif line.startswith('DIRECTION_CONFIDENCE:'):
                result.direction_confidence = float(line.split(':', 1)[1].strip())
            elif line.startswith('NOTES:'):
                result.ai_notes = line.split(':', 1)[1].strip()
        
        return result