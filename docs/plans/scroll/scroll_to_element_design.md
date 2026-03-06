# Iterative Scroll-to-Element Design

## Overview

The iterative scroll-to-element mechanism enables HAINDY to navigate to any element on a page through AI-guided visual scrolling, without relying on DOM selectors.

## Core Algorithm

### Phase 1: Initial Assessment

```python
async def _scroll_to_element_workflow(self, screenshot: bytes, instruction: str):
    """
    Iteratively scroll until target element is found and properly positioned.
    
    Args:
        screenshot: Initial page screenshot
        instruction: Description of target element (e.g., "Submit button", "Contact Us section")
    """
    
    # Initialize scroll state
    scroll_state = ScrollState(
        target_element=instruction,
        attempts=0,
        max_attempts=15,
        scroll_history=[],
        last_direction=None,
        overshoot_detected=False,
        element_partially_visible=False
    )
    
    # Phase 1: Initial visibility check
    visibility_result = await self._check_element_visibility(screenshot, instruction)
    
    if visibility_result.fully_visible:
        return ScrollResult(
            success=True,
            final_coordinates=visibility_result.coordinates,
            attempts=1,
            total_scroll_distance=0
        )
```

### Phase 2: Scroll Direction Determination

```python
    # Phase 2: Determine initial scroll direction
    if visibility_result.not_visible:
        scroll_direction = await self._determine_scroll_direction(
            screenshot, 
            instruction,
            context_clues=self._extract_context_clues(instruction)
        )
```

## AI Visibility Detection Strategy

### Multi-Level Visibility Check

The AI evaluates element visibility across three categories:

1. **Fully Visible**: Element completely within viewport, unobstructed
2. **Partially Visible**: Part of element visible (edge cases)
3. **Not Visible**: Element not in current viewport

### Visibility Check Prompt Template

```python
VISIBILITY_CHECK_PROMPT = """
Analyze this screenshot with a {grid_size} grid overlay.

Target element: {target_description}

Please determine the visibility status of the target element:

1. FULLY_VISIBLE: The entire element is visible and clickable
   - Provide grid coordinates (e.g., "M23")
   - Confidence score (0-100)

2. PARTIALLY_VISIBLE: Only part of the element is visible
   - Indicate which part is visible (top, bottom, left, right)
   - Estimate percentage visible (e.g., "30% visible at bottom")
   - Suggest scroll direction to reveal fully

3. NOT_VISIBLE: Element is not in the current viewport
   - Based on page context, suggest most likely direction:
     - UP: Element likely above current view
     - DOWN: Element likely below current view
     - LEFT: Element likely to the left
     - RIGHT: Element likely to the right
   - Provide reasoning for your suggestion

Context analysis:
- Current page section/content
- Navigation patterns (headers usually up, footers down)
- Form flow (submit buttons typically at bottom)
- Menu structures (navigation often at top or left)
"""
```

## Intelligent Scroll Amount Calculation

### Dynamic Scroll Distance

Instead of fixed pixel amounts, calculate scroll distance based on:

1. **Confidence in Direction**: Higher confidence = larger initial scroll
2. **Previous Attempts**: Reduce scroll amount as we get closer
3. **Partial Visibility**: Fine-tune when element is partially visible

```python
def _calculate_scroll_amount(self, state: ScrollState, visibility: VisibilityResult) -> int:
    """Calculate optimal scroll distance based on current state."""
    
    base_scroll = 400  # Base scroll amount in pixels
    
    # Adjust based on visibility
    if visibility.partially_visible:
        # Fine scrolling when element is partially visible
        if visibility.visible_percentage < 50:
            return base_scroll // 2  # 200px
        else:
            return base_scroll // 4  # 100px - fine tuning
    
    # Adjust based on confidence
    confidence_multiplier = visibility.direction_confidence / 100
    
    # Reduce scroll amount after multiple attempts (convergence)
    attempt_reduction = max(0.3, 1 - (state.attempts * 0.1))
    
    # Detect oscillation and reduce amount
    if self._is_oscillating(state.scroll_history):
        attempt_reduction *= 0.5
    
    final_amount = int(base_scroll * confidence_multiplier * attempt_reduction)
    return max(50, final_amount)  # Minimum 50px scroll
```

## Overshoot Detection and Correction

### Detecting Overshoot

```python
async def _detect_overshoot(self, state: ScrollState, current_visibility: VisibilityResult) -> bool:
    """Detect if we've scrolled past the target element."""
    
    # Case 1: Element was partially visible, now not visible
    if state.element_partially_visible and current_visibility.not_visible:
        return True
    
    # Case 2: Scroll direction reversal with high confidence
    if state.last_direction and current_visibility.suggested_direction:
        if self._is_opposite_direction(state.last_direction, current_visibility.suggested_direction):
            if current_visibility.direction_confidence > 80:
                return True
    
    # Case 3: AI explicitly indicates overshoot
    if "past" in current_visibility.ai_notes or "overshot" in current_visibility.ai_notes:
        return True
    
    return False
```

### Correction Strategy

```python
async def _correct_overshoot(self, state: ScrollState) -> ScrollAction:
    """Generate corrective scroll action after overshoot."""
    
    # Reverse direction
    reverse_direction = self._get_opposite_direction(state.last_direction)
    
    # Use smaller amount for correction
    last_scroll_amount = abs(state.scroll_history[-1].distance)
    correction_amount = last_scroll_amount // 3  # Scroll back 1/3 of last distance
    
    return ScrollAction(
        direction=reverse_direction,
        distance=correction_amount,
        is_correction=True
    )
```

## Complete Workflow Integration

### Main Scroll Loop

```python
async def _scroll_to_element_workflow(self, screenshot: bytes, instruction: str):
    state = ScrollState(target_element=instruction)
    
    while state.attempts < state.max_attempts:
        state.attempts += 1
        
        # Check current visibility
        visibility = await self._check_element_visibility(screenshot, instruction)
        
        # Success case
        if visibility.fully_visible:
            return ScrollResult(success=True, coordinates=visibility.coordinates)
        
        # Update state
        state.element_partially_visible = visibility.partially_visible
        
        # Detect overshoot
        if state.attempts > 1 and await self._detect_overshoot(state, visibility):
            state.overshoot_detected = True
            scroll_action = await self._correct_overshoot(state)
        else:
            # Calculate next scroll
            scroll_action = await self._plan_next_scroll(state, visibility)
        
        # Execute scroll
        await self._execute_scroll(scroll_action)
        
        # Update history
        state.scroll_history.append(scroll_action)
        state.last_direction = scroll_action.direction
        
        # Capture new screenshot
        await asyncio.sleep(0.5)  # Allow scroll animation
        screenshot = await self.browser_driver.screenshot()
        
        # Check for infinite scroll or dynamic loading
        if await self._check_dynamic_content_loaded(screenshot, state):
            await asyncio.sleep(1.0)  # Extra wait for content
            screenshot = await self.browser_driver.screenshot()
    
    # Max attempts reached
    return ScrollResult(
        success=False,
        error="Could not locate element after maximum scroll attempts"
    )
```

## Handling Edge Cases

### 1. Sticky Headers/Footers

```python
async def _adjust_for_sticky_elements(self, visibility: VisibilityResult) -> VisibilityResult:
    """Adjust coordinates if sticky elements obstruct the target."""
    
    sticky_prompt = """
    Is the target element obscured by any sticky/fixed elements like:
    - Fixed headers
    - Sticky navigation bars  
    - Cookie banners
    - Floating action buttons
    
    If yes, provide adjusted grid coordinates for the visible/clickable area.
    """
    
    # Re-evaluate with sticky element consideration
    adjusted = await self._call_ai_with_screenshot(screenshot, sticky_prompt)
    return adjusted
```

### 2. Infinite Scroll Pages

```python
async def _handle_infinite_scroll(self, state: ScrollState) -> bool:
    """Detect and handle infinite scroll scenarios."""
    
    # Check if we're at bottom but element still not found
    if state.scroll_history[-1].hit_bottom:
        # Wait for new content to load
        await asyncio.sleep(2.0)
        
        # Check if new content appeared
        new_screenshot = await self.browser_driver.screenshot()
        if await self._has_new_content(new_screenshot, state.last_screenshot):
            return True  # Continue scrolling
        else:
            return False  # Give up - element not on page
```

### 3. Horizontal Scrolling

```python
async def _handle_horizontal_scroll_areas(self, visibility: VisibilityResult) -> ScrollAction:
    """Handle elements that require horizontal scrolling."""
    
    if visibility.suggested_direction in ['LEFT', 'RIGHT']:
        # Check if horizontal scroll is within a container
        container_check = await self._identify_scroll_container(screenshot)
        
        if container_check.has_container:
            # Scroll within container
            return ScrollAction(
                direction=visibility.suggested_direction,
                distance=200,
                target_container=container_check.container_coords
            )
```

## State Tracking and Analytics

### Scroll Pattern Analysis

```python
class ScrollPatternAnalyzer:
    def analyze_scroll_efficiency(self, state: ScrollState) -> ScrollAnalytics:
        """Analyze scroll patterns for optimization."""
        
        return ScrollAnalytics(
            total_attempts=state.attempts,
            total_distance=sum(abs(s.distance) for s in state.scroll_history),
            oscillations=self._count_oscillations(state.scroll_history),
            efficiency_score=self._calculate_efficiency(state),
            suggestions=self._generate_improvements(state)
        )
```

## Testing Strategy

### Scenario 1: Simple Vertical Scroll
- Page with element below fold
- Expected: 1-2 scroll attempts

### Scenario 2: Long Page Navigation
- Navigate to footer on very long page
- Test scroll-to-bottom optimization

### Scenario 3: Overshoot Correction
- Element in middle of long page
- Test bidirectional scrolling

### Scenario 4: Sticky Header Handling
- Page with fixed header
- Ensure element not hidden behind header

### Scenario 5: Horizontal Scroll
- Wide table or carousel
- Test horizontal navigation

## Performance Optimizations

1. **Screenshot Caching**: Cache recent screenshots to detect identical states
2. **Scroll Prediction**: Use page patterns to predict element locations
3. **Binary Search Scrolling**: For very long pages, use binary search pattern
4. **Content Hashing**: Detect when scrolling has no effect (reached limits)

## Success Metrics

- Average attempts to find element: < 5
- Success rate for finding visible elements: > 95%
- Overshoot correction success: > 90%
- Performance: < 10 seconds for typical scroll operations