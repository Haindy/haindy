# Phase 17: Additional Action Types Implementation Plan

## Overview

Phase 17 extends HAINDY's Action Agent capabilities with new action types essential for comprehensive web application testing. This phase addresses current limitations that prevent testing of content below the viewport and complex interactions.

## Progress Tracking

### Overall Progress: 5/15 Actions 🟨 (PAUSED - Pending Test Runner Architecture Fix)

| Category | Progress | Status |
|----------|----------|---------|
| Scroll Actions (4) | 5/5* | ✅ Implemented (includes horizontal) |
| Extended Interactions (4) | 0/4 | ⬜ Not Started |
| Form Interactions (4) | 0/4 | ⬜ Not Started |
| Validation Actions (3) | 0/3 | ⬜ Not Started |

**Note**: Phase 17 is paused. The scroll actions are implemented but cannot be used intelligently until the Test Runner architecture is enhanced to break down test steps into multiple actions.

### Detailed Progress

#### Priority 1: Scroll Actions (5/5)* ✅

| Action | Status | Implementation | Testing |
|--------|--------|----------------|---------|
| Scroll to element | ✅ Complete | ✅ | ✅ |
| Scroll by pixels | ✅ Complete | ✅ | ✅ |
| Scroll to bottom/top | ✅ Complete | ✅ | ✅ |
| Horizontal scrolling | ✅ Complete | ✅ | ✅ |

*Note: We implemented 5 scroll actions instead of 4:
- SCROLL_TO_ELEMENT (with iterative algorithm)
- SCROLL_BY_PIXELS
- SCROLL_TO_TOP
- SCROLL_TO_BOTTOM
- SCROLL_HORIZONTAL

#### Priority 2: Extended Interactions (0/4) ⬜

| Action | Status | Implementation | Testing |
|--------|--------|----------------|---------|
| Hover/mouse over | ⬜ Not Started | ⬜ | ⬜ |
| Drag and drop | ⬜ Not Started | ⬜ | ⬜ |
| Right-click/context menu | ⬜ Not Started | ⬜ | ⬜ |
| Double-click | ⬜ Not Started | ⬜ | ⬜ |

#### Priority 2: Validation Actions (0/3) ⬜

| Action | Status | Implementation | Testing |
|--------|--------|----------------|---------|
| URL validation | ⬜ Not Started | ⬜ | ⬜ |
| Page title validation | ⬜ Not Started | ⬜ | ⬜ |
| Browser state validation | ⬜ Not Started | ⬜ | ⬜ |

#### Priority 3: Form Interactions (0/4) ⬜

| Action | Status | Implementation | Testing |
|--------|--------|----------------|---------|
| Select dropdown options | ⬜ Not Started | ⬜ | ⬜ |
| File upload | ⬜ Not Started | ⬜ | ⬜ |
| Checkbox/radio button groups | ⬜ Not Started | ⬜ | ⬜ |
| Date picker interactions | ⬜ Not Started | ⬜ | ⬜ |

## Implementation Architecture

### Core Changes Required

1. **Extend TestStep action_type enum** (`src/core/types.py`)
   - Add new action types to the ActionType enum
   - Update TestStep model to support action-specific parameters

2. **Update ActionAgent workflows** (`src/agents/action_agent.py`)
   - Add new workflow methods for each action type
   - Implement action-specific prompt templates
   - Handle action-specific parameters and validation

3. **Enhance Browser Driver** (`src/browser/driver.py`)
   - Add new Playwright methods for each action type
   - Implement proper error handling and retry logic
   - Add screenshot capture for action verification

4. **Update Test Runner** (`src/agents/test_runner.py`)
   - Handle new action types in execution flow
   - Add action-specific success criteria evaluation

## Detailed Implementation Plans

### 1. Scroll Actions (Priority 1)

#### 1.1 Scroll to Element
**Purpose**: Navigate to elements not visible in current viewport

**Detailed Design**: See [scroll_to_element_design.md](plans/scroll_to_element_design.md) for comprehensive implementation details.

**Implementation Approach**:
```python
# ActionType enum addition
SCROLL_TO_ELEMENT = "scroll_to_element"

# Core workflow with iterative scrolling
async def _scroll_to_element_workflow(self, screenshot: bytes, instruction: str):
    state = ScrollState(target_element=instruction)
    
    while state.attempts < state.max_attempts:
        # Check visibility with AI (full/partial/not visible)
        visibility = await self._check_element_visibility(screenshot, instruction)
        
        if visibility.fully_visible:
            return success
            
        # Calculate intelligent scroll distance based on:
        # - Confidence in direction
        # - Previous attempts (convergence)
        # - Partial visibility status
        # - Overshoot detection
        
        scroll_action = await self._plan_scroll_action(state, visibility)
        await self._execute_scroll(scroll_action)
        
        # Update state and capture new screenshot
```

**Key Features**:
- Multi-level visibility detection (fully/partially/not visible)
- Dynamic scroll distance calculation with convergence
- Overshoot detection and correction
- Oscillation prevention
- Progress tracking to avoid infinite loops

**Implementation Example**: See [scroll_implementation_example.py](plans/scroll_implementation_example.py) for complete code.

**Success Criteria**:
- Element becomes fully visible in viewport
- Average attempts < 5 for typical scenarios
- Handles edge cases (sticky headers, infinite scroll)
- AI confidence > 80% for final position

#### 1.2 Scroll by Pixels
**Purpose**: Precise scrolling for specific distances

**Implementation Approach**:
```python
# ActionType enum addition
SCROLL_BY_PIXELS = "scroll_by_pixels"

# Parameters in TestStep
scroll_parameters: {
    "direction": "vertical|horizontal",
    "pixels": 500
}

# Browser Driver method
async def scroll_by_pixels(self, x: int = 0, y: int = 0):
    await self.page.evaluate(f"window.scrollBy({x}, {y})")
```

**Success Criteria**:
- Page scrolled by exact pixel amount
- New viewport position verified

#### 1.3 Scroll to Bottom/Top
**Purpose**: Navigate to page extremes for footer/header testing

**Implementation Approach**:
```python
# ActionType enum additions
SCROLL_TO_TOP = "scroll_to_top"
SCROLL_TO_BOTTOM = "scroll_to_bottom"

# Browser Driver methods
async def scroll_to_top(self):
    await self.page.evaluate("window.scrollTo(0, 0)")
    
async def scroll_to_bottom(self):
    await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
```

**Success Criteria**:
- Page at absolute top/bottom position
- No further scrolling possible in target direction

#### 1.4 Horizontal Scrolling
**Purpose**: Test horizontal scroll areas, carousels, tables

**Implementation Approach**:
```python
# ActionType enum addition
SCROLL_HORIZONTAL = "scroll_horizontal"

# AI determines scroll target
# Browser executes horizontal scroll
await self.page.evaluate(f"window.scrollBy({amount}, 0)")
```

### 2. Extended Interactions (Priority 2)

#### 2.1 Hover/Mouse Over
**Purpose**: Trigger hover states, tooltips, dropdown menus

**Implementation Approach**:
```python
# ActionType enum addition
HOVER = "hover"

# Workflow combines grid detection with hover
async def _hover_workflow(self, screenshot: bytes, instruction: str):
    # 1. Use existing grid system to locate element
    # 2. Convert grid coordinates to page coordinates
    # 3. Execute hover action
    # 4. Wait for hover state to appear
    # 5. Capture screenshot of hover state
    
# Browser Driver method
async def hover(self, x: int, y: int):
    await self.page.mouse.move(x, y)
    await asyncio.sleep(0.5)  # Allow hover effects to appear
```

**Success Criteria**:
- Hover state visible (tooltip, menu, highlight)
- AI confirms expected hover effect appeared

#### 2.2 Drag and Drop
**Purpose**: Test sortable lists, file uploads, UI builders

**Implementation Approach**:
```python
# ActionType enum addition
DRAG_AND_DROP = "drag_and_drop"

# Two-phase AI interaction
async def _drag_and_drop_workflow(self, screenshot: bytes, instruction: str):
    # Phase 1: Identify source element
    source_coords = await self._get_element_coordinates(screenshot, "source element")
    
    # Phase 2: Identify target location
    target_coords = await self._get_element_coordinates(screenshot, "target location")
    
    # Execute drag and drop
    await self.browser_driver.drag_and_drop(source_coords, target_coords)
```

**Browser Implementation**:
```python
async def drag_and_drop(self, source: GridCoordinates, target: GridCoordinates):
    source_x, source_y = self._grid_to_page_coords(source)
    target_x, target_y = self._grid_to_page_coords(target)
    
    await self.page.mouse.move(source_x, source_y)
    await self.page.mouse.down()
    await self.page.mouse.move(target_x, target_y, steps=10)
    await self.page.mouse.up()
```

#### 2.3 Right-Click/Context Menu
**Purpose**: Test context menus, right-click actions

**Implementation Approach**:
```python
# ActionType enum addition
RIGHT_CLICK = "right_click"

# Workflow similar to regular click but with right button
async def right_click(self, x: int, y: int):
    await self.page.mouse.click(x, y, button="right")
    await asyncio.sleep(0.5)  # Allow context menu to appear
```

#### 2.4 Double-Click
**Purpose**: Test double-click interactions (text selection, activation)

**Implementation Approach**:
```python
# ActionType enum addition
DOUBLE_CLICK = "double_click"

# Browser Driver method
async def double_click(self, x: int, y: int):
    await self.page.mouse.dblclick(x, y)
```

### 3. Validation Actions (Priority 2)

#### 3.1 URL Validation
**Purpose**: Verify navigation, redirects, URL parameters

**Implementation Approach**:
```python
# ActionType enum addition
VALIDATE_URL = "validate_url"

# Direct programmatic validation (not visual)
async def _validate_url_workflow(self, expected_pattern: str):
    current_url = self.browser_driver.page.url
    
    # Support different validation modes
    if expected_pattern.startswith("regex:"):
        pattern = expected_pattern[6:]
        match = re.match(pattern, current_url)
    elif expected_pattern.startswith("contains:"):
        substring = expected_pattern[9:]
        match = substring in current_url
    else:
        match = current_url == expected_pattern
    
    return ValidationResult(
        success=match,
        actual_value=current_url,
        expected_value=expected_pattern
    )
```

#### 3.2 Page Title Validation
**Purpose**: Verify page titles for navigation confirmation

**Implementation Approach**:
```python
# ActionType enum addition
VALIDATE_TITLE = "validate_title"

# Browser Driver method
async def get_page_title(self) -> str:
    return await self.page.title()
```

#### 3.3 Browser State Validation
**Purpose**: Check cookies, local storage, session state

**Implementation Approach**:
```python
# ActionType enum addition
VALIDATE_BROWSER_STATE = "validate_browser_state"

# Flexible state validation
async def validate_browser_state(self, state_type: str, expected_value: Any):
    if state_type == "cookie":
        cookies = await self.page.context.cookies()
        # Validate specific cookie
    elif state_type == "local_storage":
        value = await self.page.evaluate("localStorage.getItem('key')")
        # Validate storage item
```

### 4. Form Interactions (Priority 3)

#### 4.1 Select Dropdown Options
**Purpose**: Interact with select elements, custom dropdowns

**Implementation Approach**:
```python
# ActionType enum addition
SELECT_OPTION = "select_option"

# Two-step process for custom dropdowns
async def _select_option_workflow(self, screenshot: bytes, instruction: str):
    # Step 1: Click dropdown to open
    dropdown_coords = await self._get_element_coordinates(screenshot, "dropdown")
    await self.browser_driver.click(dropdown_coords)
    
    # Step 2: Click desired option
    await asyncio.sleep(0.5)  # Wait for dropdown to open
    new_screenshot = await self.browser_driver.screenshot()
    option_coords = await self._get_element_coordinates(new_screenshot, f"option: {instruction}")
    await self.browser_driver.click(option_coords)
```

#### 4.2 File Upload
**Purpose**: Test file upload functionality

**Implementation Approach**:
```python
# ActionType enum addition
FILE_UPLOAD = "file_upload"

# Hybrid approach: visual click + programmatic file selection
async def _file_upload_workflow(self, screenshot: bytes, file_path: str):
    # Click the upload button visually
    upload_coords = await self._get_element_coordinates(screenshot, "upload button")
    
    # Use Playwright's file chooser API
    async with self.page.expect_file_chooser() as fc_info:
        await self.browser_driver.click(upload_coords)
    file_chooser = await fc_info.value
    await file_chooser.set_files(file_path)
```

#### 4.3 Checkbox/Radio Button Groups
**Purpose**: Handle grouped form controls

**Implementation Approach**:
```python
# ActionType enum addition
TOGGLE_CHECKBOX = "toggle_checkbox"
SELECT_RADIO = "select_radio"

# AI identifies all checkboxes/radios in group
async def _checkbox_group_workflow(self, screenshot: bytes, instruction: str):
    # Parse instruction for multiple selections
    # "Select checkboxes: Option A, Option C"
    options = self._parse_checkbox_instruction(instruction)
    
    for option in options:
        coords = await self._get_element_coordinates(screenshot, f"checkbox: {option}")
        await self.browser_driver.click(coords)
        await asyncio.sleep(0.3)  # Brief pause between clicks
```

#### 4.4 Date Picker Interactions
**Purpose**: Complex date picker navigation

**Implementation Approach**:
```python
# ActionType enum addition
SELECT_DATE = "select_date"

# Multi-step date selection
async def _date_picker_workflow(self, screenshot: bytes, target_date: str):
    # Step 1: Open date picker
    # Step 2: Navigate to correct month/year if needed
    # Step 3: Click target date
    # Complex interaction requiring multiple AI calls
```

## Testing Strategy

### Unit Tests
Each new action type requires:
- Workflow method tests
- Browser driver method tests
- Error handling tests
- Retry logic tests

### Integration Tests
- Test scenarios using new actions
- Multi-action sequences
- Edge cases and error conditions

### Test Scenarios to Create
1. **Scroll Test**: Navigate Wikipedia page, scroll to specific sections
2. **Hover Test**: E-commerce site with hover menus
3. **Form Test**: Complex form with all interaction types
4. **Validation Test**: Multi-page flow with URL/title checks

## Migration Path

### Phase 1: Core Implementation (Week 1)
1. Update type definitions
2. Implement Priority 1 actions (Scroll)
3. Basic testing

### Phase 2: Extended Actions (Week 2)
1. Implement Priority 2 actions
2. Integration testing
3. Update existing test scenarios

### Phase 3: Advanced Features (Week 3)
1. Implement Priority 3 actions
2. Comprehensive testing
3. Documentation updates

## Success Criteria

1. **Functional Requirements**
   - All 15 action types implemented and tested
   - Wikipedia test can verify sections below viewport
   - Can interact with dropdown menus
   - Can handle complex form interactions

2. **Technical Requirements**
   - Maintains existing code quality standards
   - 80%+ test coverage for new code
   - No regression in existing functionality
   - Clear error messages for each action type

3. **Performance Requirements**
   - Actions complete within reasonable timeframes
   - Efficient screenshot usage (minimize captures)
   - Smooth scrolling and interactions

## Risk Mitigation

1. **Complexity Management**
   - Implement actions incrementally
   - Thorough testing at each stage
   - Maintain backward compatibility

2. **AI Accuracy**
   - Clear, specific prompts for each action
   - Confidence thresholds for action execution
   - Fallback strategies for failed actions

3. **Browser Compatibility**
   - Test across different page types
   - Handle dynamic content appropriately
   - Account for animation and transitions

## Next Steps

1. Review and approve this plan
2. Begin implementation with Priority 1 (Scroll Actions)
3. Create test scenarios for each action type
4. Update documentation as we progress