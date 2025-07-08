# Fix Plan: Wikipedia Test Error Reporting and Architectural Refactoring

## Problem Statement

1. HTML reports don't show error details - just counts
2. No detailed error information in logs or reports
3. When actions fail, we need comprehensive debugging information:
   - Screenshot with grid overlay
   - AI's grid cell choice and reasoning
   - Expected vs actual outcomes
   - Bug report generation
4. Typing might not be working because click isn't focusing the search box properly
5. **Architectural Issue**: Responsibilities are split incorrectly between Test Runner and Action Agent

## Root Causes

1. **Reporter Issue**: The HTML template only shows error counts, not details
2. **Error Capture**: TestStepResult doesn't capture enough debugging info when failures occur
3. **Grid Screenshot**: We save regular screenshots but not with grid overlay for debugging
4. **Focus Issue**: Click might not be focusing the search input properly before typing
5. **Architecture**: Action Agent only finds coordinates, doesn't execute actions or validate

## Architectural Refactoring

### Current Architecture (WRONG)
- Test Runner: Takes screenshots, executes browser actions, orchestrates everything
- Action Agent: Only analyzes screenshots and returns coordinates
- Evaluator: Judges success after the fact

### Proposed Architecture (CORRECT)
- Test Runner: High-level orchestration, provides context, judges final success/failure
- Action Agent: Owns entire action execution (validation → coordinates → execution → results)
- Evaluator: Removed or merged into Test Runner's final judgment

## Progress Tracking

| Phase | Status | PR | Completion Date |
|-------|--------|----|----|
| Phase 0: Refactor Action Agent | ✅ Complete | #16 | Completed |
| Phase 1: Enhanced ActionResult | ⚠️ Incomplete | #17 | Partially Done |
| Phase 2: Refactor Test Runner | ⚠️ Incomplete | #18 | Partially Done |
| Phase 3: Remove/Merge Evaluator | ✅ Complete | - | Completed |
| Phase 4: Enhanced Error Reporting | ⚠️ Incomplete | - | Partially Done |
| Phase 5: Terminal Output Enhancement | ⚠️ Incomplete | - | Partially Done |
| Phase 6: Fix Typing Action | ✅ Complete | #23 | 2025-01-08 |
| Phase 7: Complete Architecture Migration | ✅ Complete | #24,#25,#26 | 2025-01-08 |
| **Phase 8: Comprehensive Action Agent** | 📅 Pending | - | - |

## Revised Fix Plan

### Phase 0: Refactor Action Agent Responsibilities
- [x] Move browser action execution from Test Runner to Action Agent
- [x] Add validation step before attempting actions
- [x] Create comprehensive ActionResult that includes all debugging info

#### New Action Agent Flow:
1. **Validation Prompt**: 
   - Does this action make sense in current context?
   - Is the grid visible and correct?
   - Can I see the target element?
   - Return error if validation fails

2. **Coordinate Determination**:
   - Current logic for finding grid coordinates
   - Include reasoning in results

3. **Action Execution**:
   - Take screenshot with grid overlay
   - Perform browser action (click/type)
   - Wait for UI update
   - Take after screenshot

4. **Result Compilation**:
   - Before screenshot with grid and highlighted cell
   - Grid coordinates selected
   - After screenshot
   - Current URL
   - Success/failure assessment

### Phase 1: Enhanced ActionResult Model
- [x] Create new ActionResult with:
  - `validation_passed`: bool
  - `validation_reasoning`: str
  - `grid_screenshot_before`: bytes (with overlay and highlight)
  - `grid_cell`: str
  - `coordinates`: Tuple[int, int]
  - `screenshot_after`: bytes
  - `url_before`: str
  - `url_after`: str
  - `execution_time`: float
  - `error_details`: Optional[Dict] (if failed)

### Phase 2: Refactor Test Runner Agent
- [x] Remove browser action execution code
- [x] Update to pass clear context to Action Agent:
  - Test plan summary
  - Current test step details
  - Previous steps summary
  - Expected outcome
- [x] Implement final success/failure judgment with full context

### Phase 3: Remove/Merge Evaluator Agent
- [x] Move evaluation logic into Test Runner's judgment
- [x] Use AI to assess if actual results match expected with full context

### Phase 4: Enhanced Error Reporting
- [x] Update TestStepResult to capture ActionResult details
- [x] Create BugReport model for failed steps
- [x] Update HTML template to show:
  - Grid screenshots with clicked cells highlighted
  - AI reasoning for validation and coordinate selection
  - Before/after screenshots
  - URL changes
  - Detailed error messages

### Phase 5: Terminal Output Enhancement
- [x] Add bug report summary at end of test run
- [x] Show failed steps with:
  - What was attempted
  - What went wrong
  - Screenshots (as file paths)
  - AI reasoning

### Phase 6: Fix Typing Action
- [x] Ensure click properly focuses elements
- [x] Add validation that element is focusable
- [x] Consider multiple click strategies (single, double, click-and-wait)

### Phase 7: Complete Architecture Migration
**Problem**: Phases 1-5 created new `EnhancedActionResult` but didn't fully migrate all legacy code expecting dictionary format, causing `.get()` attribute errors.

**Root Cause**: Half-migration state - new architecture exists but legacy code still expects old dictionary-based ActionResult format.

**Exit Criteria**: Wikipedia test runs successfully without any `.get()` attribute errors.

#### Phase 7a: Audit and Fix Broken Legacy Code
- [x] Search codebase for all `.get()` calls on result/action objects
- [x] Identify all locations expecting dictionary-format ActionResult 
- [x] Map which files need fixing:
  - `test_runner.py` - **CRITICAL**: 20+ broken `.get()` calls in bug reporting, terminal output
  - `test_runner_v2.py` - **LEGACY**: May be unused file with broken `.get()` calls
  - `journal/manager.py` - **DIFFERENT ISSUE**: Uses JournalActionResult (separate type)  
  - `monitoring/reporter.py` - **CLEAN**: No ActionResult dependency found
- [x] **AUDIT COMPLETE**: Created PHASE_7A_AUDIT.md with detailed findings and priority order

#### Phase 7b: Fix Core Test Runner
- [x] Replace all `.get()` calls in `TestStepResult.create_bug_report()` with proper object access
- [x] Replace all `.get()` calls in `TestRunner._print_terminal_summary()` with proper object access  
- [x] Fix `.get()` calls in `judge_final_test_result()` method
- [x] **IMMEDIATE FIX**: Fixed line 980 causing Wikipedia test failure
- [x] **VERIFIED**: Wikipedia test now runs without `.get()` attribute errors

#### Phase 7c: Fix Supporting Systems
- [x] Fix Journal Manager to handle `EnhancedActionResult` properly
- [x] Fix Script Recorder to extract data from new format  
- [x] Fix HTML Reporter to use new result structure
- [x] Fix any monitoring/analytics code

#### Phase 7d: End-to-End Validation
- [x] Run Wikipedia test successfully (infrastructure works, Action Agent incomplete)
- [x] Verify no `.get()` attribute errors in logs
- [x] Confirm all debugging info displays correctly
- [x] Validate HTML reports show enhanced error details
- [x] Architecture migration complete - ready for Phase 8

#### Phase 7e: Final Cleanup
- [x] Remove any remaining old ActionResult references
- [x] Update type hints throughout codebase
- [x] Verify all tests pass with new format

### Phase 8: Comprehensive Action Agent
**Problem**: Current Action Agent only handles basic click/type actions with grid coordinates. It cannot handle navigation, dropdowns, toggles, drag operations, or other complex browser interactions.

**Root Cause**: Action Agent was designed as a simple coordinate-finding function, not as a true autonomous agent capable of multi-step browser interactions.

**Exit Criteria**: Wikipedia test runs successfully with all action types working properly.

#### Phase 8a: Multi-Step Action Framework
**Core Design**: Visual-only approach using grid coordinates - NO DOM access, selectors, or element IDs.

**Implementation Approach**: Hybrid model where Test Runner provides action_type hint, but AI can override based on visual analysis.

**Key Components**:
1. **Action Analysis**:
   - [ ] AI analyzes screenshot + instruction to determine actual workflow needed
   - [ ] Can override Test Runner's action_type if visual evidence differs
   - [ ] Returns workflow type and relevant visual markers

2. **Workflow Architecture**:
   - [ ] Base `ActionWorkflow` class with standard execution pattern
   - [ ] Specialized workflows for each action type (NavigateWorkflow, ClickWorkflow, etc.)
   - [ ] Each workflow can make multiple AI calls and screenshots as needed
   - [ ] All interactions through grid coordinates only

3. **Multi-Step Execution**:
   - [ ] Pre-execution validation (can this action be performed?)
   - [ ] Main action steps (may involve multiple screenshots/clicks)
   - [ ] Post-execution verification (did it work?)
   - [ ] State tracking between steps (dropdown open, drag in progress)

4. **Coordinate-Based Operations**:
   - [ ] All browser interactions via absolute coordinates
   - [ ] Support for: click(x,y), drag(x1,y1,x2,y2), type(text), key(keycode)
   - [ ] Grid refinement for precision when needed
   - [ ] No DOM queries or element selection

5. **Example Dropdown Workflow**:
   ```
   1. Screenshot with grid → AI locates dropdown visually
   2. Click at grid coordinates → Open dropdown
   3. New screenshot → See open menu
   4. If target not visible:
      - AI identifies scroll area
      - Click-drag to scroll
      - New screenshot
      - Repeat until found
   5. AI finds target option → Click at coordinates
   6. Verify selection visually
   ```

#### Phase 8b: Navigation Actions
**Design**: Navigation is handled by Action Agent like any other action, but uses browser.navigate() instead of grid coordinates.

**Navigation Workflow**:
1. **Execute Navigation**:
   - [ ] Extract URL from instruction or test step
   - [ ] Call browser.navigate(url)
   - [ ] Wait 1-2 seconds for page load (no complex detection needed)
   - [ ] Take screenshot for validation

2. **Visual Validation**:
   - [ ] AI compares screenshot against expected_outcome from test plan
   - [ ] Test Planner must provide detailed expected outcomes (e.g., "Wikipedia article about artificial intelligence with title visible")
   - [ ] AI confirms if expectation is met visually

3. **Error Detection**:
   - [ ] If result doesn't match expected, AI describes what it sees
   - [ ] Common errors: 404 pages, error messages, blank pages, wrong page
   - [ ] AI provides detailed description of unexpected result
   - [ ] Return EnhancedActionResult with validation details

4. **First Action Requirement**:
   - [ ] Enforce that first action in test plan is navigation (unless chained)
   - [ ] Test Planner should always start with "Navigate to [URL]"
   - [ ] Validation ensures we're at correct starting point

5. **Example Navigation Validation**:
   ```
   Expected: "Wikipedia homepage with search bar visible"
   AI Analysis: 
   - Success: "I see the Wikipedia homepage with logo and search bar"
   - Failure: "I see a 404 error page with text 'Page not found'"
   - Failure: "I see a blank white page with no content"
   ```

#### Phase 8c: Dropdown and Select Actions
**Design**: Visual-only dropdown interaction with scrolling support for long lists.

**Dropdown Workflow**:
1. **Initial Analysis**:
   - [ ] AI identifies dropdown element from screenshot + instruction
   - [ ] Check for blockers (animations, overlays) in same analysis
   - [ ] If blocked, wait and retry or fail with clear explanation
   - [ ] Return grid coordinates for dropdown click

2. **Open Dropdown**:
   - [ ] Click at dropdown coordinates
   - [ ] Wait brief moment (500ms)
   - [ ] Take screenshot to verify dropdown opened
   - [ ] If not open, retry once (some dropdowns are slow)

3. **Find Target Option**:
   - [ ] AI searches for target option in visible area
   - [ ] If option not visible:
     - Click-drag vertically in center of dropdown to scroll
     - Take new screenshot
     - Repeat until found or reach end
   - [ ] Return coordinates of target option

4. **Select Option**:
   - [ ] Click at option coordinates
   - [ ] Wait for dropdown to close (500ms)
   - [ ] Take screenshot for validation

5. **Validate Selection**:
   - [ ] AI verifies dropdown shows selected value
   - [ ] Standard dropdowns: Selected text visible in collapsed dropdown
   - [ ] Custom dropdowns: Use expected_outcome from test plan
   - [ ] If ambiguous or unclear, fail with explanation

6. **Error Handling**:
   - [ ] Can't find dropdown: "Unable to locate dropdown element"
   - [ ] Dropdown won't open: "Clicked dropdown but menu didn't appear"
   - [ ] Option not found: "Scrolled entire list but couldn't find 'Premium Plan'"
   - [ ] Ambiguous options: "Found multiple options matching 'Plan', unclear which to select"

#### Phase 8d: Toggle and Slider Actions
**Design**: Visual detection of toggle states and appropriate interaction methods.

**Toggle/Checkbox/Radio Workflow**:
1. **Visual State Detection**:
   - [ ] AI identifies element type and current state from screenshot
   - [ ] Toggle switches: Visual on/off position
   - [ ] Checkboxes: Checked/unchecked visual state
   - [ ] Radio buttons: Selected/unselected state
   - [ ] Return current state with coordinates

2. **Determine Action**:
   - [ ] Compare current state with desired state
   - [ ] If already in desired state, skip action
   - [ ] Most toggles: Simple click to change state
   - [ ] Some toggle switches: May need horizontal drag

3. **Execute Toggle**:
   - [ ] Click at element coordinates (most common)
   - [ ] For drag-style toggles: Click-drag horizontally
   - [ ] Wait brief moment for animation
   - [ ] Take screenshot for validation

4. **Validate State Change**:
   - [ ] AI confirms new visual state matches expected
   - [ ] If state didn't change, try alternative method (drag if clicked)
   - [ ] Return success/failure with visual confirmation

**Slider Workflow**:
1. **Slider Handling**:
   - [ ] For MVP: Focus on sliders with text input boxes
   - [ ] Type value in associated text box if available
   - [ ] For drag-only sliders: Best effort drag, not pixel-perfect
   - [ ] Validate final position visually

2. **Radio Button Groups**:
   - [ ] Instruction specifies which option to select
   - [ ] AI finds specific radio button by label
   - [ ] Click to select, verify visually selected
   - [ ] No special handling needed - AI understands groups

#### Phase 8e: Drag and Window Operations
**Design**: Implement drag support in browser driver and use for complex interactions.

**Driver Enhancement**:
1. **Add Drag Support to PlaywrightDriver**:
   - [ ] Add `drag(start_x, start_y, end_x, end_y)` method
   - [ ] Implementation using Playwright mouse API:
     ```python
     async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int):
         await self._page.mouse.move(start_x, start_y)
         await self._page.mouse.down()
         await self._page.mouse.move(end_x, end_y)
         await self._page.mouse.up()
     ```
   - [ ] Add optional speed parameter for smooth dragging
   - [ ] Update BrowserDriver interface to include drag method

**Drag Workflows**:
1. **Element Dragging**:
   - [ ] AI identifies draggable element from screenshot
   - [ ] AI identifies drop target location
   - [ ] Get start coordinates (center of draggable)
   - [ ] Get end coordinates (center of drop zone)
   - [ ] Execute drag(start_x, start_y, end_x, end_y)
   - [ ] Validate element moved visually

2. **Window/Pane Resizing**:
   - [ ] AI identifies resize handle (corner or edge)
   - [ ] Click and drag from handle to new position
   - [ ] Validate new size visually

3. **Scrollbar Dragging**:
   - [ ] AI identifies scrollbar thumb
   - [ ] Calculate drag distance based on desired scroll
   - [ ] Drag scrollbar thumb to new position
   - [ ] More precise than mouse wheel scrolling

4. **File Drag-and-Drop**:
   - [ ] For MVP: Focus on dragging within the page
   - [ ] File system integration is out of scope
   - [ ] Drag from file list to upload zone if both visible

5. **Canvas/Drawing**:
   - [ ] Support basic drawing gestures
   - [ ] Signature fields: Click-drag to draw
   - [ ] Selection rectangles: Click-drag to select area
   - [ ] Best effort - not pixel-perfect art

#### Phase 8f: Advanced Interactions [FUTURE - Out of Scope]
**Note**: These advanced interactions are not required for MVP and will be addressed in future iterations.

- [ ] Context menu operations (right-click → select option)
- [ ] File upload dialogs  
- [ ] Multi-step form validation and error handling
- [ ] Keyboard shortcuts and key combinations
- [ ] Frame and iframe navigation

#### Phase 8g: Integration and Testing
**Exit Criteria**: Wikipedia test passes completely.

**Simple Goal**: Make the Wikipedia test work end-to-end.

1. **Test Execution**:
   - [ ] Run `python -m src.main --json-test-plan test_scenarios/wikipedia_search.json`
   - [ ] If it passes: Phase 8 COMPLETE! 🎉
   - [ ] If it fails: Generate detailed failure report

2. **Failure Reporting** (if test fails):
   - [ ] Which step failed? (Step number and description)
   - [ ] What was the error? (Exact error message)
   - [ ] What did the AI see? (Screenshot analysis)
   - [ ] What did the AI try to do? (Action attempted)
   - [ ] Why did it fail? (Root cause analysis)
   - [ ] DO NOT attempt fixes - just document clearly

3. **Success Criteria**:
   - [ ] All 11 steps of Wikipedia test execute successfully
   - [ ] Search functionality works (can type in search box)
   - [ ] Navigation works (can click on article)
   - [ ] Validation works (can verify page content)
   - [ ] No errors or exceptions during execution

## Git Workflow

Each phase should follow this workflow:

1. **Create feature branch**: 
   - Branch from `fix-plan` (not main)
   - Name format: `pX-phase-description` (e.g., `p4-enhanced-error-reporting`)

2. **Development**:
   - Implement the phase requirements
   - Write/update tests as needed
   - Ensure all tests pass

3. **Testing**:
   - Run relevant tests: `source venv/bin/activate && python -m pytest tests/test_file.py -v`
   - Verify no regressions in related tests
   - Fix any test failures before proceeding

4. **Documentation**:
   - Update this fix plan with completion status
   - Mark completed items with [x]
   - Update progress tracking table

5. **Create PR**:
   - Create PR against `fix-plan` branch (NOT main)
   - Include comprehensive description of changes
   - Reference the phase number and objectives

6. **Merge and Continue**:
   - After PR is merged, pull latest from `fix-plan`
   - Start next phase with a new branch

## Implementation Order

1. Phase 0: Refactor Action Agent (most critical)
2. Phase 1: Enhanced ActionResult
3. Phase 2: Refactor Test Runner
4. Phase 3: Remove/Merge Evaluator
5. Phase 4: Enhanced Error Reporting
6. Phase 5: Terminal Output
7. Phase 6: Fix Typing

## Success Criteria

- [x] Action Agent validates before executing
- [x] Failed actions include comprehensive debugging info
- [x] Grid screenshots show exactly what was clicked
- [x] Bug reports clearly explain what went wrong
- [ ] **Navigation actions work reliably** ⚠️ BLOCKED by Phase 8
- [ ] **All browser interaction types supported** ⚠️ BLOCKED by Phase 8
- [ ] **Multi-step actions execute autonomously** ⚠️ BLOCKED by Phase 8
- [ ] **Wikipedia test runs end-to-end without errors** ⚠️ PRIMARY EXIT CRITERIA

## Current Status: INCOMPLETE ACTION AGENT

**Critical Issue**: Phase 7 (Architecture Migration) is complete, but the Action Agent is fundamentally incomplete and cannot handle basic browser interactions:

1. ✅ **Architecture Migration Complete**: No more `.get()` errors, EnhancedActionResult working
2. ❌ **Action Agent Cannot Navigate**: Fails on Step 1 because it treats navigation like clicking
3. ❌ **Limited Action Types**: Only supports click/type with grid coordinates
4. ❌ **Single-Step Design**: Cannot handle complex multi-step browser interactions

**Next Steps**: Complete Phase 8 (Comprehensive Action Agent) to make Wikipedia test functional.

## Notes

- This is a significant architectural change but necessary for proper debugging
- Action Agent becomes responsible for ALL action-related activities
- Test Runner focuses on orchestration and context management
- Every action produces rich debugging information