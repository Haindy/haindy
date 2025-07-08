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
| Phase 1: Enhanced ActionResult | ✅ Complete | #17 | Completed |
| Phase 2: Refactor Test Runner | ✅ Complete | #18 | Completed |
| Phase 3: Remove/Merge Evaluator | ✅ Complete | - | Completed |
| Phase 4: Enhanced Error Reporting | ✅ Complete | - | Completed |
| Phase 5: Terminal Output Enhancement | ✅ Complete | - | Completed |
| Phase 6: Fix Typing Action | ✅ Complete | #23 | 2025-01-08 |

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
- [ ] Typing works reliably on Wikipedia search
- [ ] Architecture follows clean separation of concerns

## Notes

- This is a significant architectural change but necessary for proper debugging
- Action Agent becomes responsible for ALL action-related activities
- Test Runner focuses on orchestration and context management
- Every action produces rich debugging information