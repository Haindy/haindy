# Phase 17 — Additional Action Types

## Overview
Implement additional action types beyond click and type to support more complex test scenarios.

**ETA**: TBD (Currently Paused)

## Status: 🟨 PAUSED

### Reason for Pause
Discovered architectural limitation in Test Runner - it currently only executes literal test steps and cannot dynamically inject helper actions (like scrolling). This prevents intelligent use of the new scroll actions.

### Detailed Implementation Plan
The old detailed plan file was removed because it no longer reflected the current codebase. Use this phase document and the current implementation as the source of truth.

### Current Progress
- ✅ Scroll actions (5/5) implemented and tested
- 🚫 Phase paused pending Test Runner enhancement

### Problem Discovered
The Test Runner Agent currently only executes literal test steps and cannot dynamically inject helper actions (like scrolling). This prevents intelligent use of the new scroll actions.

### Required Fix (New Phase Needed)
Enhance Test Runner to:
1. Understand test step intent (not just literal execution)
2. Break down steps into multiple actions when needed
3. Inject helper actions (scroll, hover, wait) intelligently
4. Retry failed assertions with different strategies

### Action Types to Implement

#### 1. Scroll Actions (Priority 1) ✅ COMPLETED
- Scroll to element ✅
- Scroll by pixels ✅
- Scroll to bottom/top ✅
- Horizontal scrolling ✅

#### 2. Extended Interactions (Priority 2) - PENDING
- Hover/mouse over
- Drag and drop
- Right-click/context menu
- Double-click

#### 3. Form Interactions (Priority 3) - PENDING
- Select dropdown options
- File upload
- Checkbox/radio button groups
- Date picker interactions

#### 4. Validation Actions (Priority 2) - PENDING
- URL validation (programmatic, not visual)
- Page title validation
- Browser state validation

### Success Criteria
- Wikipedia test can verify sections below viewport
- Can interact with dropdown menus
- Can handle complex form interactions
- Test Runner intelligently uses helper actions

### Next Steps
1. Create new phase for Test Runner enhancement
2. Implement dynamic action injection in Test Runner
3. Resume Phase 14 implementation
4. Test all action types with enhanced Test Runner
