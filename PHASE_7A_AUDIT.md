# Phase 7a: Audit Results - Broken Legacy Code

## Overview
This audit identifies all locations where legacy code expects dictionary-format ActionResult but now receives EnhancedActionResult objects, causing `.get()` attribute errors.

## Critical Issues Found

### 1. src/agents/test_runner.py - MULTIPLE LOCATIONS
**Status**: CRITICAL - Active errors blocking Wikipedia test

#### TestStepResult.create_bug_report() method (lines ~95-109)
```python
# BROKEN - expects dict format
if details.get("validation_passed") is False:
elif details.get("execution_success") is False:
ai_analysis_data = details.get("ai_analysis", {})
confidence_scores = {
    "validation": details.get("validation_confidence", 0.0),
    "coordinate": details.get("coordinate_confidence", 0.0),
    "execution": 1.0 if details.get("execution_success") else 0.0,
}
```

#### Bug report creation (lines ~134-145)  
```python
# BROKEN - expects dict format
detailed_error = details.get("execution_error")
grid_screenshot = details.get("grid_screenshot_highlighted")
grid_cell_targeted = details.get("grid_cell")
coordinates_used = GridCoordinate(
    cell=details.get("grid_cell", ""),
    offset_x=details.get("offset_x", 0.5),
    # ... etc
)
```

#### _print_terminal_summary() method (lines ~950+)
```python
# BROKEN - expects dict format  
ai_analysis = result.action_result_details.get("ai_analysis", {})
grid_screenshot = result.action_result_details.get("grid_screenshot_highlighted")
```

#### Additional broken locations (lines ~980+)
```python
# BROKEN - causing current error
detail["validation_passed"] = result.action_result_details.get("validation_passed", False)
detail["validation_reasoning"] = result.action_result_details.get("validation_reasoning", "")
detail["execution_error"] = result.action_result_details.get("execution_error", "")
detail["ai_analysis"] = result.action_result_details.get("ai_analysis", {})
```

### 2. src/agents/test_runner_v2.py - LEGACY FILE
**Status**: BROKEN - May be unused legacy

#### Multiple .get() calls on ai_analysis
```python
# BROKEN - expects dict format
return action_result.ai_analysis.get("success", False)
AI Analysis: {action_result.ai_analysis.get('actual_outcome', 'Not available')}
actual_outcome=action_result.ai_analysis.get("actual_outcome", "Action failed"),
```

### 3. Journal System - DIFFERENT TYPE ISSUE
**Status**: NEEDS INVESTIGATION - Uses JournalActionResult, not EnhancedActionResult

#### src/journal/manager.py
- Uses `JournalActionResult` type (defined in journal/models.py)
- This is a separate type from both old ActionResult and new EnhancedActionResult
- May need adapter to convert EnhancedActionResult → JournalActionResult

### 4. Monitoring System - CLEAN
**Status**: OK - No obvious ActionResult dependency found in reporter.py

## Data Structure Analysis

### Current EnhancedActionResult Structure
```python
class EnhancedActionResult:
    validation: ValidationResult
    coordinates: CoordinateResult  
    execution: ExecutionResult
    ai_analysis: AIAnalysis
    browser_state_before: BrowserState
    browser_state_after: BrowserState
    grid_screenshot_before: bytes
    grid_screenshot_highlighted: bytes
    # ... etc
```

### Expected Legacy Dictionary Format
```python
{
    "validation_passed": bool,
    "validation_confidence": float,
    "validation_reasoning": str,
    "execution_success": bool,
    "execution_error": str,
    "coordinate_confidence": float,
    "grid_cell": str,
    "offset_x": float, 
    "offset_y": float,
    "ai_analysis": {
        "success": bool,
        "confidence": float,
        "actual_outcome": str,
        "anomalies": list,
        "recommendations": list
    },
    "grid_screenshot_highlighted": bytes,
    "url_before": str,
    "url_after": str,
    # ... etc
}
```

## Mapping Required

### Phase 7b Tasks (Fix Core Test Runner)
1. **TestStepResult.create_bug_report()**: Replace 20+ `.get()` calls with object access
2. **TestRunner._print_terminal_summary()**: Replace 5+ `.get()` calls with object access  
3. **Action recording methods**: Ensure proper EnhancedActionResult handling
4. **ActionResult creation**: Verify all creation uses new format

### Phase 7c Tasks (Fix Supporting Systems)
1. **Journal Manager**: Create adapter EnhancedActionResult → JournalActionResult
2. **test_runner_v2.py**: Determine if legacy file can be deleted or needs fixing
3. **Any other systems**: TBD based on further investigation

## Priority Order
1. **IMMEDIATE**: Fix test_runner.py line ~980 causing current Wikipedia test failure
2. **HIGH**: Fix all other test_runner.py `.get()` calls  
3. **MEDIUM**: Handle journal system integration
4. **LOW**: Clean up test_runner_v2.py if it's truly legacy

## Exit Criteria Verification  
- [ ] No `.get()` attribute errors when running Wikipedia test
- [ ] All debugging info displays correctly in terminal
- [ ] HTML reports generate without errors
- [ ] Enhanced typing functionality works end-to-end