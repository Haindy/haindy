# Test Failure Visual Summary

## Failure Distribution

```
Pydantic Validation Errors  ████████████████████████████████████████████████ 45 (50%)
Mock Response Issues        ████████████           11 (12%)
Integration Tests           █████████              8 (9%)
Test Runner Issues          ████████               7 (8%)
Import/Module Issues        █████                  5 (5%)
Other/Unknown              ███████████████         15 (16%)
```

## By Test File

### Most Affected Files:
1. **test_state_manager.py** - 20 failures (ALL tests failing)
2. **test_journal_manager.py** - 11 failures (ALL tests failing)  
3. **test_coordinator.py** - 8 failures/errors
4. **test_enhanced_types.py** - 6 failures
5. **test_action_agent.py** - 5 failures
6. **test_fix_typing_action.py** - 5 failures
7. **test_scroll_actions.py** - 8 failures

### Files with Mixed Results:
- **test_action_agent_refactored.py** - 2 failed, 4 passed
- **test_core_types.py** - 5 failed, 1 passed
- **test_end_to_end_integration.py** - 4 failed, 2 passed
- **test_test_runner.py** - 2 failed, many passed

## Error Types

### Validation Errors (Pydantic)
```
TestStep missing fields:
├── action (required)
└── expected_result (required)

TestState missing fields:
├── test_plan (required)
└── status (required)

EnhancedActionResult missing fields:
├── test_step_id (required)
├── test_step (required)
└── test_context (required)
```

### Mock Response Errors
```
Expected: GridCoordinate(cell="M23", offset_x=0.5, offset_y=0.5)
Actual:   GridCoordinate(cell="A1", offset_x=0.0, offset_y=0.0)

Expected: {"type": "scroll", "direction": "down", "amount": 500}
Actual:   {"error": "Failed to parse coordinate response"}
```

### Import Errors
```
- Cannot import 'EVALUATE_RESULT' from communication.py
- Cannot import 'EVALUATION_COMPLETE' from communication.py
```

## Quick Fix Priority

### 🔴 Critical (Fix First)
1. **Pydantic validation in test fixtures** - 45 tests
   - Add missing required fields to all model creation
   - Use factory functions to ensure consistency

### 🟡 Important (Fix Second)  
2. **Mock response formats** - 11 tests
   - Update mocks to match implementation
   - Consider using real response examples

### 🟢 Nice to Have (Fix Later)
3. **Integration test updates** - 8 tests
   - Update for new 3-agent architecture
   - May require deeper refactoring

## Impact on CI/CD

Current state would **BLOCK** deployments:
- Coverage: 25% (requires 60%)
- Test failures: 91 tests not passing
- Critical path tests failing

## Recommended Action Plan

1. **Day 1**: Fix all Pydantic validation errors
   - Create test factories for models
   - Update all test fixtures
   
2. **Day 2**: Fix mock responses
   - Audit actual response formats
   - Update all mock returns
   
3. **Day 3**: Fix integration tests
   - Update for new architecture
   - Remove Evaluator references

4. **Day 4**: Cleanup and verification
   - Fix remaining tests
   - Ensure 60%+ coverage
   - Full regression test