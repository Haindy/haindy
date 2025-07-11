# Test Files with Failures/Errors

## Files with Failing Tests (11 files)

1. **tests/test_action_agent.py**
   - 5 failures related to coordinate parsing
   - Core functionality broken (grid coordinate parsing)
   
2. **tests/test_action_agent_refactored.py**
   - 6 errors, 2 failures
   - Missing methods/imports after refactoring
   - Likely needs complete rewrite to match new architecture

3. **tests/test_base_agent.py**
   - 2 failures
   - Initialization and client loading issues

4. **tests/test_communication.py**
   - 1 failure
   - Message passing between agents

5. **tests/test_coordinator.py**
   - 6 failures
   - Agent initialization and test execution flow
   - Major refactoring impact

6. **tests/test_end_to_end_integration.py**
   - 2 errors
   - Integration test failures

7. **tests/test_enhanced_error_reporting.py**
   - 5 errors (all ValidationError)
   - TestStep model changes
   - Could be fixed by updating test data

8. **tests/test_fix_typing_action.py**
   - 1 failure
   - Text input action handling

9. **tests/test_main.py**
   - 2 failures
   - CLI main entry point issues

10. **tests/test_scroll_actions.py**
    - 4 errors (all ValidationError)
    - TestStep model changes
    - Could be fixed by updating test data

11. **tests/test_test_runner.py**
    - 6 failures
    - Decision logic and bug reporting
    - Major refactoring impact

## Recommendations for Complete Rewrites

Based on the analysis, these test files might benefit from complete rewrites:

1. **test_action_agent_refactored.py** - Too many errors indicate fundamental mismatch with current implementation
2. **test_coordinator.py** - 6 failures suggest the coordinator has been significantly refactored
3. **test_test_runner.py** - 6 failures indicate major changes in test runner logic

## Quick Fixes Possible

These files just need test data updates:

1. **test_enhanced_error_reporting.py** - Just needs TestStep model fields added
2. **test_scroll_actions.py** - Just needs TestStep model fields added

## Core Functionality Broken

1. **test_action_agent.py** - Coordinate parsing is fundamentally broken but might be fixable