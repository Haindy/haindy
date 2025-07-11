# Test Failure Fix Examples

## 1. Pydantic Validation Fixes

### Problem: TestStep Missing Required Fields
```python
# ❌ FAILING TEST
test_step = TestStep(
    step_id=uuid4(),
    step_number=1,
    depends_on=[],
    optional=False
)
```

### Solution: Add Required Fields
```python
# ✅ FIXED TEST
test_step = TestStep(
    step_id=uuid4(),
    step_number=1,
    action="Click the login button",  # REQUIRED
    expected_result="User is redirected to dashboard",  # REQUIRED
    depends_on=[],
    optional=False
)
```

## 2. Mock Response Format Fixes

### Problem: Incorrect GridCoordinate Response
```python
# ❌ FAILING MOCK
mock_response = {
    "content": '{"cell": "A1", "confidence": 0.9}'
}
```

### Solution: Match Expected Format
```python
# ✅ FIXED MOCK
mock_response = {
    "content": json.dumps({
        "grid_cell": "M23",
        "offset_x": 0.5,
        "offset_y": 0.5,
        "confidence": 0.9,
        "element_type": "button",
        "reasoning": "Found login button at center of cell M23"
    })
}
```

## 3. Import Error Fixes

### Problem: Removed Message Types
```python
# ❌ FAILING IMPORT
from src.orchestration.communication import (
    MessageType, 
    EVALUATE_RESULT,  # No longer exists
    EVALUATION_COMPLETE  # No longer exists
)
```

### Solution: Remove or Replace
```python
# ✅ FIXED IMPORT
from src.orchestration.communication import MessageType

# Update test to use existing message types
assert MessageType.STEP_COMPLETED in MessageType
assert MessageType.ACTION_DETERMINED in MessageType
```

## 4. Enhanced Types Creation Fixes

### Problem: Missing Required Fields
```python
# ❌ FAILING TEST
result = EnhancedActionResult(
    overall_success=True,
    timestamp_end=datetime.now()
)
```

### Solution: Include All Required Fields
```python
# ✅ FIXED TEST
result = EnhancedActionResult(
    test_step_id=uuid4(),  # REQUIRED
    test_step=test_step,  # REQUIRED
    test_context={"test": "context"},  # REQUIRED
    overall_success=True,
    timestamp_start=datetime.now(),
    timestamp_end=datetime.now(),
    validation=ValidationResult(
        valid=True,
        confidence=0.9,
        reasoning="Test validation"
    )
)
```

## 5. State Manager Test Fixes

### Problem: Invalid State Creation
```python
# ❌ FAILING TEST
state = await state_manager.create_test_state(
    test_id=test_id,
    test_name="Test"
)
```

### Solution: Provide Complete Test Plan
```python
# ✅ FIXED TEST
test_plan = TestPlan(
    plan_id=uuid4(),
    name="Test Plan",
    description="Test description",
    requirements_source="Test requirements",
    test_cases=[],  # Can be empty for basic tests
    created_at=datetime.now(timezone.utc),
    created_by="Test"
)

state = await state_manager.create_test_state(
    test_plan=test_plan,
    test_id=test_id
)
```

## 6. Test Runner Mock Fixes

### Problem: AI Response Not Matching Expected Format
```python
# ❌ FAILING MOCK
async def mock_ai_response(*args, **kwargs):
    return {"content": "Failed to interpret"}
```

### Solution: Return Proper JSON Response
```python
# ✅ FIXED MOCK
async def mock_ai_response(*args, **kwargs):
    return {
        "content": json.dumps({
            "actions": [
                {
                    "type": "click",
                    "target": "Login button",
                    "description": "Click the login button",
                    "critical": True
                }
            ]
        })
    }
```

## Common Test Factory Pattern

### Create Reusable Factories
```python
# test_helpers.py
def create_test_step(**kwargs):
    """Factory for creating valid TestStep instances."""
    defaults = {
        "step_id": uuid4(),
        "step_number": 1,
        "action": "Default action",
        "expected_result": "Default expected result",
        "description": "Test step",
        "depends_on": [],
        "optional": False
    }
    defaults.update(kwargs)
    return TestStep(**defaults)

def create_test_plan(**kwargs):
    """Factory for creating valid TestPlan instances."""
    defaults = {
        "plan_id": uuid4(),
        "name": "Test Plan",
        "description": "Test description",
        "requirements_source": "Test",
        "test_cases": [],
        "created_at": datetime.now(timezone.utc),
        "created_by": "Test"
    }
    defaults.update(kwargs)
    return TestPlan(**defaults)

# Usage in tests
test_step = create_test_step(action="Click login", step_number=1)
test_plan = create_test_plan(name="Login Test Plan")
```

## Debugging Tips

1. **Check Required Fields**: Look at the model definition in `src/core/types.py`
2. **Check Mock Format**: Add logging to see actual vs expected responses
3. **Use Real Examples**: Copy response formats from working integration tests
4. **Validate Early**: Add assertions to verify mock data before using it