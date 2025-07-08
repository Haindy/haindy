# AsyncMock Issue with call_openai in ActionAgent Tests

## Problem Summary

The `_parse_coordinate_response` method in ActionAgent is returning a default confidence of 0.1 instead of the mocked value of 0.9. This happens because the method is receiving a coroutine object instead of a dictionary.

## Root Cause

In `action_agent.py`, the `_parse_coordinate_response` method has this check:

```python
if asyncio.iscoroutine(response):
    logger.error("Response is a coroutine, not a dict - likely test mock issue")
    raise ValueError("Response is a coroutine")
```

This exception is caught and causes the method to return the fallback coordinate with confidence 0.1.

## Why This Happens

1. In `_analyze_screenshot`, the code properly awaits `call_openai`:
   ```python
   response = await self.call_openai(...)  # Returns a dict
   ```

2. However, if the mock is not set up correctly, `response` might still be a coroutine or AsyncMock object.

## The Fix

### Option 1: Ensure Mock Returns Plain Dict

Instead of:
```python
agent.call_openai = AsyncMock(return_value={...})
```

Do this:
```python
# Create the mock
agent.call_openai = AsyncMock()
# Set return_value to a plain dict
agent.call_openai.return_value = {
    "content": json.dumps({
        "cell": "M23",
        "offset_x": 0.5,
        "offset_y": 0.5,
        "confidence": 0.9,
        "reasoning": "Button found"
    })
}
```

### Option 2: Use a Regular Async Function

```python
async def mock_call_openai(*args, **kwargs):
    return {
        "content": json.dumps({
            "cell": "M23",
            "offset_x": 0.5,
            "offset_y": 0.5,
            "confidence": 0.9,
            "reasoning": "Button found"
        })
    }

agent.call_openai = mock_call_openai
```

### Option 3: For Multiple Calls, Use side_effect

```python
responses = [
    {"content": json.dumps({"valid": True, "confidence": 0.95})},  # validation
    {"content": json.dumps({"cell": "M23", "confidence": 0.9})},   # coordinate
    {"content": json.dumps({"success": True, "confidence": 0.85})} # analysis
]

agent.call_openai = AsyncMock(side_effect=responses)
```

## Testing the Fix

To verify the fix works:

1. Add logging to `_parse_coordinate_response`:
   ```python
   logger.debug(f"Response type: {type(response)}, is_coroutine: {asyncio.iscoroutine(response)}")
   ```

2. In your test, ensure the ActionAgent instance has the mock:
   ```python
   # If using mock_action_agent fixture
   if hasattr(mock_action_agent, '_client'):
       mock_action_agent.call_openai = AsyncMock(side_effect=[...])
   ```

3. Run the test and check logs to ensure response is a dict, not a coroutine.

## Common Pitfalls

1. **Wrong Mock Target**: Ensure you're mocking `call_openai` on the actual ActionAgent instance, not the TestRunnerAgent.

2. **Mock Not Propagating**: If using a mock ActionAgent, the internal `call_openai` might not be mocked.

3. **Await Chain**: Ensure the entire chain properly awaits async calls.

## Example Working Test

```python
@pytest.mark.asyncio
async def test_action_agent_coordinate_determination():
    # Create real ActionAgent
    mock_browser = Mock()
    mock_browser.get_viewport_size = AsyncMock(return_value=(1920, 1080))
    agent = ActionAgent(browser_driver=mock_browser)
    
    # Mock call_openai properly
    coordinate_response = {
        "content": json.dumps({
            "cell": "M23",
            "offset_x": 0.5,
            "offset_y": 0.5,
            "confidence": 0.9,
            "reasoning": "Button found"
        })
    }
    
    # This ensures the mock returns a dict when awaited
    agent.call_openai = AsyncMock()
    agent.call_openai.return_value = coordinate_response
    
    # Now _parse_coordinate_response will receive a dict, not a coroutine
```