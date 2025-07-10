# Phase 5 - Evaluator Agent

## Tasks
Result assessment, success/failure detection, UI state validation.

## ETA
2 days

## Status
Completed

## Overview

The Evaluator Agent is responsible for assessing screenshot results against expected outcomes. It determines whether actions were successful, validates UI states, and provides feedback to the Test Runner Agent for decision-making.

## Agent Definition

From the multi-agent architecture (Section 4.1):

```python
class EvaluatorAgent(BaseAgent):
    """Screenshot + expectation → Success/failure assessment"""
    def evaluate_result(self, screenshot: bytes, expected_outcome: str) -> EvaluationResult: pass
```

## Role and Responsibilities

### Primary Functions
- **Result Validation**: Assesses whether actions achieved expected outcomes
- **UI State Assessment**: Analyzes current UI state from screenshots
- **Error Detection**: Identifies failures, errors, and unexpected states
- **Confidence Scoring**: Provides confidence levels for evaluations

### Agent Specialization
From Section 4.3, the Evaluator Agent specializes in:
- Result validation
- UI state assessment
- Error detection
- Confidence scoring for outcomes

## Workflow Position

In the multi-agent execution flow (Section 4.2):

```
BrowserDriver: Executes action, waits, captures screenshot  
    ↓
EvaluatorAgent: "Success - product page loaded correctly"
    ↓
TestRunnerAgent: Processes result → "Step 2: Add to cart"
```

## Evaluation Process

### Input Processing
The Evaluator Agent receives:
- Screenshot of current browser state
- Expected outcome description from test step
- Context from previous actions (if relevant)

### Output Format
Returns an EvaluationResult containing:
- Success/failure determination
- Confidence score
- Detailed explanation of findings
- Identified UI elements or states
- Any error messages or unexpected conditions

## Validation Strategies

### Hierarchical Validation (Section 8.1)

The Evaluator participates in the layered validation architecture:

**Bottom-Up Validation**:
- Returns results with confidence scores
- Validates its own output before passing upstream
- Triggers retry or escalation based on confidence

**Cross-Agent Verification**:
- Validates Action Agent decisions
- Provides feedback for Test Runner decisions
- Prevents cascading errors through verification

### Confidence Scoring

Uses the same threshold system as other agents (Section 8.2):
- 95-100%: High confidence in evaluation
- 80-94%: Moderate confidence, may need review
- 60-79%: Low confidence, consider re-evaluation
- Below 60%: Uncertain, escalate to human

## Evaluation Criteria

### Success Detection
- Visual confirmation of expected UI changes
- Presence of success indicators (messages, counters, etc.)
- Correct page navigation
- Form submission confirmations
- Data updates visible on screen

### Failure Detection
- Error messages or alerts
- Unexpected page states
- Missing UI elements
- Loading indicators stuck
- Navigation failures

### State Validation
- Current page identification
- UI element visibility
- Dynamic content loading status
- Form field states
- Modal/popup detection

## Execution Journaling

### Logging Format (Section 6.1)
The Evaluator contributes to the structured logging:

```json
{
  "expected_result": "Product added to cart, cart counter increments",
  "actual_result": "Cart counter changed from 0 to 1, success notification appeared",
  "agent_confidence": 0.95,
  "success": true
}
```

## Error Handling

### Validation Checkpoints (Section 8.3)
The Evaluator implements key validation checkpoints:

- **Post-action validation**: "Did the action achieve expected outcome?"
- **Sequence validation**: "Does current state align with test plan progression?"
- **State consistency**: "Is the UI in an expected state?"

### Hallucination Mitigation
- Multi-factor evaluation approach
- Visual evidence requirements
- Consistency checks with test context
- Historical pattern matching

## Integration Points

### With Action Agent
- Validates Action Agent's execution results
- Provides feedback on action effectiveness
- Helps determine if refinement or retry is needed

### With Test Runner Agent
- Reports evaluation results for decision-making
- Influences next step selection
- Triggers error handling when needed

### With Test Planner Agent
- Validates against original test plan expectations
- Ensures test objectives are being met

## Technical Implementation

### File Structure
Located in: `src/agents/evaluator.py`
- Inherits from `BaseAgent` class
- Implements `evaluate_result` method
- Uses evaluation prompts from configuration

### Data Models
Works with:
- `EvaluationResult`: Contains success/failure, confidence, and details
- `TestState`: Updates test execution state based on evaluation
- Screenshots and expected outcomes

## Caching and Pattern Recognition

### Pattern Learning (Section 6.2)
- Successful evaluation patterns stored for reuse
- UI element recognition across test runs
- Builds adaptive pattern library over time

### Performance Optimization
- Cached evaluation patterns
- Quick recognition of common UI states
- Efficient error pattern matching

## Best Practices

1. **Clear Expectations**: Expected outcomes must be specific and verifiable
2. **Visual Evidence**: Base evaluations on visible UI elements
3. **Context Awareness**: Consider test flow and previous steps
4. **Detailed Feedback**: Provide clear explanations for decisions
5. **Confidence Calibration**: Accurate confidence scores for reliability

## Success Metrics

- Achieves >80% accuracy in result evaluation
- Correctly identifies success and failure states
- Provides actionable feedback for test progression
- Maintains high confidence in clear scenarios
- Successfully handles ambiguous states with appropriate confidence levels

## Special Considerations

### Visual-Only Validation
- No DOM access for validation
- Purely based on screenshot analysis
- Must handle dynamic content and animations
- Timing considerations for UI updates

### Multi-State Recognition
- Loading states vs. final states
- Transient messages and notifications
- Progressive UI updates
- Partial success scenarios