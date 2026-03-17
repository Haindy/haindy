# Phase 6 — Test Runner Agent

**Tasks**: Test orchestration, step coordination, execution flow management.

**ETA**: 3 days

**Status**: Completed

## Overview

The Test Runner Agent is responsible for orchestrating test execution and deciding next steps. It maintains test context, coordinates execution, handles branching logic, and manages scripted automation fallbacks.

## AI Agent System Implementation

```python
class TestRunnerAgent(BaseAgent):
    """Orchestrates test execution → Decides next steps"""  
    def get_next_action(self, test_plan: TestPlan, current_state: TestState) -> ActionInstruction: pass
    def evaluate_step_result(self, step_result: StepResult) -> TestState: pass
```

## Agent Coordination Workflow

The Test Runner Agent plays a central role in the multi-agent execution flow:

```python
# Multi-agent execution flow
Human Input: "Test checkout flow for Product X"
    ↓
TestPlannerAgent: Creates structured test plan (8 steps)
    ↓
TestRunnerAgent: "Step 1: Navigate to product page"
    ↓
ActionAgent: Analyzes screenshot → "Click grid cell B7"
    ↓
BrowserDriver: Executes action, waits, captures screenshot  
    ↓
EvaluatorAgent: "Success - product page loaded correctly"
    ↓
TestRunnerAgent: Processes result → "Step 2: Add to cart"
    ↓
[Loop continues until test completion]
```

## Core Responsibilities

### Test Orchestration
- Receives structured test plans from Test Planner Agent
- Maintains execution context throughout test run
- Decides which step to execute next based on current state
- Handles branching logic and conditional execution

### Step Coordination
- Translates test steps into action instructions
- Coordinates with Action Agent for execution
- Processes evaluation results from Evaluator Agent
- Manages state transitions between steps

### Execution Flow Management
- Tracks current step, completed steps, and remaining steps
- Handles test plan updates based on discovered UI changes
- Manages retry logic and fallback strategies
- Coordinates scripted automation fallbacks

## State Management

The Test Runner maintains comprehensive state information:

- **Test execution state**: Current step, completed steps, remaining steps
- **Browser state tracking**: Page loaded, navigation state, UI changes
- **Agent communication state**: Message passing, result sharing
- **Test plan state**: Dynamic plan updates based on discovered UI changes

## Integration Points

### With Test Planner Agent
- Receives structured test plans
- Reports back on plan feasibility
- Requests plan clarification when needed

### With Action Agent
- Sends action instructions with context
- Receives execution results
- Handles action failures and retries

### With Evaluator Agent
- Receives success/failure assessments
- Uses evaluation results to determine next steps
- Requests re-evaluation when needed

### With Browser Driver
- Monitors browser state
- Coordinates navigation timing
- Manages page load detection

## Error Handling

- Implements retry logic (max 3 attempts per action)
- Falls back to alternative action paths when primary approaches fail
- Escalates to human intervention when confidence drops
- Maintains execution continuity despite individual step failures

## Observability

- Logs all test execution decisions
- Tracks step timing and performance
- Records state transitions
- Provides execution timeline with full audit trail