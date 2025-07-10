# Phase 3 - Test Planner Agent

## Tasks
Requirements analysis, test plan generation, system prompts.

## ETA
2-3 days

## Status
Completed

## Overview

The Test Planner Agent is the first agent in the multi-agent workflow. It analyzes high-level requirements, PRDs (Product Requirements Documents), or test scenarios and creates structured test plans that can be executed by the system.

## Agent Definition

From the multi-agent architecture (Section 4.1):

```python
class TestPlannerAgent(BaseAgent):
    """Analyzes requirements/PRDs → Creates structured test plans"""
    def create_test_plan(self, requirements: str) -> TestPlan: pass
```

## Role and Responsibilities

### Primary Functions
- **Requirements Analysis**: Understands business requirements from natural language input
- **Test Plan Creation**: Creates comprehensive, structured test plans with clear steps
- **Context Understanding**: Interprets PRDs, user stories, and test case descriptions

### Agent Specialization
From Section 4.3, the Test Planner Agent specializes in:
- Understanding business requirements
- Creating comprehensive test plans
- Breaking down complex testing scenarios into actionable steps
- Ensuring test coverage matches requirements

## Workflow Position

In the multi-agent execution flow (Section 4.2):

```
Human Input: "Test checkout flow for Product X"
    ↓
TestPlannerAgent: Creates structured test plan (8 steps)
    ↓
[Continues to Test Runner Agent]
```

## Implementation Details

### Input Processing
- Accepts natural language requirements
- Supports PRDs, user stories, and test case descriptions
- Can handle file-based inputs (as per Phase 11b CLI improvements)

### Output Format
The Test Planner generates structured test plans containing:
- Test scenario name
- Numbered test steps
- Expected outcomes for each step
- Success criteria
- Any prerequisites or setup requirements

### System Prompts
The Test Planner uses specialized system prompts that:
- Guide the agent to create clear, actionable test steps
- Ensure comprehensive coverage of the requirements
- Maintain consistency in test plan format
- Consider edge cases and error scenarios

## Integration Points

### With Test Runner Agent
- Passes structured test plans for execution
- Test plans must be in a format the Test Runner can orchestrate

### With CLI Interface
- Integrates with the improved CLI (Phase 11b) to accept:
  - Interactive requirements input
  - File-based plan input
  - Direct JSON test plan execution

### State Management
From Section 4.5:
- Maintains test plan state
- Supports dynamic plan updates based on discovered UI changes

## Validation and Quality

### Cross-Agent Validation
From Section 8.3:
- Test plans validated for logical flow
- Step sequences checked for consistency
- Integration with the hierarchical validation strategy

### Error Handling
From Section 4.4:
- Validates its own outputs before passing to Test Runner
- Ensures test plans are complete and executable

## Technical Implementation

### Data Model
Works with the TestPlan data model defined in the core foundation:
- Test scenario metadata
- Ordered list of test steps
- Expected outcomes
- Success criteria

### File Structure
Located in: `src/agents/test_planner.py`
- Inherits from `BaseAgent` class
- Implements the `create_test_plan` method
- Uses prompts from `src/config/agent_prompts.py`

## Best Practices

1. **Clear Step Definition**: Each test step should be atomic and verifiable
2. **Explicit Expected Outcomes**: Every step must have clear success criteria
3. **Context Preservation**: Maintain awareness of application state throughout the plan
4. **Flexibility**: Plans should be adaptable to UI changes discovered during execution
5. **Human Readability**: Test plans must be understandable to both AI agents and humans

## Success Metrics

- Successfully analyzes diverse requirement formats
- Generates executable test plans with >80% accuracy
- Test plans result in successful test execution
- Minimal need for plan adjustments during execution