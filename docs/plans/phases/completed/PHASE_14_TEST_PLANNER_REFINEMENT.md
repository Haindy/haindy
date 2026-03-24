# Phase 14 — Test Planner Agent Refinement

## Overview
Refine the Test Planner Agent to create well-structured test plans with clear hierarchy and both machine-readable (JSON) and human-readable (Markdown) outputs.

**ETA**: 2 days

## Status: 📅 PLANNED

## 🚨 IMPORTANT: Development Strategy 🚨
**NO BACKWARD COMPATIBILITY REQUIRED**
- This is an actively developed tool with no production users
- We will REPLACE old code completely - no gradual migrations
- No need for dual implementations or compatibility layers
- Development will happen on `agent-refinement` branch
- Each phase will PR against `agent-refinement`, not `main`
- If things go wrong, we'll abandon the branch and start over

## Motivation
The Test Planner Agent needs clearer responsibilities and better structured outputs. It should create comprehensive test plans that contain test cases, which in turn contain steps, following a clear hierarchical structure.

## Current State
- Test Planner creates simple JSON test plans
- Limited structure in test plan outputs
- No human-readable format generation
- Unclear separation between test plan, test cases, and steps

## Goals

### Primary Goals
1. Implement clear test plan hierarchy: Test Plan → Test Cases → Steps
2. Generate both JSON (for machine processing) and Markdown (for human readability) outputs
3. Enhance requirement analysis and interpretation capabilities
4. Improve test coverage determination logic

### Test Plan Structure
```
Test Plan
├── Metadata (name, description, requirements source, created date)
├── Test Case 1
│   ├── Name
│   ├── Description
│   ├── Priority (Critical/High/Medium/Low)
│   ├── Prerequisites
│   ├── Step 1 (action, expected result)
│   ├── Step 2 (action, expected result)
│   └── Step N (action, expected result)
└── Test Case N
```

## Implementation Tasks

### 1. Data Model Enhancement
- [ ] Create TestPlan dataclass with metadata
- [ ] Create TestCase dataclass with proper fields
- [ ] Update TestStep model if needed
- [ ] Add validation for hierarchical structure

### 2. Test Planner Agent Updates
- [ ] Update system prompt for hierarchical understanding
- [ ] Implement structured output generation
- [ ] Add logic for test case prioritization
- [ ] Enhance requirement parsing capabilities

### 3. Output Generation
- [ ] Implement JSON serialization for test plans
- [ ] Create Markdown formatter for human-readable output
- [ ] Add template system for consistent formatting
- [ ] Ensure bidirectional conversion (JSON ↔ Markdown)

### 4. Integration Updates
- [ ] Update interfaces to support new data model
- [ ] Modify coordinator to handle hierarchical plans
- [ ] Update test runner to consume new format
- [ ] Clean replacement of existing code

### 5. Testing
- [ ] Unit tests for new data models
- [ ] Test planner output validation
- [ ] Integration tests with test runner
- [ ] End-to-end test with sample requirements

## Success Criteria
- Test plans have clear three-level hierarchy
- Both JSON and Markdown outputs are generated
- Human-readable plans are easily understandable
- Test Runner can consume and execute new format
- Existing tests continue to pass

## Technical Details

### Sample JSON Structure
```json
{
  "test_plan": {
    "name": "E-commerce Checkout Flow",
    "description": "Comprehensive test plan for checkout process",
    "created_at": "2025-01-10T10:00:00Z",
    "requirements_source": "PRD v1.2",
    "test_cases": [
      {
        "id": "TC001",
        "name": "Guest Checkout",
        "description": "Verify guest users can complete checkout",
        "priority": "Critical",
        "prerequisites": ["Product catalog available", "Payment gateway active"],
        "steps": [
          {
            "step_number": 1,
            "action": "Navigate to product page",
            "expected_result": "Product page loads with details"
          }
        ]
      }
    ]
  }
}
```

### Sample Markdown Output
```markdown
# Test Plan: E-commerce Checkout Flow

**Description**: Comprehensive test plan for checkout process
**Created**: 2025-01-10
**Requirements**: PRD v1.2

## Test Case TC001: Guest Checkout

**Priority**: Critical
**Description**: Verify guest users can complete checkout

### Prerequisites
- Product catalog available
- Payment gateway active

### Test Steps
1. **Navigate to product page**
   _Expected_: Product page loads with details
```

## Dependencies
- Existing Test Planner Agent implementation
- Core data models
- Test Runner Agent (for consumption)

## Risks and Mitigations
- **Risk**: Breaking existing test execution
  - **Mitigation**: Complete replacement on separate branch, thorough testing before merge
- **Risk**: Overly complex test plans
  - **Mitigation**: Clear guidelines and examples in prompts

## Future Enhancements
- Test plan versioning
- Test case dependencies
- Conditional test execution
- Test data management
