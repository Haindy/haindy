# HAINDY Agent Architecture Design Document

## Overview
This document defines the conceptual architecture and responsibilities of the multi-agent system in HAINDY. It clarifies the roles, boundaries, and interactions between agents to ensure a cohesive and efficient testing framework.

## Core Philosophy
HAINDY employs a specialized multi-agent architecture where each agent has a focused domain of expertise. Agents collaborate through well-defined interfaces while maintaining independent decision-making capabilities within their domains.

## Agent Roles and Responsibilities

### 1. Test Planner Agent

#### Purpose
The Test Planner Agent is responsible for understanding high-level requirements and transforming them into structured, executable test plans.

#### Core Responsibilities
- **Requirement Analysis**: Parse and understand various input formats (PRDs, user stories, URLs, natural language descriptions)
- **Test Plan Generation**: Create hierarchical test plans with clear structure
- **Coverage Determination**: Ensure comprehensive test coverage based on requirements
- **Priority Assignment**: Determine test case priorities based on criticality

#### Data Model Hierarchy
```
Test Plan
    ├── Metadata (name, description, source, creation date)
    ├── Test Case 1
    │   ├── ID & Name
    │   ├── Description
    │   ├── Priority (Critical/High/Medium/Low)
    │   ├── Prerequisites
    │   ├── Test Steps
    │   │   ├── Step 1 (action, expected result)
    │   │   ├── Step 2 (action, expected result)
    │   │   └── Step N (action, expected result)
    │   └── Post-conditions
    └── Test Case N
```

#### Outputs
1. **JSON Format**: Machine-readable test plan for execution
2. **Markdown Format**: Human-readable documentation for review and understanding

#### Key Design Principles
- **Completeness**: Every requirement should map to at least one test case
- **Clarity**: Each step should be unambiguous and actionable
- **Independence**: Test cases should be executable independently when possible
- **Traceability**: Clear linkage between requirements and test cases

### 2. Test Runner Agent

#### Purpose
The Test Runner Agent orchestrates test execution, maintains test state, generates comprehensive reports, and makes intelligent decisions about test flow.

#### Core Responsibilities

##### A. Test Orchestration
- Navigate through test plans systematically
- Maintain execution context and state
- Coordinate with other agents for task execution

##### B. Step Interpretation and Decomposition
- **Understand Intent**: Comprehend what each test step aims to achieve
- **Action Planning**: Break down high-level steps into executable actions
- **Dynamic Adaptation**: Inject helper actions (scroll, wait) when needed

##### C. Report Management
- **Living Documentation**: Maintain real-time test execution reports
- **Evidence Collection**: Gather screenshots, logs, and execution details
- **Bug Reporting**: Generate detailed bug reports for failures

##### D. Intelligent Decision Making
- **Failure Analysis**: Determine if failures are blockers or can be continued
- **Test Flow Control**: Decide whether to continue, skip, or abort based on results
- **Recovery Strategies**: Attempt intelligent recovery from non-critical failures

#### Execution Workflow
```
1. Initialize Test Report
2. For each Test Case:
   a. Evaluate test case objective and context
   b. For each Step:
      i. Interpret step requirements
      ii. Plan necessary actions
      iii. Execute actions sequentially via Action Agent
      iv. Validate results
      v. Update report with results
   c. On failure:
      i. Analyze failure severity
      ii. Generate bug report if needed
      iii. Determine if test case can continue
   d. Decide on next test case execution
3. Finalize and publish test report
```

#### Failure Handling Matrix
| Failure Type | Severity | Test Case Impact | Test Plan Impact |
|--------------|----------|------------------|------------------|
| Element Not Found | High | Stop if critical | Continue unless dependency |
| Assertion Failed | High | Stop test case | Continue |
| Navigation Error | Critical | Stop test case | Evaluate dependencies |
| Timeout | Medium | Retry or continue | Continue |
| Visual Mismatch | Low | Log and continue | Continue |

### 3. Action Agent

#### Purpose
The Action Agent is responsible for executing specific browser interactions using visual recognition and grid-based coordinate systems.

#### Core Responsibilities
- **Visual Analysis**: Analyze screenshots to locate UI elements
- **Coordinate Mapping**: Convert visual elements to grid coordinates
- **Action Execution**: Perform browser actions (click, type, scroll, etc.)
- **Result Verification**: Confirm actions completed successfully
- **Adaptive Refinement**: Use grid refinement for precision when needed

#### Supported Actions
1. **Navigation**: URL navigation, back/forward
2. **Interaction**: Click, double-click, right-click
3. **Input**: Type text, key combinations
4. **Scrolling**: Vertical/horizontal scroll, scroll to element
5. **Validation**: Visual element presence/absence
6. **Advanced**: Drag-drop, hover, file upload

#### Key Design Principles
- **Visual-First**: No dependency on DOM or CSS selectors
- **Adaptive Precision**: Grid refinement for accurate targeting
- **Reliable Execution**: Built-in retry and validation mechanisms
- **Performance**: Efficient screenshot analysis and action execution

### 4. Evaluator Agent (Under Review)

#### Current Status
The Evaluator Agent is under architectural review as part of Phase 16. Initial analysis suggests that evaluation responsibilities are better distributed among other agents rather than centralized.

#### Proposed Resolution
- **Visual Evaluation**: Integrate into Action Agent (post-action validation)
- **Result Evaluation**: Keep within Test Runner Agent (step success/failure)
- **Assertion Logic**: Distribute based on context and ownership

## Inter-Agent Communication

### Communication Patterns
1. **Request-Response**: Synchronous communication for immediate actions
2. **Event-Driven**: Asynchronous updates for state changes
3. **Shared State**: Centralized state management for test execution context

### Message Flow Example
```
Test Runner → Action Agent: "Click on 'Add to Cart' button"
Action Agent → Test Runner: {success: true, screenshot: "...", details: {...}}
Test Runner → Report: Update step status and evidence
```

### Error Propagation
- Agents provide detailed error information
- Errors include context, screenshots, and recovery suggestions
- Test Runner makes final decisions on error handling

## Design Principles

### 1. Single Responsibility
Each agent has a clearly defined domain and doesn't overlap with others.

### 2. Loose Coupling
Agents interact through well-defined interfaces, not implementation details.

### 3. High Cohesion
Related functionality is grouped within the same agent.

### 4. Fault Tolerance
System continues functioning even if individual actions fail.

### 5. Observability
Every decision and action is logged and traceable.

### 6. Extensibility
New agents or capabilities can be added without major refactoring.

## Future Considerations

### Potential New Agents
1. **Test Data Agent**: Manage test data generation and cleanup
2. **Environment Agent**: Handle test environment setup and configuration
3. **Analytics Agent**: Provide insights from test execution history
4. **Healing Agent**: Auto-fix broken tests using ML/pattern recognition

### Scalability Patterns
1. **Agent Pooling**: Multiple instances of agents for parallel execution
2. **Distributed Execution**: Agents running on different machines
3. **Queue-Based Communication**: For high-volume test execution

## Implementation Timeline
1. **Phase 14**: Test Planner Agent Refinement (2 days)
2. **Phase 15**: Test Runner Agent Enhancement (3 days)
3. **Phase 16**: Evaluator Agent Reassessment (1 day)
4. **Phase 17**: Additional Action Types (Previously Phase 14)

## Conclusion
This architecture provides a clear separation of concerns while enabling sophisticated test automation capabilities. By defining clear boundaries and responsibilities, we ensure that each agent can evolve independently while maintaining system cohesion.