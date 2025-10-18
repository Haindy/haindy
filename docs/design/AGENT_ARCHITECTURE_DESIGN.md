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
- **Context Packaging**: Provide the Action Agent with recent history, plan/case metadata, and decomposed instructions so Computer Use has all necessary signals.

##### C. Report Management
- **Living Documentation**: Maintain real-time test execution reports
- **Evidence Collection**: Gather screenshots, logs, and execution details
- **Bug Reporting**: Generate detailed bug reports for failures
- **Computer Use Traceability**: Persist the conversation transcript, response IDs, and instrumented browser calls for each step.

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
The Action Agent executes browser interactions on behalf of the Test Runner. It combines visual analysis, grid-based coordinate refinement, and the OpenAI Computer Use tool to deliver human-like actions while enforcing project safety policies.

#### Core Responsibilities
- **Computer Use Orchestration**: Drive the OpenAI computer-use model with rich context, manage multi-turn tool loops, and translate model output into concrete browser operations.
- **Visual Analysis & Grid Mapping**: When direct coordinate work is required, analyze screenshots to localise targets on the 60x60 grid and refine positions as needed.
- **Action Execution**: Issue clicks, key presses, typing, scrolls, waits, and assertions through the browser driver, falling back to manual execution if the model omits critical payloads.
- **Mode Management**: Distinguish between execution steps and observe-only assertions, setting the appropriate policy before invoking the computer-use model.
- **Result Packaging**: Capture pre/post states, validation results, AI analysis, and raw computer-use turns for downstream reporting.

#### Supported Actions
1. **Navigation**: URL navigation, back/forward (performed directly without Computer Use).
2. **Computer Use-Driven Interactions**: Click, double-click, right-click, drag, hover, type, key press, wait, screenshot.
3. **Scrolling**: Vertical/horizontal scroll, scroll to element/by pixels.
4. **Validation**: Visual assertions, including observe-only confirmation steps.
5. **Advanced**: Composite flows that stitch together multiple computer-use turns with context reminders.

#### Key Design Principles
- **Computer Use First**: Default to the OpenAI model for stateful UI work, supplementing with synthetic payloads only when the model response is incomplete.
- **Safety by Construction**: Apply domain allowlists/blocklists, enforce observe-only modes, propagate safety identifiers, and stop on fail-fast safety checks.
- **Confirmation Avoidance**: Auto-resolve clarification questions from the model and remind it of the current interaction mode to keep execution progressing without human babysitting.
- **Visual-First**: Maintain the grid/cv toolchain for fallback scenarios and coordinate verification.
- **High Observability**: Persist conversation history, screenshots, and computer-use turn metadata to support debugging.

#### Computer Use Workflow
1. **Goal Construction** – The Action Agent builds a natural-language goal describing the step, target, expected outcome, and an explicit interaction mode (`execute` vs `observe-only`). This message is combined with the fresh screenshot and a context block (plan/case names, step number, current URL, safety identifier).
2. **Model Turn Handling** – Each computer_call is converted into a `ComputerToolTurn`. The agent canonicalises action types, enforces the allowed-action set for the current mode, and rejects unsupported navigation attempts.
3. **Policy Enforcement** – Stateful actions consult the configured allow/block domain lists and respect test-run safety identifiers. Assertion steps supply an observe-only allowlist so the model cannot mutate page state.
4. **Execution & Fallbacks** – If the model omits required payloads (e.g., `keys` for a keypress or `text` for typing), the agent synthesises the missing data from test instructions to ensure the browser command executes or fails loudly.
5. **Clarification Guardrails** – When the model replies with confirmation questions during execute mode, the Action Agent automatically acknowledges with a generic “Yes, proceed” follow-up and resubmits the screenshot, keeping the loop moving.
6. **Trace Capture** – After every turn the agent records screenshots, current URL, latency, metadata, and aggregated response IDs for audit trails and reporting.

#### Safety & Observation Modes
- **Execute Mode** – Used for interaction steps. All stateful actions are permitted unless blocked by domain policy. The agent injects reminders that the model must perform the action without pausing for approval.
- **Observe-Only Mode** – Applied automatically to assertion steps. Only passive actions (`screenshot`, `wait`, `scroll`) are allowed; any attempt to click or type is rejected and logged as policy violations.
- **Safety Identifiers** – Stable identifiers (derived from run/test/case context) are forwarded with every request so OpenAI safety systems can correlate the session.
- **Fail-Fast Safety** – Pending safety checks returned by the model abort the loop immediately, logging the event and surfacing it to the Test Runner.

#### Reporting & Telemetry
- The Action Agent streams conversation history, response IDs, and AI analysis back to the Test Runner, which embeds them in action logs, reports, and bug artifacts.
- Instrumented browser calls (screenshots, key presses, scrolls) are captured per action, enabling replay-style diagnostics alongside the computer-use transcript.

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
