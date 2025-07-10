# Phase 7 — Agent Coordination

**Tasks**: Inter-agent communication, state management, workflow orchestration.

**ETA**: 2-3 days

**Status**: Completed

## Overview

Phase 7 implements the core coordination framework that enables the four specialized AI agents to work together effectively. This includes inter-agent communication protocols, state management systems, and workflow orchestration mechanisms.

## Multi-Agent Architecture

The system employs four specialized AI agents working in coordination:

1. **Test Planner Agent**: Analyzes requirements/PRDs → Creates structured test plans
2. **Test Runner Agent**: Orchestrates test execution → Decides next steps
3. **Action Agent**: Screenshot + instruction → Adaptive grid coordinates with refinement
4. **Evaluator Agent**: Screenshot + expectation → Success/failure assessment

## Agent Communication Protocol

### Message Passing System
The coordination framework implements structured message passing between agents:

```python
# Example communication flow
{
    "sender": "TestRunnerAgent",
    "recipient": "ActionAgent",
    "message_type": "action_request",
    "payload": {
        "instruction": "Click the 'Add to Cart' button",
        "context": {
            "current_step": 3,
            "previous_action": "navigate_to_product",
            "expected_ui_state": "product_page"
        }
    },
    "timestamp": "2024-01-15T10:30:45.123Z",
    "correlation_id": "test_run_123_step_3"
}
```

### Communication Patterns
- **Request-Response**: Direct agent-to-agent queries
- **Broadcast**: State updates sent to all agents
- **Event-Driven**: Asynchronous notifications
- **Pipeline**: Sequential processing through agent chain

## State Management

### Global Test State
The coordinator maintains a centralized state that all agents can access:

```python
class TestState:
    test_plan: TestPlan
    current_step: int
    completed_steps: List[CompletedStep]
    browser_state: BrowserState
    execution_history: List[ExecutionRecord]
    agent_states: Dict[str, AgentState]
```

### Agent-Specific State
Each agent maintains its own internal state while sharing relevant information:

- **Test Planner**: Test plan versions, requirement interpretations
- **Test Runner**: Execution context, step dependencies, retry counts
- **Action Agent**: Grid coordinates, refinement history, confidence scores
- **Evaluator**: Expected outcomes, validation criteria, success metrics

### State Synchronization
- Eventual consistency model for non-critical state
- Strong consistency for execution flow state
- Conflict resolution through timestamps and precedence rules

## Workflow Orchestration

### Execution Flow Control
The coordinator manages the overall test execution workflow:

1. **Initialization Phase**
   - Load test requirements
   - Initialize all agents
   - Establish communication channels

2. **Planning Phase**
   - Test Planner creates structured plan
   - Test Runner validates plan feasibility
   - Plan optimization and adjustments

3. **Execution Loop**
   - Test Runner selects next action
   - Action Agent executes with refinement
   - Evaluator assesses results
   - State updates and decision making

4. **Completion Phase**
   - Final evaluation
   - Report generation
   - Resource cleanup

### Agent Coordination Patterns

#### Sequential Coordination
```
TestPlanner → TestRunner → ActionAgent → Evaluator → TestRunner (loop)
```

#### Parallel Coordination
- Multiple evaluation requests processed simultaneously
- Concurrent screenshot analysis for efficiency
- Parallel action planning for complex scenarios

#### Conditional Coordination
- Dynamic agent selection based on context
- Fallback chains for error scenarios
- Adaptive workflow based on confidence levels

## Error Handling & Recovery

### Cross-Agent Validation
- Each agent validates inputs from other agents
- Results verified by subsequent agents in the chain
- Consensus mechanisms for conflicting assessments

### Coordination Failures
- Circuit breaker pattern for agent communication
- Timeout handling with graceful degradation
- Fallback to single-agent mode when coordination fails

### Recovery Strategies
- State rollback to last known good state
- Partial result propagation
- Human escalation triggers

## Security & Safety

### Rate Limiting Coordination
- Synchronized API throttling across all agents
- Shared rate limit counters
- Backpressure mechanisms

### Data Protection
- Sensitive data isolation between agents
- Audit trail for all agent communications
- Access control for state modifications

## Observability

### Multi-Agent Logging
- Correlation IDs for tracking requests across agents
- Unified timeline view of agent interactions
- Performance metrics for each agent

### Coordination Metrics
- Agent response times
- Communication overhead
- State synchronization delays
- Workflow completion rates

### Debugging Support
- Agent communication replay
- State history visualization
- Decision tree tracking
- Breakpoint support for workflow debugging