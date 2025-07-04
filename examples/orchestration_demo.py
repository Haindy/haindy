#!/usr/bin/env python3
"""
Demonstration of the Agent Coordination and Orchestration system.

This script shows how the MessageBus, StateManager, and WorkflowCoordinator
work together to enable multi-agent test execution.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import (
    AgentMessage,
    TestPlan,
    TestStep,
    ActionInstruction,
    ActionType,
)
from src.orchestration.communication import MessageBus, MessageType
from src.orchestration.state_manager import StateManager, StateTransition
from src.orchestration.coordinator import WorkflowCoordinator
from src.monitoring.logger import setup_logging


async def demonstrate_message_bus():
    """Demonstrate message bus functionality."""
    print("\n1. Message Bus Demo")
    print("=" * 50)
    
    # Create message bus
    bus = MessageBus()
    
    # Register agents
    print("\nRegistering agents...")
    bus.register_agent("planner")
    bus.register_agent("runner")
    bus.register_agent("evaluator")
    
    # Set up message handlers
    messages_received = []
    
    async def handle_message(msg: AgentMessage):
        messages_received.append(msg)
        print(f"   Handler received: {msg.message_type} from {msg.from_agent}")
    
    # Subscribe to messages
    bus.subscribe(MessageType.PLAN_TEST, handle_message)
    bus.subscribe(MessageType.EXECUTE_STEP, handle_message)
    
    print("\nPublishing messages...")
    
    # Publish some messages
    await bus.publish(AgentMessage(
        from_agent="planner",
        to_agent="broadcast",
        message_type=MessageType.PLAN_TEST,
        content={"requirements": "Test login flow"}
    ))
    
    await bus.publish(AgentMessage(
        from_agent="runner",
        to_agent="evaluator",
        message_type=MessageType.EXECUTE_STEP,
        content={"step": "Click login button"}
    ))
    
    # Get messages for specific agent
    print("\nChecking evaluator's message queue...")
    evaluator_messages = await bus.get_messages("evaluator")
    print(f"   Evaluator has {len(evaluator_messages)} messages")
    
    # Show statistics
    stats = bus.get_statistics()
    print(f"\nMessage Bus Statistics:")
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Registered agents: {stats['registered_agents']}")
    print(f"   Active subscriptions: {sum(stats['active_subscriptions'].values())}")
    
    await bus.shutdown()


async def demonstrate_state_manager():
    """Demonstrate state management functionality."""
    print("\n\n2. State Manager Demo")
    print("=" * 50)
    
    # Create state manager
    manager = StateManager()
    
    # Create a test plan
    test_plan = TestPlan(
        plan_id=uuid4(),
        name="Login Test Demo",
        description="Demonstrate state management",
        requirements="User should be able to login",
        steps=[
            TestStep(
                step_id=uuid4(),
                step_number=1,
                description="Navigate to login",
                action_instruction=ActionInstruction(
                    action_type=ActionType.NAVIGATE,
                    description="Go to login page",
                    expected_outcome="Login page displayed"
                )
            ),
            TestStep(
                step_id=uuid4(),
                step_number=2,
                description="Enter credentials",
                action_instruction=ActionInstruction(
                    action_type=ActionType.TYPE,
                    description="Enter username and password",
                    expected_outcome="Credentials entered"
                )
            ),
            TestStep(
                step_id=uuid4(),
                step_number=3,
                description="Submit login",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click login button",
                    expected_outcome="Dashboard displayed"
                )
            )
        ]
    )
    
    print("\nCreating test state...")
    test_state = await manager.create_test_state(test_plan)
    print(f"   Test ID: {test_plan.plan_id}")
    print(f"   Status: {test_state.status}")
    
    # Start test
    print("\nStarting test execution...")
    test_state = await manager.update_test_state(
        test_plan.plan_id,
        StateTransition.START
    )
    print(f"   Status: {test_state.status}")
    print(f"   Current step: {test_state.current_step.description}")
    
    # Complete first step
    print("\nCompleting first step...")
    test_state = await manager.update_test_state(
        test_plan.plan_id,
        StateTransition.COMPLETE_STEP,
        {"step_id": test_plan.steps[0].step_id}
    )
    print(f"   Completed steps: {len(test_state.completed_steps)}")
    print(f"   Current step: {test_state.current_step.description}")
    
    # Get progress
    progress = await manager.get_test_progress(test_plan.plan_id)
    print(f"\nTest Progress:")
    print(f"   Progress: {progress['progress_percentage']:.0f}%")
    print(f"   Completed: {progress['completed_steps']}/{progress['total_steps']}")
    
    # Simulate pause/resume
    print("\nPausing test...")
    test_state = await manager.update_test_state(
        test_plan.plan_id,
        StateTransition.PAUSE
    )
    print(f"   Status: {test_state.status}")
    
    print("\nResuming test...")
    test_state = await manager.update_test_state(
        test_plan.plan_id,
        StateTransition.RESUME
    )
    print(f"   Status: {test_state.status}")
    
    # Complete remaining steps
    for step in test_plan.steps[1:]:
        await manager.update_test_state(
            test_plan.plan_id,
            StateTransition.COMPLETE_STEP,
            {"step_id": step.step_id}
        )
    
    # Complete test
    print("\nCompleting test...")
    test_state = await manager.update_test_state(
        test_plan.plan_id,
        StateTransition.COMPLETE
    )
    print(f"   Final status: {test_state.status}")
    print(f"   Total time: {(test_state.end_time - test_state.start_time).total_seconds():.2f}s")
    
    await manager.shutdown()


async def demonstrate_coordinator():
    """Demonstrate workflow coordination."""
    print("\n\n3. Workflow Coordinator Demo")
    print("=" * 50)
    
    # Create coordinator
    coordinator = WorkflowCoordinator()
    
    print("\nInitializing coordinator...")
    await coordinator.initialize()
    print("   ✓ Agents created and registered")
    print("   ✓ Message subscriptions set up")
    print("   ✓ State callbacks registered")
    
    # Show coordinator state
    state = coordinator.get_coordinator_state()
    print(f"\nCoordinator State:")
    print(f"   Status: {state['state']}")
    print(f"   Active agents: {state['agents']}")
    print(f"   Active tests: {state['active_tests']}")
    
    # Simulate test execution (without real browser)
    print("\nSimulating test workflow...")
    
    # Mock the agents to avoid real API calls
    from unittest.mock import AsyncMock
    
    # Mock test plan creation
    mock_test_plan = TestPlan(
        plan_id=uuid4(),
        name="Demo Test Plan",
        description="Created by coordinator",
        requirements="Test the login flow",
        steps=[
            TestStep(
                step_id=uuid4(),
                step_number=1,
                description="Login step",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click login",
                    expected_outcome="Success"
                )
            )
        ]
    )
    
    coordinator._agents["test_planner"].create_test_plan = AsyncMock(
        return_value=mock_test_plan
    )
    
    # Mock test execution
    from src.core.types import TestState, TestStatus
    mock_test_state = TestState(
        test_plan=mock_test_plan,
        status=TestStatus.COMPLETED
    )
    
    coordinator._agents["test_runner"].execute_test_plan = AsyncMock(
        return_value=mock_test_state
    )
    
    print("\n   Phase 1: Planning")
    print("   - Analyzing requirements")
    print("   - Generating test plan")
    
    print("\n   Phase 2: Execution")
    print("   - Initializing browser")
    print("   - Running test steps")
    print("   - Evaluating results")
    
    print("\n   Phase 3: Completion")
    print("   - Test completed successfully")
    print("   - Results recorded")
    
    # Show message flow
    print("\n\nMessage Flow During Execution:")
    print("   Coordinator → Test Planner: PLAN_TEST")
    print("   Test Planner → Broadcast: PLAN_CREATED")
    print("   Coordinator → Test Runner: START_TEST")
    print("   Test Runner → Action Agent: DETERMINE_ACTION")
    print("   Action Agent → Test Runner: ACTION_DETERMINED")
    print("   Test Runner → Evaluator: EVALUATE_RESULT")
    print("   Evaluator → Test Runner: EVALUATION_COMPLETE")
    print("   Test Runner → Coordinator: TEST_COMPLETED")
    
    await coordinator.shutdown()


async def demonstrate_integration():
    """Demonstrate integrated multi-agent workflow."""
    print("\n\n4. Integrated Workflow Demo")
    print("=" * 50)
    
    # Create all components
    bus = MessageBus()
    state_manager = StateManager()
    coordinator = WorkflowCoordinator(
        message_bus=bus,
        state_manager=state_manager
    )
    
    print("\nSetting up integrated system...")
    await coordinator.initialize()
    
    # Track message flow
    message_log = []
    
    async def log_messages(msg: AgentMessage):
        message_log.append(f"{msg.from_agent} → {msg.to_agent}: {msg.message_type}")
    
    # Subscribe to all message types
    for msg_type in MessageType:
        bus.subscribe(msg_type, log_messages)
    
    print("\nSimulating complete test execution workflow...")
    
    # Create test plan
    test_id = uuid4()
    await bus.publish(AgentMessage(
        from_agent="user",
        to_agent="coordinator",
        message_type=MessageType.START_TEST,
        content={"test_id": str(test_id), "requirements": "Test checkout flow"}
    ))
    
    # Simulate workflow steps
    workflow_steps = [
        ("coordinator", "test_planner", MessageType.PLAN_TEST),
        ("test_planner", "coordinator", MessageType.PLAN_CREATED),
        ("coordinator", "test_runner", MessageType.EXECUTE_STEP),
        ("test_runner", "action_agent", MessageType.DETERMINE_ACTION),
        ("action_agent", "test_runner", MessageType.ACTION_DETERMINED),
        ("test_runner", "evaluator", MessageType.EVALUATE_RESULT),
        ("evaluator", "test_runner", MessageType.EVALUATION_COMPLETE),
        ("test_runner", "coordinator", MessageType.STEP_COMPLETED),
    ]
    
    for from_agent, to_agent, msg_type in workflow_steps:
        await bus.publish(AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=msg_type,
            content={"test_id": str(test_id)}
        ))
        await asyncio.sleep(0.1)  # Small delay for visualization
    
    print(f"\nMessage Flow ({len(message_log)} messages):")
    for msg in message_log[-10:]:  # Show last 10 messages
        print(f"   {msg}")
    
    # Show final statistics
    print("\n\nFinal System Statistics:")
    bus_stats = bus.get_statistics()
    print(f"   Messages processed: {bus_stats['total_messages']}")
    print(f"   Active agents: {len(bus_stats['registered_agents'])}")
    
    await coordinator.shutdown()


async def main():
    """Run all demonstrations."""
    setup_logging()
    
    print("\nHAINDY Agent Coordination & Orchestration Demo")
    print("=" * 70)
    print("\nThis demo showcases the multi-agent coordination system including:")
    print("- Message bus for inter-agent communication")
    print("- State manager for test execution tracking")
    print("- Workflow coordinator for orchestrating agents")
    print("- Integrated multi-agent workflows")
    
    try:
        # Run each demonstration
        await demonstrate_message_bus()
        await demonstrate_state_manager()
        await demonstrate_coordinator()
        await demonstrate_integration()
        
        print("\n\nDemo Summary")
        print("=" * 50)
        print("✓ Message bus enables async agent communication")
        print("✓ State manager tracks test execution progress")
        print("✓ Coordinator orchestrates multi-agent workflows")
        print("✓ All components integrate seamlessly")
        print("\nThe orchestration system is ready for multi-agent test execution!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())