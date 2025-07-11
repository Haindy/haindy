"""
Tests for the WorkflowCoordinator - the central orchestrator for multi-agent test execution.

This test file covers the coordinator's responsibilities:
- Managing agent lifecycle and communication
- Coordinating test execution workflow
- Handling test state transitions
- Managing concurrent test limits
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.types import (
    ActionInstruction, ActionType, AgentMessage,
    TestPlan, TestCase, TestStep, TestState, TestStatus
)
from src.orchestration.communication import MessageBus, MessageType
from src.orchestration.coordinator import WorkflowCoordinator, CoordinatorState
from src.orchestration.state_manager import StateManager


@pytest.fixture
def mock_browser_driver():
    """Create a mock browser driver."""
    driver = AsyncMock()
    driver.navigate_to = AsyncMock()
    driver.capture_screenshot = AsyncMock(return_value=b"screenshot")
    driver.get_url = AsyncMock(return_value="https://example.com")
    driver.get_title = AsyncMock(return_value="Example Page")
    return driver


@pytest.fixture
def coordinator(mock_browser_driver):
    """Create a WorkflowCoordinator instance for testing."""
    return WorkflowCoordinator(
        browser_driver=mock_browser_driver,
        max_steps=50
    )


@pytest.fixture
def sample_test_plan():
    """Create a sample test plan for testing."""
    return TestPlan(
        name="Login Test",
        description="Test user login functionality",
        requirements="User should be able to login with valid credentials",
        test_cases=[
            TestCase(
                name="Valid Login",
                description="Test login with valid credentials",
                priority="high",
                steps=[
                    TestStep(
                        step_number=1,
                        action="Navigate to login page",
                        expected_result="Login page is displayed",
                        action_instruction=ActionInstruction(
                            action_type=ActionType.NAVIGATE,
                            description="Navigate to login page",
                            target="Login page",
                            value="https://example.com/login",
                            expected_outcome="Login page is displayed"
                        )
                    ),
                    TestStep(
                        step_number=2,
                        action="Enter username",
                        expected_result="Username is entered",
                        action_instruction=ActionInstruction(
                            action_type=ActionType.TYPE,
                            description="Enter username",
                            target="Username field",
                            value="testuser",
                            expected_outcome="Username is entered"
                        )
                    )
                ]
            )
        ]
    )


class TestWorkflowCoordinatorInitialization:
    """Test coordinator initialization and setup."""
    
    @pytest.mark.asyncio
    async def test_initialize_creates_agents(self, coordinator):
        """Test that initialization creates all required agents."""
        await coordinator.initialize()
        
        # Verify all agents are created
        assert len(coordinator._agents) == 3
        assert "test_planner" in coordinator._agents
        assert "test_runner" in coordinator._agents
        assert "action_agent" in coordinator._agents
        
        # Verify coordinator state
        assert coordinator._state == CoordinatorState.IDLE
        
        # Verify test runner has action agent reference
        assert coordinator._agents["test_runner"].action_agent == coordinator._agents["action_agent"]
    
    @pytest.mark.asyncio
    async def test_initialize_registers_agents_with_message_bus(self, coordinator):
        """Test that agents are registered with the message bus."""
        await coordinator.initialize()
        
        # Get message bus statistics
        stats = coordinator.message_bus.get_statistics()
        
        # Verify agents are registered
        assert "test_planner" in stats["registered_agents"]
        assert "test_runner" in stats["registered_agents"]
        assert "action_agent" in stats["registered_agents"]
    
    @pytest.mark.asyncio
    async def test_initialize_sets_up_subscriptions(self, coordinator):
        """Test that message subscriptions are set up correctly."""
        await coordinator.initialize()
        
        # Verify subscriptions exist for each agent
        bus = coordinator.message_bus
        
        # Test planner should subscribe to PLAN_TEST
        planner_subs = bus._subscriptions.get(MessageType.PLAN_TEST, {})
        assert any("test_planner" in str(sub) for sub in planner_subs)
        
        # Test runner should subscribe to EXECUTE_STEP
        runner_subs = bus._subscriptions.get(MessageType.EXECUTE_STEP, {})
        assert any("test_runner" in str(sub) for sub in runner_subs)
        
        # Action agent should subscribe to DETERMINE_ACTION
        action_subs = bus._subscriptions.get(MessageType.DETERMINE_ACTION, {})
        assert any("action_agent" in str(sub) for sub in action_subs)


class TestWorkflowCoordinatorTestExecution:
    """Test the main test execution workflow."""
    
    @pytest.mark.asyncio
    async def test_execute_test_from_requirements_success(
        self, coordinator, sample_test_plan
    ):
        """Test successful test execution from requirements."""
        await coordinator.initialize()
        
        # Mock test planner to return our sample plan
        with patch.object(
            coordinator._agents["test_planner"],
            'create_test_plan',
            AsyncMock(return_value=sample_test_plan)
        ):
            # Mock test runner to return completed state
            with patch.object(
                coordinator._agents["test_runner"],
                'execute_test_plan',
                AsyncMock(return_value=TestState(
                    test_plan_id=sample_test_plan.plan_id,
                    test_plan=sample_test_plan,
                    status=TestStatus.COMPLETED,
                    current_test_case_index=0,
                    current_step_index=1
                ))
            ):
                # Execute test
                result = await coordinator.execute_test_from_requirements(
                    "Test login functionality",
                    initial_url="https://example.com"
                )
                
                # Verify result
                assert isinstance(result, TestState)
                assert result.status == TestStatus.COMPLETED
                
                # Verify coordinator state progression
                assert coordinator._state == CoordinatorState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_concurrent_test_limit(self, coordinator):
        """Test that concurrent test limit is enforced."""
        await coordinator.initialize()
        coordinator._max_concurrent_tests = 2
        
        # Create mock active tests
        coordinator._active_tests = {
            uuid4(): AsyncMock(),
            uuid4(): AsyncMock()
        }
        
        # Try to start another test - should raise error
        with pytest.raises(RuntimeError, match="Maximum concurrent tests"):
            await coordinator.execute_test_from_requirements("Another test")
    
    @pytest.mark.asyncio
    async def test_execute_test_handles_planning_failure(self, coordinator):
        """Test handling of test planning failures."""
        await coordinator.initialize()
        
        # Mock test planner to raise error
        with patch.object(
            coordinator._agents["test_planner"],
            'create_test_plan',
            AsyncMock(side_effect=Exception("Failed to generate test plan"))
        ):
            # Execute test - should raise error
            with pytest.raises(Exception, match="Failed to generate test plan"):
                await coordinator.execute_test_from_requirements(
                    "Test requirements"
                )
            
            # Verify coordinator state
            assert coordinator._state == CoordinatorState.ERROR


class TestWorkflowCoordinatorTestControl:
    """Test test control operations (pause, resume, stop)."""
    
    @pytest.mark.asyncio
    async def test_pause_test(self, coordinator):
        """Test pausing a test."""
        await coordinator.initialize()
        test_id = uuid4()
        
        # Create test state
        await coordinator.state_manager.create_test_state(
            TestPlan(
                plan_id=test_id,
                name="Test",
                description="Test",
                requirements="Test"
            )
        )
        
        # Pause test
        await coordinator.pause_test(test_id)
        
        # Verify pause message was published
        messages = coordinator.message_bus.get_message_history(
            message_type=MessageType.PAUSE_TEST
        )
        assert len(messages) > 0
        assert messages[0].data["test_id"] == test_id
    
    @pytest.mark.asyncio
    async def test_resume_test(self, coordinator):
        """Test resuming a paused test."""
        await coordinator.initialize()
        test_id = uuid4()
        
        # Create test state
        await coordinator.state_manager.create_test_state(
            TestPlan(
                plan_id=test_id,
                name="Test",
                description="Test",
                requirements="Test"
            )
        )
        
        # Resume test
        await coordinator.resume_test(test_id)
        
        # Verify resume message was published
        messages = coordinator.message_bus.get_message_history(
            message_type=MessageType.RESUME_TEST
        )
        assert len(messages) > 0
        assert messages[0].data["test_id"] == test_id
    
    @pytest.mark.asyncio
    async def test_stop_test(self, coordinator):
        """Test stopping a running test."""
        await coordinator.initialize()
        test_id = uuid4()
        
        # Create mock active test
        test_task = asyncio.create_task(asyncio.sleep(10))
        coordinator._active_tests[test_id] = test_task
        
        # Create test state
        await coordinator.state_manager.create_test_state(
            TestPlan(
                plan_id=test_id,
                name="Test",
                description="Test",
                requirements="Test"
            )
        )
        
        # Stop test
        await coordinator.stop_test(test_id)
        
        # Verify task was cancelled
        assert test_task.cancelled()
        assert test_id not in coordinator._active_tests
        
        # Verify stop message was published
        messages = coordinator.message_bus.get_message_history(
            message_type=MessageType.STOP_TEST
        )
        assert len(messages) > 0


class TestWorkflowCoordinatorStateAndProgress:
    """Test state management and progress tracking."""
    
    def test_get_active_tests(self, coordinator):
        """Test getting list of active tests."""
        # Add some mock active tests
        test_ids = [uuid4(), uuid4()]
        for test_id in test_ids:
            coordinator._active_tests[test_id] = AsyncMock()
        
        # Get active tests
        active = coordinator.get_active_tests()
        
        assert len(active) == 2
        assert all(test_id in active for test_id in test_ids)
    
    @pytest.mark.asyncio
    async def test_get_test_progress(self, coordinator):
        """Test getting test progress information."""
        await coordinator.initialize()
        test_id = uuid4()
        
        # Create test state
        test_plan = TestPlan(
            plan_id=test_id,
            name="Test",
            description="Test",
            requirements="Test",
            test_cases=[
                TestCase(
                    name="Case 1",
                    description="Test case",
                    steps=[
                        TestStep(
                            step_number=1,
                            action="Step 1",
                            expected_result="Result 1"
                        ),
                        TestStep(
                            step_number=2,
                            action="Step 2",
                            expected_result="Result 2"
                        )
                    ]
                )
            ]
        )
        
        await coordinator.state_manager.create_test_state(test_plan)
        await coordinator.state_manager.update_test_state(
            test_id,
            "start",
            {"current_test_case_index": 0, "current_step_index": 1}
        )
        
        # Get progress
        progress = await coordinator.get_test_progress(test_id)
        
        assert progress["test_id"] == test_id
        assert progress["status"] == TestStatus.IN_PROGRESS
        assert progress["current_test_case"] == 0
        assert progress["current_step"] == 1
        assert progress["total_test_cases"] == 1
        assert progress["total_steps"] == 2
        assert progress["progress_percentage"] == 50.0
    
    def test_get_coordinator_state(self, coordinator):
        """Test getting coordinator state information."""
        # Set some state
        coordinator._state = CoordinatorState.EXECUTING
        coordinator._active_tests = {uuid4(): AsyncMock()}
        
        # Get state
        state = coordinator.get_coordinator_state()
        
        assert state["state"] == CoordinatorState.EXECUTING
        assert state["active_tests"] == 1
        assert state["max_concurrent_tests"] == 5


class TestWorkflowCoordinatorCleanup:
    """Test cleanup and shutdown operations."""
    
    @pytest.mark.asyncio
    async def test_cleanup_cancels_active_tests(self, coordinator):
        """Test that cleanup cancels active tests."""
        await coordinator.initialize()
        
        # Create mock active tests
        test_tasks = [
            asyncio.create_task(asyncio.sleep(10)),
            asyncio.create_task(asyncio.sleep(10))
        ]
        
        for i, task in enumerate(test_tasks):
            coordinator._active_tests[uuid4()] = task
        
        # Cleanup
        await coordinator.cleanup()
        
        # Verify all tasks cancelled
        assert all(task.cancelled() for task in test_tasks)
        assert len(coordinator._active_tests) == 0
    
    @pytest.mark.asyncio
    async def test_generate_test_plan_method(self, coordinator, sample_test_plan):
        """Test the public generate_test_plan method."""
        await coordinator.initialize()
        
        # Mock test planner
        with patch.object(
            coordinator._agents["test_planner"],
            'create_test_plan',
            AsyncMock(return_value=sample_test_plan)
        ):
            # Generate plan
            plan = await coordinator.generate_test_plan("Test requirements")
            
            # Verify
            assert plan == sample_test_plan
            assert plan.name == "Login Test"