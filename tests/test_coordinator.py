"""
Tests for the workflow coordinator.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.types import TestPlan, TestState, TestStatus, TestStep, ActionInstruction, ActionType
from src.orchestration.communication import MessageBus
from src.orchestration.coordinator import WorkflowCoordinator, CoordinatorState
from src.orchestration.state_manager import StateManager


@pytest.fixture
def message_bus():
    """Create a MessageBus instance for testing."""
    return MessageBus()


@pytest.fixture
def state_manager():
    """Create a StateManager instance for testing."""
    return StateManager()


@pytest.fixture
def mock_browser_driver():
    """Create a mock browser driver."""
    driver = Mock()
    driver.navigate = AsyncMock()
    driver.take_screenshot = AsyncMock(return_value=b"screenshot")
    return driver


@pytest.fixture
def coordinator(message_bus, state_manager, mock_browser_driver):
    """Create a WorkflowCoordinator instance for testing."""
    return WorkflowCoordinator(
        message_bus=message_bus,
        state_manager=state_manager,
        browser_driver=mock_browser_driver
    )


@pytest.fixture
def sample_test_plan():
    """Create a sample test plan."""
    return TestPlan(
        plan_id=uuid4(),
        name="Test Plan",
        description="Test",
        requirements="Test requirements",
        steps=[
            TestStep(
                step_id=uuid4(),
                step_number=1,
                description="Test step",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click button",
                    expected_outcome="Success"
                )
            )
        ]
    )


class TestWorkflowCoordinator:
    """Test cases for WorkflowCoordinator."""
    
    @pytest.mark.asyncio
    async def test_initialize(self, coordinator):
        """Test coordinator initialization."""
        await coordinator.initialize()
        
        assert coordinator._state == CoordinatorState.IDLE
        assert len(coordinator._agents) == 3  # All agent types (evaluator removed)
        assert "test_planner" in coordinator._agents
        assert "test_runner" in coordinator._agents
        assert "action_agent" in coordinator._agents
        
        # Verify agents are registered with message bus
        stats = coordinator.message_bus.get_statistics()
        assert "test_planner" in stats["registered_agents"]
    
    @pytest.mark.asyncio
    async def test_execute_test_from_requirements(self, coordinator, sample_test_plan):
        """Test executing test from requirements."""
        await coordinator.initialize()
        
        # Mock agent methods
        with patch.object(
            coordinator._agents["test_planner"], 
            'create_test_plan',
            AsyncMock(return_value=sample_test_plan)
        ):
            with patch.object(
                coordinator._agents["test_runner"],
                'execute_test_plan',
                AsyncMock(return_value=TestState(
                    test_plan=sample_test_plan,
                    status=TestStatus.COMPLETED
                ))
            ):
                result = await coordinator.execute_test_from_requirements(
                    "Test login functionality",
                    initial_url="https://example.com"
                )
                
                assert isinstance(result, TestState)
                assert result.status == TestStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_concurrent_test_limit(self, coordinator):
        """Test concurrent test execution limit."""
        await coordinator.initialize()
        coordinator._max_concurrent_tests = 1
        
        # Start first test
        test_task = asyncio.create_task(
            asyncio.sleep(1)  # Simulate long-running test
        )
        coordinator._active_tests[uuid4()] = test_task
        
        # Try to start another test
        with pytest.raises(RuntimeError, match="Maximum concurrent tests"):
            await coordinator.execute_test_from_requirements("Test")
        
        # Cleanup
        test_task.cancel()
        try:
            await test_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_pause_resume_test(self, coordinator):
        """Test pausing and resuming a test."""
        await coordinator.initialize()
        test_id = uuid4()
        
        # Create test state
        test_plan = TestPlan(
            plan_id=test_id,
            name="Test",
            description="Test",
            requirements="Test",
            steps=[]
        )
        await coordinator.state_manager.create_test_state(test_plan)
        await coordinator.state_manager.update_test_state(
            test_id,
            "start"
        )
        
        # Pause test
        await coordinator.pause_test(test_id)
        
        # Verify message published
        messages = coordinator.message_bus.get_message_history(
            message_type="pause_test"
        )
        assert len(messages) > 0
        
        # Resume test
        await coordinator.resume_test(test_id)
        
        messages = coordinator.message_bus.get_message_history(
            message_type="resume_test"
        )
        assert len(messages) > 0
    
    @pytest.mark.asyncio
    async def test_stop_test(self, coordinator):
        """Test stopping a test."""
        await coordinator.initialize()
        test_id = uuid4()
        
        # Create mock test task
        test_task = asyncio.create_task(asyncio.sleep(10))
        coordinator._active_tests[test_id] = test_task
        
        # Create test state
        test_plan = TestPlan(
            plan_id=test_id,
            name="Test",
            description="Test",
            requirements="Test",
            steps=[]
        )
        await coordinator.state_manager.create_test_state(test_plan)
        await coordinator.state_manager.update_test_state(test_id, "start")
        
        # Stop test
        await coordinator.stop_test(test_id)
        
        # Wait a bit for cancellation to complete
        await asyncio.sleep(0.1)
        
        # Verify task cancelled
        assert test_task.done()  # Task should be done (either cancelled or completed)
        assert test_id not in coordinator._active_tests
        
        # Verify message published
        messages = coordinator.message_bus.get_message_history(
            message_type="stop_test"
        )
        assert len(messages) > 0
    
    @pytest.mark.asyncio
    async def test_test_timeout(self, coordinator, sample_test_plan):
        """Test handling test execution timeout."""
        await coordinator.initialize()
        coordinator._agent_timeout = 0.1  # Very short timeout
        
        # Create test state first
        await coordinator.state_manager.create_test_state(sample_test_plan)
        
        # Mock test runner to sleep longer than timeout
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(1)
            return TestState(test_plan=sample_test_plan, status=TestStatus.COMPLETED)
        
        with patch.object(
            coordinator._agents["test_runner"],
            'execute_test_plan',
            slow_execution
        ):
            with pytest.raises(asyncio.TimeoutError):
                await coordinator._execute_test_plan(sample_test_plan)
            
            # Verify test marked as aborted
            test_state = await coordinator.state_manager.get_test_state(
                sample_test_plan.plan_id
            )
            assert test_state.status == TestStatus.FAILED
    
    def test_get_active_tests(self, coordinator):
        """Test getting active test IDs."""
        test_id1 = uuid4()
        test_id2 = uuid4()
        
        coordinator._active_tests[test_id1] = Mock()
        coordinator._active_tests[test_id2] = Mock()
        
        active_tests = coordinator.get_active_tests()
        assert len(active_tests) == 2
        assert test_id1 in active_tests
        assert test_id2 in active_tests
    
    @pytest.mark.asyncio
    async def test_get_test_progress(self, coordinator):
        """Test getting test progress."""
        await coordinator.initialize()
        test_id = uuid4()
        
        # Create test state
        test_plan = TestPlan(
            plan_id=test_id,
            name="Test",
            description="Test",
            requirements="Test",
            steps=[]
        )
        await coordinator.state_manager.create_test_state(test_plan)
        
        progress = await coordinator.get_test_progress(test_id)
        assert progress["test_id"] == str(test_id)
        assert progress["test_name"] == "Test"
    
    def test_get_coordinator_state(self, coordinator):
        """Test getting coordinator state."""
        coordinator._state = CoordinatorState.EXECUTING
        coordinator._active_tests[uuid4()] = Mock()
        
        state = coordinator.get_coordinator_state()
        
        assert state["state"] == CoordinatorState.EXECUTING
        assert state["active_tests"] == 1
        assert "agents" in state
        assert "message_stats" in state
    
    @pytest.mark.asyncio
    async def test_state_change_callback(self, coordinator):
        """Test state change callback handling."""
        await coordinator.initialize()
        
        test_id = uuid4()
        test_state = TestState(
            test_plan=TestPlan(
                plan_id=test_id,
                name="Test",
                description="Test",
                requirements="Test",
                steps=[]
            ),
            status=TestStatus.IN_PROGRESS
        )
        
        # Trigger state change
        await coordinator._on_state_change(test_id, test_state, "start")
        
        # Verify status update message published
        messages = coordinator.message_bus.get_message_history(
            message_type="status_update"
        )
        assert len(messages) > 0
        assert messages[0].content["test_id"] == str(test_id)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, coordinator):
        """Test error handling during test execution."""
        await coordinator.initialize()
        
        # Mock planner to raise error
        with patch.object(
            coordinator._agents["test_planner"],
            'create_test_plan',
            AsyncMock(side_effect=Exception("Test error"))
        ):
            with pytest.raises(Exception, match="Test error"):
                await coordinator.execute_test_from_requirements("Test")
            
            assert coordinator._state == CoordinatorState.ERROR
    
    @pytest.mark.asyncio
    async def test_shutdown(self, coordinator):
        """Test coordinator shutdown."""
        await coordinator.initialize()
        
        # Add active test
        test_task = asyncio.create_task(asyncio.sleep(10))
        coordinator._active_tests[uuid4()] = test_task
        
        # Shutdown
        await coordinator.shutdown()
        
        # Verify cleanup
        assert test_task.done()  # Task should be done
        assert coordinator._state == CoordinatorState.IDLE
        # Active tests cleared after gathering
        assert len(coordinator._active_tests) == 0
    
    @pytest.mark.asyncio
    async def test_agent_subscriptions(self, coordinator):
        """Test agent message subscriptions are set up correctly."""
        await coordinator.initialize()
        
        # Verify subscriptions exist
        stats = coordinator.message_bus.get_statistics()
        assert stats["active_subscriptions"]["plan_test"] > 0
        assert stats["active_subscriptions"]["execute_step"] > 0
        assert stats["active_subscriptions"]["determine_action"] > 0
        # evaluate_result subscription removed with evaluator agent