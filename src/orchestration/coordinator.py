"""
Workflow coordinator for managing multi-agent test execution.
"""

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from src.agents import (
    TestPlannerAgent,
    TestRunnerAgent,
    ActionAgent,
    EvaluatorAgent,
)
from src.browser.driver import BrowserDriver
from src.core.types import AgentMessage, TestPlan, TestState, TestStatus
from src.orchestration.communication import MessageBus, MessageType
from src.orchestration.state_manager import StateManager, StateTransition
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class CoordinatorState(str, Enum):
    """States of the workflow coordinator."""
    
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class WorkflowCoordinator:
    """
    Central coordinator for multi-agent test execution workflows.
    
    Manages the overall test execution lifecycle, coordinates agents,
    and ensures smooth workflow progression.
    """
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        state_manager: Optional[StateManager] = None,
        browser_driver: Optional[BrowserDriver] = None
    ):
        """
        Initialize the workflow coordinator.
        
        Args:
            message_bus: Message bus for agent communication
            state_manager: State manager for test execution
            browser_driver: Browser driver for web automation
        """
        self.message_bus = message_bus or MessageBus()
        self.state_manager = state_manager or StateManager()
        self.browser_driver = browser_driver
        
        # Agent instances
        self._agents: Dict[str, Any] = {}
        self._agent_tasks: Dict[str, asyncio.Task] = {}
        
        # Coordinator state
        self._state = CoordinatorState.IDLE
        self._active_tests: Dict[UUID, asyncio.Task] = {}
        
        # Configuration
        self._max_concurrent_tests = 5
        self._agent_timeout = 300  # 5 minutes
        
        logger.info("Workflow coordinator initialized")
    
    async def initialize(self) -> None:
        """Initialize the coordinator and create agent instances."""
        logger.info("Initializing workflow coordinator")
        
        # Create agents
        self._agents = {
            "test_planner": TestPlannerAgent(name="TestPlanner"),
            "test_runner": TestRunnerAgent(
                name="TestRunner",
                browser_driver=self.browser_driver
            ),
            "action_agent": ActionAgent(name="ActionAgent"),
            "evaluator_agent": EvaluatorAgent(name="Evaluator")
        }
        
        # Set up test runner dependencies
        if "test_runner" in self._agents:
            self._agents["test_runner"].action_agent = self._agents["action_agent"]
            self._agents["test_runner"].evaluator_agent = self._agents["evaluator_agent"]
        
        # Register agents with message bus
        for agent_name, agent in self._agents.items():
            self.message_bus.register_agent(agent_name)
            
            # Subscribe to relevant messages
            await self._setup_agent_subscriptions(agent_name, agent)
        
        # Register state callbacks
        self.state_manager.register_state_callback(self._on_state_change)
        
        self._state = CoordinatorState.IDLE
        logger.info("Workflow coordinator initialized successfully")
    
    async def _setup_agent_subscriptions(self, agent_name: str, agent: Any) -> None:
        """Set up message subscriptions for an agent."""
        if agent_name == "test_planner":
            self.message_bus.subscribe(
                MessageType.PLAN_TEST,
                lambda msg: asyncio.create_task(self._handle_plan_request(msg)),
                agent_name
            )
        
        elif agent_name == "test_runner":
            self.message_bus.subscribe(
                MessageType.EXECUTE_STEP,
                lambda msg: asyncio.create_task(self._handle_execute_step(msg)),
                agent_name
            )
        
        elif agent_name == "action_agent":
            self.message_bus.subscribe(
                MessageType.DETERMINE_ACTION,
                lambda msg: asyncio.create_task(self._handle_determine_action(msg)),
                agent_name
            )
        
        elif agent_name == "evaluator_agent":
            self.message_bus.subscribe(
                MessageType.EVALUATE_RESULT,
                lambda msg: asyncio.create_task(self._handle_evaluate_result(msg)),
                agent_name
            )
    
    async def execute_test_from_requirements(
        self,
        requirements: str,
        initial_url: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TestState:
        """
        Execute a test from high-level requirements.
        
        Args:
            requirements: Natural language test requirements
            initial_url: Optional starting URL
            context: Optional execution context
            
        Returns:
            Final test state
        """
        logger.info("Starting test execution from requirements")
        
        # Check concurrent test limit
        if len(self._active_tests) >= self._max_concurrent_tests:
            raise RuntimeError(
                f"Maximum concurrent tests ({self._max_concurrent_tests}) reached"
            )
        
        try:
            # Phase 1: Generate test plan
            self._state = CoordinatorState.PLANNING
            test_plan = await self._generate_test_plan(requirements)
            
            # Phase 2: Initialize test state
            test_state = await self.state_manager.create_test_state(
                test_plan, 
                context
            )
            
            # Phase 3: Execute test
            self._state = CoordinatorState.EXECUTING
            final_state = await self._execute_test_plan(
                test_plan, 
                initial_url
            )
            
            self._state = CoordinatorState.COMPLETED
            return final_state
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            self._state = CoordinatorState.ERROR
            raise
    
    async def _generate_test_plan(self, requirements: str) -> TestPlan:
        """Generate test plan using Test Planner agent."""
        logger.info("Generating test plan from requirements")
        
        planner = self._agents.get("test_planner")
        if not planner:
            raise RuntimeError("Test Planner agent not available")
        
        # Send plan request
        message = AgentMessage(
            from_agent="coordinator",
            to_agent="test_planner",
            message_type=MessageType.PLAN_TEST,
            content={"requirements": requirements}
        )
        
        await self.message_bus.publish(message)
        
        # Create test plan
        test_plan = await planner.create_test_plan(requirements)
        
        # Notify plan created
        await self.message_bus.publish(
            AgentMessage(
                from_agent="test_planner",
                to_agent="broadcast",
                message_type=MessageType.PLAN_CREATED,
                content={"test_plan": test_plan.model_dump()}
            )
        )
        
        return test_plan
    
    async def _execute_test_plan(
        self,
        test_plan: TestPlan,
        initial_url: Optional[str] = None
    ) -> TestState:
        """Execute test plan using Test Runner agent."""
        logger.info(f"Executing test plan: {test_plan.name}")
        
        runner = self._agents.get("test_runner")
        if not runner:
            raise RuntimeError("Test Runner agent not available")
        
        # Create execution task
        test_task = asyncio.create_task(
            runner.execute_test_plan(test_plan, initial_url)
        )
        
        # Track active test
        self._active_tests[test_plan.plan_id] = test_task
        
        try:
            # Wait for completion with timeout
            final_state = await asyncio.wait_for(
                test_task,
                timeout=self._agent_timeout
            )
            
            return final_state
            
        except asyncio.TimeoutError:
            logger.error("Test execution timed out")
            test_task.cancel()
            
            # Update state to failed
            await self.state_manager.update_test_state(
                test_plan.plan_id,
                StateTransition.ABORT,
                {"reason": "timeout"}
            )
            
            raise
            
        finally:
            # Remove from active tests
            self._active_tests.pop(test_plan.plan_id, None)
    
    async def pause_test(self, test_id: UUID) -> None:
        """Pause an executing test."""
        logger.info(f"Pausing test: {test_id}")
        
        # Update state
        await self.state_manager.update_test_state(
            test_id,
            StateTransition.PAUSE
        )
        
        # Send pause message
        await self.message_bus.publish(
            AgentMessage(
                from_agent="coordinator",
                to_agent="test_runner",
                message_type=MessageType.PAUSE_TEST,
                content={"test_id": str(test_id)}
            )
        )
    
    async def resume_test(self, test_id: UUID) -> None:
        """Resume a paused test."""
        logger.info(f"Resuming test: {test_id}")
        
        # Update state
        await self.state_manager.update_test_state(
            test_id,
            StateTransition.RESUME
        )
        
        # Send resume message
        await self.message_bus.publish(
            AgentMessage(
                from_agent="coordinator",
                to_agent="test_runner",
                message_type=MessageType.RESUME_TEST,
                content={"test_id": str(test_id)}
            )
        )
    
    async def stop_test(self, test_id: UUID) -> None:
        """Stop an executing test."""
        logger.info(f"Stopping test: {test_id}")
        
        # Cancel test task if running
        test_task = self._active_tests.get(test_id)
        if test_task and not test_task.done():
            test_task.cancel()
        
        # Remove from active tests
        self._active_tests.pop(test_id, None)
        
        # Update state
        await self.state_manager.update_test_state(
            test_id,
            StateTransition.ABORT,
            {"reason": "user_requested"}
        )
        
        # Send stop message
        await self.message_bus.publish(
            AgentMessage(
                from_agent="coordinator",
                to_agent="broadcast",
                message_type=MessageType.STOP_TEST,
                content={"test_id": str(test_id)}
            )
        )
    
    async def _handle_plan_request(self, message: AgentMessage) -> None:
        """Handle test plan request."""
        logger.debug("Handling plan request")
        # Implementation handled by direct agent call
        pass
    
    async def _handle_execute_step(self, message: AgentMessage) -> None:
        """Handle step execution request."""
        logger.debug("Handling execute step")
        # Implementation handled by Test Runner
        pass
    
    async def _handle_determine_action(self, message: AgentMessage) -> None:
        """Handle action determination request."""
        logger.debug("Handling determine action")
        # Implementation handled by Action Agent
        pass
    
    async def _handle_evaluate_result(self, message: AgentMessage) -> None:
        """Handle result evaluation request."""
        logger.debug("Handling evaluate result")
        # Implementation handled by Evaluator Agent
        pass
    
    async def _on_state_change(
        self,
        test_id: UUID,
        test_state: TestState,
        transition: StateTransition
    ) -> None:
        """Handle test state changes."""
        logger.info(
            f"Test state changed: {test_id} - {transition}",
            extra={"status": test_state.status}
        )
        
        # Broadcast state change
        await self.message_bus.publish(
            AgentMessage(
                from_agent="coordinator",
                to_agent="broadcast",
                message_type=MessageType.STATUS_UPDATE,
                content={
                    "test_id": str(test_id),
                    "status": test_state.status,
                    "transition": transition
                }
            )
        )
    
    def get_active_tests(self) -> List[UUID]:
        """Get list of active test IDs."""
        return list(self._active_tests.keys())
    
    async def get_test_progress(self, test_id: UUID) -> Dict[str, Any]:
        """Get progress for a specific test."""
        return await self.state_manager.get_test_progress(test_id)
    
    def get_coordinator_state(self) -> Dict[str, Any]:
        """Get current coordinator state."""
        return {
            "state": self._state,
            "active_tests": len(self._active_tests),
            "agents": list(self._agents.keys()),
            "message_stats": self.message_bus.get_statistics()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the coordinator and cleanup resources."""
        logger.info("Shutting down workflow coordinator")
        
        # Cancel all active tests
        for test_id, task in self._active_tests.items():
            if not task.done():
                logger.warning(f"Cancelling active test: {test_id}")
                task.cancel()
        
        # Wait for tasks to complete
        if self._active_tests:
            await asyncio.gather(
                *self._active_tests.values(),
                return_exceptions=True
            )
        
        # Clear active tests
        self._active_tests.clear()
        
        # Shutdown components
        await self.message_bus.shutdown()
        await self.state_manager.shutdown()
        
        self._state = CoordinatorState.IDLE
        logger.info("Workflow coordinator shutdown complete")