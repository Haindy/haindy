"""
Centralized state management for test execution.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from src.core.types import (
    AgentMessage,
    TestPlan,
    TestState,
    TestStatus,
    TestStep,
    ActionResult,
    EvaluationResult,
)
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class StateTransition(str, Enum):
    """Valid state transitions for test execution."""
    
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    COMPLETE_STEP = "complete_step"
    FAIL_STEP = "fail_step"
    SKIP_STEP = "skip_step"
    ABORT = "abort"
    COMPLETE = "complete"


class StateManager:
    """
    Manages the global state of test execution across all agents.
    
    Provides thread-safe state access, transition validation, and
    state persistence for recovery.
    """
    
    def __init__(self):
        """Initialize the state manager."""
        # Current test states by test ID
        self._test_states: Dict[UUID, TestState] = {}
        
        # Test plans by test ID
        self._test_plans: Dict[UUID, TestPlan] = {}
        
        # Step results by test ID
        self._step_results: Dict[UUID, List[Dict[str, Any]]] = {}
        
        # Active agents by test ID
        self._active_agents: Dict[UUID, Set[str]] = {}
        
        # State change history
        self._state_history: List[Dict[str, Any]] = []
        self._history_limit = 1000
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # State change callbacks
        self._state_change_callbacks: List[Any] = []
        
        logger.info("State manager initialized")
    
    async def create_test_state(
        self, 
        test_plan: TestPlan,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> TestState:
        """
        Create a new test state for a test plan.
        
        Args:
            test_plan: The test plan to execute
            initial_context: Optional initial context
            
        Returns:
            Created test state
        """
        async with self._lock:
            # Create test state
            test_state = TestState(
                test_plan=test_plan,
                current_step=None,
                completed_steps=[],
                failed_steps=[],
                skipped_steps=[],
                status=TestStatus.PENDING,
                start_time=None,
                end_time=None,
                error_count=0,
                warning_count=0
            )
            
            # Store state and plan
            self._test_states[test_plan.plan_id] = test_state
            self._test_plans[test_plan.plan_id] = test_plan
            self._step_results[test_plan.plan_id] = []
            self._active_agents[test_plan.plan_id] = set()
            
            # Record state change
            await self._record_state_change(
                test_plan.plan_id,
                StateTransition.START,
                {"status": "created"}
            )
            
            logger.info(f"Test state created for plan: {test_plan.name}", extra={
                "test_id": str(test_plan.plan_id)
            })
            
            return test_state
    
    async def get_test_state(self, test_id: UUID) -> Optional[TestState]:
        """
        Get current test state.
        
        Args:
            test_id: Test plan ID
            
        Returns:
            Current test state or None
        """
        async with self._lock:
            return self._test_states.get(test_id)
    
    async def update_test_state(
        self,
        test_id: UUID,
        transition: StateTransition,
        data: Optional[Dict[str, Any]] = None
    ) -> TestState:
        """
        Update test state with transition validation.
        
        Args:
            test_id: Test plan ID
            transition: State transition to apply
            data: Optional transition data
            
        Returns:
            Updated test state
            
        Raises:
            ValueError: If transition is invalid
        """
        async with self._lock:
            test_state = self._test_states.get(test_id)
            if not test_state:
                raise ValueError(f"Test state not found: {test_id}")
            
            # Validate transition
            if not self._is_valid_transition(test_state, transition):
                raise ValueError(
                    f"Invalid transition {transition} from status {test_state.status}"
                )
            
            # Apply transition
            if transition == StateTransition.START:
                test_state.status = TestStatus.IN_PROGRESS
                test_state.start_time = datetime.now(timezone.utc)
                if test_state.test_plan.steps:
                    test_state.current_step = test_state.test_plan.steps[0]
            
            elif transition == StateTransition.PAUSE:
                test_state.status = TestStatus.SKIPPED
            
            elif transition == StateTransition.RESUME:
                test_state.status = TestStatus.IN_PROGRESS
            
            elif transition == StateTransition.COMPLETE_STEP:
                if data and "step_id" in data:
                    step_id = data["step_id"]
                    test_state.completed_steps.append(step_id)
                    
                    # Move to next step
                    next_step = self._get_next_step(test_state)
                    test_state.current_step = next_step
                    
                    # Store step result
                    if "result" in data:
                        self._step_results[test_id].append(data["result"])
            
            elif transition == StateTransition.FAIL_STEP:
                if data and "step_id" in data:
                    step_id = data["step_id"]
                    test_state.failed_steps.append(step_id)
                    test_state.error_count += 1
                    
                    # Check if critical step
                    if data.get("is_critical", False):
                        test_state.status = TestStatus.FAILED
                        test_state.end_time = datetime.now(timezone.utc)
            
            elif transition == StateTransition.SKIP_STEP:
                if data and "step_id" in data:
                    step_id = data["step_id"]
                    test_state.skipped_steps.append(step_id)
                    test_state.warning_count += 1
            
            elif transition == StateTransition.ABORT:
                test_state.status = TestStatus.FAILED
                test_state.end_time = datetime.now(timezone.utc)
            
            elif transition == StateTransition.COMPLETE:
                test_state.status = TestStatus.PASSED
                test_state.end_time = datetime.now(timezone.utc)
            
            # Record state change
            await self._record_state_change(test_id, transition, data)
            
            # Notify callbacks
            await self._notify_state_change(test_id, test_state, transition)
            
            return test_state
    
    def _is_valid_transition(
        self, 
        test_state: TestState, 
        transition: StateTransition
    ) -> bool:
        """Check if a state transition is valid."""
        current_status = test_state.status
        
        # Define valid transitions
        valid_transitions = {
            TestStatus.PENDING: [
                StateTransition.START,
                StateTransition.ABORT
            ],
            TestStatus.IN_PROGRESS: [
                StateTransition.PAUSE,
                StateTransition.COMPLETE_STEP,
                StateTransition.FAIL_STEP,
                StateTransition.SKIP_STEP,
                StateTransition.ABORT,
                StateTransition.COMPLETE
            ],
            TestStatus.SKIPPED: [],
            TestStatus.PASSED: [],
            TestStatus.FAILED: []
        }
        
        return transition in valid_transitions.get(current_status, [])
    
    def _get_next_step(self, test_state: TestState) -> Optional[TestStep]:
        """Get the next step to execute."""
        if not test_state.test_plan.steps:
            return None
        
        # Get all step IDs
        all_step_ids = [step.step_id for step in test_state.test_plan.steps]
        
        # Find completed/failed/skipped steps
        processed_steps = set(
            test_state.completed_steps + 
            test_state.failed_steps + 
            test_state.skipped_steps
        )
        
        # Find next unprocessed step
        for step in test_state.test_plan.steps:
            if step.step_id not in processed_steps:
                # Check dependencies
                if self._check_dependencies(step, test_state):
                    return step
        
        return None
    
    def _check_dependencies(
        self, 
        step: TestStep, 
        test_state: TestState
    ) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        
        # All dependencies must be completed
        return all(
            dep_id in test_state.completed_steps 
            for dep_id in step.dependencies
        )
    
    async def register_agent(self, test_id: UUID, agent_name: str) -> None:
        """Register an agent as active for a test."""
        async with self._lock:
            if test_id in self._active_agents:
                self._active_agents[test_id].add(agent_name)
                logger.debug(f"Agent {agent_name} registered for test {test_id}")
    
    async def unregister_agent(self, test_id: UUID, agent_name: str) -> None:
        """Unregister an agent from a test."""
        async with self._lock:
            if test_id in self._active_agents:
                self._active_agents[test_id].discard(agent_name)
                logger.debug(f"Agent {agent_name} unregistered from test {test_id}")
    
    async def get_active_agents(self, test_id: UUID) -> Set[str]:
        """Get active agents for a test."""
        async with self._lock:
            return self._active_agents.get(test_id, set()).copy()
    
    async def _record_state_change(
        self,
        test_id: UUID,
        transition: StateTransition,
        data: Optional[Dict[str, Any]]
    ) -> None:
        """Record state change in history."""
        change_record = {
            "test_id": str(test_id),
            "transition": transition,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self._state_history.append(change_record)
        
        # Trim history
        if len(self._state_history) > self._history_limit:
            self._state_history = self._state_history[-self._history_limit:]
    
    async def _notify_state_change(
        self,
        test_id: UUID,
        test_state: TestState,
        transition: StateTransition
    ) -> None:
        """Notify registered callbacks of state change."""
        for callback in self._state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(test_id, test_state, transition)
                else:
                    await asyncio.to_thread(callback, test_id, test_state, transition)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def register_state_callback(self, callback: Any) -> None:
        """Register a callback for state changes."""
        self._state_change_callbacks.append(callback)
        logger.debug("State change callback registered")
    
    def unregister_state_callback(self, callback: Any) -> None:
        """Unregister a state change callback."""
        if callback in self._state_change_callbacks:
            self._state_change_callbacks.remove(callback)
            logger.debug("State change callback unregistered")
    
    async def get_test_progress(self, test_id: UUID) -> Dict[str, Any]:
        """Get test execution progress."""
        async with self._lock:
            test_state = self._test_states.get(test_id)
            if not test_state:
                return {}
            
            total_steps = len(test_state.test_plan.steps)
            completed_steps = len(test_state.completed_steps)
            failed_steps = len(test_state.failed_steps)
            skipped_steps = len(test_state.skipped_steps)
            
            progress = {
                "test_id": str(test_id),
                "test_name": test_state.test_plan.name,
                "status": test_state.status,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "skipped_steps": skipped_steps,
                "progress_percentage": (
                    (completed_steps + failed_steps + skipped_steps) / total_steps * 100
                    if total_steps > 0 else 0
                ),
                "current_step": (
                    test_state.current_step.description 
                    if test_state.current_step else None
                ),
                "error_count": test_state.error_count,
                "warning_count": test_state.warning_count,
                "start_time": (
                    test_state.start_time.isoformat() 
                    if test_state.start_time else None
                ),
                "elapsed_time": (
                    (datetime.now(timezone.utc) - test_state.start_time).total_seconds()
                    if test_state.start_time else 0
                )
            }
            
            return progress
    
    async def get_step_results(self, test_id: UUID) -> List[Dict[str, Any]]:
        """Get all step results for a test."""
        async with self._lock:
            return self._step_results.get(test_id, []).copy()
    
    def get_state_history(
        self, 
        test_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get state change history."""
        history = self._state_history
        
        if test_id:
            history = [
                h for h in history 
                if h.get("test_id") == str(test_id)
            ]
        
        return history[-limit:]
    
    async def cleanup_test(self, test_id: UUID) -> None:
        """Clean up test state and resources."""
        async with self._lock:
            self._test_states.pop(test_id, None)
            self._test_plans.pop(test_id, None)
            self._step_results.pop(test_id, None)
            self._active_agents.pop(test_id, None)
            
            logger.info(f"Test state cleaned up: {test_id}")
    
    async def shutdown(self) -> None:
        """Shutdown state manager and cleanup."""
        logger.info("Shutting down state manager")
        
        async with self._lock:
            self._test_states.clear()
            self._test_plans.clear()
            self._step_results.clear()
            self._active_agents.clear()
            self._state_change_callbacks.clear()
        
        logger.info("State manager shutdown complete")