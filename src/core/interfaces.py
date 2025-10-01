"""
Core interfaces and abstract base classes for the HAINDY framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.core.types import (
    ActionInstruction,
    ActionResult,
    AgentMessage,
    EvaluationResult,
    GridAction,
    GridCoordinate,
    TestPlan,
    TestState,
    TestStep,
)


class Agent(ABC):
    """Abstract base class for all AI agents."""

    def __init__(self, name: str, model: str = "gpt-5") -> None:
        """
        Initialize the agent.

        Args:
            name: Name identifier for the agent
            model: OpenAI model to use
        """
        self.name = name
        self.model = model
        self._message_history: List[AgentMessage] = []

    @abstractmethod
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message and optionally return a response.

        Args:
            message: Incoming message to process

        Returns:
            Optional response message
        """
        pass

    def add_to_history(self, message: AgentMessage) -> None:
        """Add a message to the agent's history."""
        self._message_history.append(message)

    def get_history(self) -> List[AgentMessage]:
        """Get the agent's message history."""
        return self._message_history.copy()


class TestPlannerAgent(Agent):
    """Abstract base class for test planning agents."""

    @abstractmethod
    async def create_test_plan(self, requirements: str) -> TestPlan:
        """
        Create a test plan from high-level requirements.

        Args:
            requirements: Natural language requirements or PRD

        Returns:
            Structured test plan
        """
        pass


class TestRunnerAgent(Agent):
    """Abstract base class for test execution coordination agents."""

    @abstractmethod
    async def get_next_action(
        self, test_plan: TestPlan, current_state: TestState
    ) -> Optional[ActionInstruction]:
        """
        Determine the next action to execute based on current state.

        Args:
            test_plan: The test plan being executed
            current_state: Current execution state

        Returns:
            Next action instruction or None if complete
        """
        pass

    @abstractmethod
    async def evaluate_step_result(
        self, step: TestStep, result: ActionResult
    ) -> TestState:
        """
        Evaluate the result of a test step and update state.

        Args:
            step: The test step that was executed
            result: The execution result

        Returns:
            Updated test state
        """
        pass


class ActionAgent(Agent):
    """Abstract base class for action determination agents."""

    @abstractmethod
    async def determine_action(
        self, screenshot: bytes, instruction: ActionInstruction
    ) -> GridAction:
        """
        Determine grid coordinates for an action from a screenshot.

        Args:
            screenshot: Screenshot of current state
            instruction: Action instruction to execute

        Returns:
            Grid-based action with coordinates
        """
        pass

    @abstractmethod
    async def refine_coordinates(
        self, cropped_region: bytes, initial_coords: GridCoordinate
    ) -> GridCoordinate:
        """
        Refine grid coordinates using adaptive refinement.

        Args:
            cropped_region: Cropped screenshot region
            initial_coords: Initial grid coordinates

        Returns:
            Refined grid coordinates with higher precision
        """
        pass



class BrowserDriver(ABC):
    """Abstract interface for browser automation."""

    @abstractmethod
    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        pass

    @abstractmethod
    async def click(self, x: int, y: int) -> None:
        """Click at absolute coordinates."""
        pass

    @abstractmethod
    async def type_text(self, text: str) -> None:
        """Type text at current focus."""
        pass

    @abstractmethod
    async def scroll(self, direction: str, amount: int) -> None:
        """Scroll in given direction."""
        pass

    @abstractmethod
    async def screenshot(self) -> bytes:
        """Take a screenshot and return as bytes."""
        pass

    @abstractmethod
    async def wait(self, milliseconds: int) -> None:
        """Wait for specified duration."""
        pass

    @abstractmethod
    async def get_viewport_size(self) -> Tuple[int, int]:
        """Get current viewport dimensions."""
        pass


class GridSystem(ABC):
    """Abstract interface for grid overlay system."""

    @abstractmethod
    def initialize(self, width: int, height: int, grid_size: int = 60) -> None:
        """
        Initialize grid with viewport dimensions.

        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
            grid_size: Number of grid cells (default 60x60)
        """
        pass

    @abstractmethod
    def coordinate_to_pixels(self, coord: GridCoordinate) -> Tuple[int, int]:
        """
        Convert grid coordinate to pixel position.

        Args:
            coord: Grid coordinate

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        pass

    @abstractmethod
    def get_cell_bounds(self, cell: str) -> Tuple[int, int, int, int]:
        """
        Get pixel bounds of a grid cell.

        Args:
            cell: Cell identifier (e.g., 'M23')

        Returns:
            Tuple of (x, y, width, height)
        """
        pass

    @abstractmethod
    def create_overlay_image(self, screenshot: bytes) -> bytes:
        """
        Create screenshot with grid overlay for debugging.

        Args:
            screenshot: Original screenshot

        Returns:
            Screenshot with grid overlay
        """
        pass


class TestExecutor(ABC):
    """Abstract interface for test execution orchestration."""

    @abstractmethod
    async def execute_test_plan(self, test_plan: TestPlan) -> TestState:
        """
        Execute a complete test plan.

        Args:
            test_plan: Test plan to execute

        Returns:
            Final test state after execution
        """
        pass

    @abstractmethod
    async def execute_step(
        self, step: TestStep, state: TestState
    ) -> ActionResult:
        """
        Execute a single test step.

        Args:
            step: Test step to execute
            state: Current test state

        Returns:
            Action execution result
        """
        pass


class ConfigProvider(ABC):
    """Abstract interface for configuration management."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass

    @abstractmethod
    def get_required(self, key: str) -> Any:
        """Get required configuration value, raise if missing."""
        pass

    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        pass
