"""
Core data models and types for the HAINDY testing framework.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Status of a test execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ActionType(str, Enum):
    """Types of actions that can be performed."""

    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    ASSERT = "assert"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent decisions."""

    VERY_HIGH = "very_high"  # 95-100%
    HIGH = "high"  # 80-94%
    MEDIUM = "medium"  # 60-79%
    LOW = "low"  # 40-59%
    VERY_LOW = "very_low"  # 0-39%


class GridCoordinate(BaseModel):
    """Represents a coordinate in the grid system."""

    cell: str = Field(..., description="Grid cell identifier (e.g., 'M23')")
    offset_x: float = Field(
        0.5, ge=0.0, le=1.0, description="X offset within cell (0.0-1.0)"
    )
    offset_y: float = Field(
        0.5, ge=0.0, le=1.0, description="Y offset within cell (0.0-1.0)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for this coordinate"
    )
    refined: bool = Field(
        False, description="Whether adaptive refinement was applied"
    )


class ActionInstruction(BaseModel):
    """Instructions for an action to be performed."""

    action_type: ActionType
    description: str = Field(..., description="Human-readable action description")
    target: Optional[str] = Field(None, description="Target element description")
    value: Optional[str] = Field(None, description="Value for type actions")
    expected_outcome: str = Field(..., description="Expected result of the action")
    timeout: int = Field(5000, description="Timeout in milliseconds")


class GridAction(BaseModel):
    """A grid-based action to be performed."""

    instruction: ActionInstruction
    coordinate: GridCoordinate
    screenshot_before: Optional[str] = Field(
        None, description="Path to screenshot before action"
    )
    fallback_strategy: Optional[str] = Field(
        None, description="Alternative approach if primary fails"
    )


class ActionResult(BaseModel):
    """Result of an executed action."""

    action_id: UUID = Field(default_factory=uuid4)
    success: bool
    action: GridAction
    screenshot_after: Optional[str] = Field(
        None, description="Path to screenshot after action"
    )
    execution_time_ms: int
    error_message: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestStep(BaseModel):
    """A single step in a test plan."""

    step_id: UUID = Field(default_factory=uuid4)
    step_number: int
    description: str
    action_instruction: ActionInstruction
    dependencies: List[UUID] = Field(
        default_factory=list, description="IDs of steps that must complete first"
    )
    optional: bool = Field(False, description="Whether this step can be skipped")
    max_retries: int = Field(3, description="Maximum retry attempts")


class TestPlan(BaseModel):
    """A complete test plan generated from requirements."""

    plan_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    requirements: str = Field(..., description="Original requirements text")
    steps: List[TestStep]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_duration_seconds: Optional[int] = None
    tags: List[str] = Field(default_factory=list)


class TestState(BaseModel):
    """Current state of test execution."""

    test_plan: TestPlan
    current_step: Optional[TestStep] = None
    completed_steps: List[UUID] = Field(default_factory=list)
    failed_steps: List[UUID] = Field(default_factory=list)
    skipped_steps: List[UUID] = Field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_count: int = 0
    warning_count: int = 0


class EvaluationResult(BaseModel):
    """Result of evaluating a test step outcome."""

    step_id: UUID
    success: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_outcome: str
    actual_outcome: str
    deviations: List[str] = Field(
        default_factory=list, description="List of deviations from expected"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for next actions"
    )
    screenshot_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Detailed screenshot analysis data"
    )


class ExecutionJournal(BaseModel):
    """Detailed journal entry for test execution."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    test_scenario: str
    step_reference: str
    action_taken: str
    grid_coordinates: Dict[str, Any] = Field(
        ..., description="Grid coordinate details including refinement"
    )
    expected_result: str
    actual_result: str
    agent_confidence: float = Field(..., ge=0.0, le=1.0)
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    execution_time_ms: int
    success: bool
    playwright_command: Optional[str] = Field(
        None, description="Recorded Playwright command for replay"
    )


class AgentMessage(BaseModel):
    """Message passed between agents."""

    message_id: UUID = Field(default_factory=uuid4)
    from_agent: str
    to_agent: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    requires_response: bool = False
    correlation_id: Optional[UUID] = Field(
        None, description="ID to correlate related messages"
    )