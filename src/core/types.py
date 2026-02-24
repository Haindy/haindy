"""
Core data models and types for the HAINDY testing framework.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class TestStatus(str, Enum):
    """Status of a test execution."""

    __test__ = False

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ActionType(str, Enum):
    """Types of actions that can be performed."""

    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"  # Deprecated - use specific scroll types
    NAVIGATE = "navigate"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    ASSERT = "assert"
    KEY_PRESS = "key_press"
    # Specific scroll actions
    SCROLL_TO_ELEMENT = "scroll_to_element"
    SCROLL_BY_PIXELS = "scroll_by_pixels"
    SCROLL_TO_TOP = "scroll_to_top"
    SCROLL_TO_BOTTOM = "scroll_to_bottom"
    SCROLL_HORIZONTAL = "scroll_horizontal"
    SKIP_NAVIGATION = "skip_navigation"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent decisions."""

    VERY_HIGH = "very_high"  # 95-100%
    HIGH = "high"  # 80-94%
    MEDIUM = "medium"  # 60-79%
    LOW = "low"  # 40-59%
    VERY_LOW = "very_low"  # 0-39%


class CoordinateReference(BaseModel):
    """Provider-neutral coordinate metadata for an interaction target."""

    target_reference: str | None = Field(
        None,
        description="Provider-neutral target reference when available",
    )
    pixel_coordinates: tuple[int, int] | None = Field(
        None,
        description="Absolute pixel coordinates as (x, y)",
    )
    relative_x: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Normalized horizontal position within the selected target area",
    )
    relative_y: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Normalized vertical position within the selected target area",
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this coordinate reference",
    )
    adjusted: bool = Field(
        False,
        description="Whether post-selection coordinate adjustment was applied",
    )


class ActionInstruction(BaseModel):
    """Instructions for an action to be performed."""

    action_type: ActionType
    description: str = Field(..., description="Human-readable action description")
    target: str | None = Field(None, description="Target element description")
    value: str | None = Field(None, description="Value for type actions")
    expected_outcome: str = Field(..., description="Expected result of the action")
    computer_use_prompt: str | None = Field(
        None,
        description="Fully prepared prompt for the Computer Use executor",
    )
    timeout: int = Field(5000, description="Timeout in milliseconds")


class ResolvedAction(BaseModel):
    """A resolved action with provider-neutral coordinate metadata."""

    instruction: ActionInstruction
    coordinates: CoordinateReference | None = None
    screenshot_before: str | None = Field(
        None, description="Path to screenshot before action"
    )
    fallback_strategy: str | None = Field(
        None, description="Alternative approach if primary fails"
    )


class ActionResult(BaseModel):
    """Result of an executed action."""

    action_id: UUID = Field(default_factory=uuid4)
    success: bool
    action: ResolvedAction
    screenshot_after: str | None = Field(
        None, description="Path to screenshot after action"
    )
    execution_time_ms: int
    error_message: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StepIntent(str, Enum):
    """Describes the goal of a test step for execution-time heuristics."""

    SETUP = "setup"  # prepare state without heavy validation
    VALIDATION = "validation"  # default expectation-driven check
    GROUP_ASSERT = "group_assert"  # combined assertions captured together


class TestStep(BaseModel):
    """A single step in a test case."""

    __test__ = False

    step_id: UUID = Field(default_factory=uuid4)
    step_number: int
    description: str = Field(..., description="Human-readable description of the step")
    action: str = Field(..., description="Action to be performed")
    expected_result: str = Field(..., description="Expected outcome of the action")
    # Keep action_instruction for backward compatibility during transition
    action_instruction: ActionInstruction | None = Field(None, description="Detailed action instruction (deprecated)")
    dependencies: list[int] = Field(
        default_factory=list, description="Step numbers that must complete first"
    )
    optional: bool = Field(False, description="Whether this step can be skipped")
    intent: StepIntent = Field(
        StepIntent.VALIDATION,
        description="Execution intent that guides runner heuristics",
    )
    max_retries: int = Field(3, description="Maximum retry attempts")
    cache_label: str | None = Field(
        None, description="Cache label for coordinate caching and replay"
    )
    cache_action: str = Field(
        "click", description="Cache action type for coordinate caching"
    )
    environment: str | None = Field(
        None, description="Execution environment override (desktop or web)"
    )
    can_be_replayed: bool | None = Field(
        None, description="Allow execution replay cache for this step"
    )
    loop: bool = Field(False, description="Repeat the step until validated")
    scroll_policy: str = Field(
        "auto", description="Scroll policy override (auto/allow/disallow)"
    )
    capture_clipboard: bool = Field(
        False, description="Capture clipboard output during execution"
    )
    clipboard_output_key: str | None = Field(
        None, description="Key to attach clipboard output to action results"
    )


class TestCasePriority(str, Enum):
    """Priority levels for test cases."""

    __test__ = False

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestCase(BaseModel):
    """A test case containing multiple test steps."""

    __test__ = False

    case_id: UUID = Field(default_factory=uuid4)
    test_id: str = Field(..., description="Human-readable test case ID (e.g., TC001)")
    name: str = Field(..., description="Test case name")
    description: str = Field(..., description="Detailed description of what is being tested")
    priority: TestCasePriority = Field(TestCasePriority.MEDIUM, description="Test case priority")
    prerequisites: list[str] = Field(default_factory=list, description="Prerequisites for this test case")
    steps: list[TestStep] = Field(..., description="Ordered list of test steps")
    postconditions: list[str] = Field(default_factory=list, description="Expected state after test completion")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class TestPlan(BaseModel):
    """A complete test plan containing multiple test cases."""

    __test__ = False

    plan_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Test plan name")
    description: str = Field(..., description="Overall test plan description")
    requirements_source: str = Field(..., description="Source of requirements (e.g., PRD v1.2, URL)")
    test_cases: list[TestCase] = Field(..., description="List of test cases in this plan")
    steps: list[TestStep] = Field(
        default_factory=list,
        description="Flattened list of all steps for legacy consumers",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field("HAINDY Test Planner", description="Who/what created this plan")
    tags: list[str] = Field(default_factory=list, description="Overall plan tags")
    estimated_duration_seconds: int | None = Field(None, description="Total estimated duration")

    @model_validator(mode="before")
    @classmethod
    def _synchronize_steps(cls, data: Any) -> Any:
        """Ensure legacy `steps` access works regardless of input structure."""
        if not isinstance(data, dict):
            return data

        test_cases = data.get("test_cases")
        steps = data.get("steps")

        if test_cases:
            aggregated_steps: list[Any] = []
            for case in test_cases:
                if isinstance(case, TestCase):
                    aggregated_steps.extend(case.steps)
                elif isinstance(case, dict):
                    aggregated_steps.extend(case.get("steps", []))
            if aggregated_steps and not steps:
                data["steps"] = aggregated_steps
        elif steps:
            default_case = {
                "test_id": "TC001",
                "name": data.get("name", "Legacy Test Case"),
                "description": data.get(
                    "description", "Test case generated from legacy steps"
                ),
                "priority": TestCasePriority.MEDIUM,
                "prerequisites": [],
                "steps": steps,
                "postconditions": [],
                "tags": [],
            }
            data["test_cases"] = data.get("test_cases") or [default_case]
        else:
            data.setdefault("test_cases", [])
            data.setdefault("steps", [])

        return data


class ScopeTriageResult(BaseModel):
    """Structured summary of scope triage analysis."""

    in_scope: str = Field(
        "",
        description="Plain-language summary of functionality explicitly in scope",
    )
    explicit_exclusions: list[str] = Field(
        default_factory=list,
        description="Items explicitly ruled out of scope",
    )
    ambiguous_points: list[str] = Field(
        default_factory=list,
        description="Requirements or details that remain unclear but are not blockers",
    )
    blocking_questions: list[str] = Field(
        default_factory=list,
        description="Contradictions or missing details that must be resolved before planning",
    )

    def has_blockers(self) -> bool:
        """Return True when unresolved blockers are present."""
        return any(item.strip() for item in self.blocking_questions)

    def build_planner_brief(self) -> str:
        """Generate the curated scope brief for the test planner."""
        sections: list[str] = []

        normalized_scope = self.in_scope.strip()
        if normalized_scope:
            sections.append("IN-SCOPE SUMMARY:\n" + normalized_scope)

        exclusions = [item.strip() for item in self.explicit_exclusions if item.strip()]
        if exclusions:
            exclusions_text = "\n".join(f"- {item}" for item in exclusions)
            sections.append("EXPLICIT EXCLUSIONS:\n" + exclusions_text)

        if not sections:
            return "IN-SCOPE SUMMARY:\nFull scope is permitted. No explicit exclusions noted."

        return "\n\n".join(sections)

    def ambiguous_report(self) -> str | None:
        """Return a formatted ambiguous points report, if any."""
        items = [item.strip() for item in self.ambiguous_points if item.strip()]
        if not items:
            return None
        lines = "\n".join(f"- {item}" for item in items)
        return f"FOLLOW-UP NOTES:\n{lines}"

    def blocking_report(self) -> str | None:
        """Return a formatted blocking questions report, if any."""
        items = [item.strip() for item in self.blocking_questions if item.strip()]
        if not items:
            return None
        lines = "\n".join(f"- {item}" for item in items)
        return f"BLOCKING QUESTIONS:\n{lines}"


class TestState(BaseModel):
    """Current state of test execution."""

    __test__ = False

    test_plan: TestPlan
    current_step: TestStep | None = None
    completed_steps: list[UUID] = Field(default_factory=list)
    failed_steps: list[UUID] = Field(default_factory=list)
    skipped_steps: list[UUID] = Field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    error_count: int = 0
    warning_count: int = 0
    context: dict[str, Any] = Field(default_factory=dict)
    test_report: Optional['TestReport'] = Field(None, description="Comprehensive test execution report")  # Will be populated by TestRunner


class EvaluationResult(BaseModel):
    """Result of evaluating a test step outcome."""

    step_id: UUID
    success: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_outcome: str
    actual_outcome: str
    deviations: list[str] = Field(
        default_factory=list, description="List of deviations from expected"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for next actions"
    )
    screenshot_analysis: dict[str, Any] | None = Field(
        None, description="Detailed screenshot analysis data"
    )


class ExecutionJournal(BaseModel):
    """Detailed journal entry for test execution."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    test_scenario: str
    step_reference: str
    action_taken: str
    coordinate_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-neutral coordinate metadata for visual interactions",
    )
    expected_result: str
    actual_result: str
    agent_confidence: float = Field(..., ge=0.0, le=1.0)
    screenshot_before: str | None = None
    screenshot_after: str | None = None
    execution_time_ms: int
    success: bool
    automation_command: str | None = Field(
        None, description="Recorded Automation command for replay"
    )


class AgentMessage(BaseModel):
    """Message passed between agents."""

    message_id: UUID = Field(default_factory=uuid4)
    from_agent: str
    to_agent: str
    message_type: str
    content: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    requires_response: bool = False
    correlation_id: UUID | None = Field(
        None, description="ID to correlate related messages"
    )


# Scroll-specific models
class ScrollDirection(str, Enum):
    """Direction for scrolling actions."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class VisibilityStatus(str, Enum):
    """Element visibility status for scroll operations."""

    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"


class ScrollParameters(BaseModel):
    """Parameters for scroll actions."""

    direction: ScrollDirection | None = Field(None, description="Scroll direction")
    pixels: int | None = Field(None, description="Number of pixels to scroll")
    target_element: str | None = Field(None, description="Description of element to scroll to")
    max_attempts: int = Field(15, description="Maximum scroll attempts")


class VisibilityResult(BaseModel):
    """Result of element visibility check."""

    status: VisibilityStatus
    coordinates: CoordinateReference | None = Field(
        None,
        description="Coordinate metadata if the target is visible",
    )
    visible_percentage: int | None = Field(None, description="Percentage visible if partial")
    suggested_direction: ScrollDirection | None = Field(None, description="Suggested scroll direction")
    direction_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in direction")
    notes: str = Field("", description="Additional AI observations")


class ScrollAction(BaseModel):
    """A scroll action to be executed."""

    direction: ScrollDirection
    distance: int = Field(..., description="Distance in pixels")
    is_correction: bool = Field(False, description="Whether this is a correction scroll")
    executed_at: datetime | None = None


class ScrollState(BaseModel):
    """State tracking for scroll operations."""

    target_element: str
    attempts: int = 0
    max_attempts: int = 15
    scroll_history: list[ScrollAction] = Field(default_factory=list)
    last_direction: ScrollDirection | None = None
    overshoot_detected: bool = False
    element_partially_visible: bool = False
    last_screenshot_hash: str | None = None


class ScrollResult(BaseModel):
    """Result of a scroll operation."""

    success: bool
    action_type: str = "scroll_to_element"
    coordinates: CoordinateReference | None = None
    confidence: float | None = None
    attempts: int | None = None
    total_scroll_distance: int | None = None
    error: str | None = None
    scroll_history: list[dict[str, Any]] | None = None


class BugSeverity(str, Enum):
    """Severity levels for bug reports."""

    CRITICAL = "critical"  # Blocks test execution
    HIGH = "high"  # Major functionality broken
    MEDIUM = "medium"  # Minor functionality issue
    LOW = "low"  # Cosmetic or edge case


class StepResult(BaseModel):
    """Result of executing a single test step."""

    step_id: UUID
    step_number: int
    status: TestStatus
    started_at: datetime
    completed_at: datetime
    action: str
    expected_result: str
    actual_result: str
    screenshot_before: str | None = Field(None, description="Path to screenshot before action")
    screenshot_after: str | None = Field(None, description="Path to screenshot after action")
    error_message: str | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    actions_performed: list[dict[str, Any]] = Field(default_factory=list, description="List of sub-actions performed")


class BugReport(BaseModel):
    """Detailed bug report for a failed test step."""

    bug_id: UUID = Field(default_factory=uuid4)
    step_id: UUID
    test_case_id: UUID
    test_plan_id: UUID
    step_number: int
    description: str
    severity: BugSeverity
    error_type: str = Field(..., description="Type of error (e.g., 'element_not_found', 'assertion_failed')")
    expected_result: str
    actual_result: str
    screenshot_path: str | None = None
    error_details: str | None = None
    reproduction_steps: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    plan_blocker: bool | None = Field(
        None,
        description="Whether plan-level evaluation marked this issue as blocking",
    )
    plan_blocker_reason: str | None = Field(
        None,
        description="Plan-level rationale for blocking determination",
    )
    plan_recommended_severity: BugSeverity | None = Field(
        None,
        description="Severity suggested by plan-level analysis",
    )
    plan_assessment_notes: str | None = Field(
        None,
        description="Additional notes captured during plan-level assessment",
    )
    plan_recommendations: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions from plan-level assessment",
    )


class TestCaseResult(BaseModel):
    """Result of executing a test case."""

    case_id: UUID
    test_id: str
    name: str
    status: TestStatus
    started_at: datetime
    completed_at: datetime | None = None
    steps_total: int
    steps_completed: int
    steps_failed: int
    step_results: list[StepResult] = Field(default_factory=list)
    bugs: list[BugReport] = Field(default_factory=list)
    error_message: str | None = None


class TestSummary(BaseModel):
    """Summary statistics for test execution."""

    total_test_cases: int
    completed_test_cases: int
    failed_test_cases: int
    total_steps: int
    completed_steps: int
    failed_steps: int
    critical_bugs: int
    high_bugs: int
    medium_bugs: int
    low_bugs: int
    success_rate: float = Field(..., ge=0.0, le=1.0)
    execution_time_seconds: float


class TestReport(BaseModel):
    """Comprehensive test execution report."""

    report_id: UUID = Field(default_factory=uuid4)
    test_plan_id: UUID
    test_plan_name: str
    started_at: datetime
    completed_at: datetime | None = None
    status: TestStatus
    test_cases: list[TestCaseResult] = Field(default_factory=list)
    summary: TestSummary | None = None
    bugs: list[BugReport] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict, description="Test environment details")
    artifacts: dict[str, Any] = Field(
        default_factory=dict, description="Paths to additional run artifacts"
    )
    created_by: str = Field("HAINDY Test Runner", description="Who/what executed the tests")
