"""
Enhanced data models for improved error reporting and debugging.

This module contains enhanced versions of core types that provide
comprehensive debugging information for test execution.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    """Status of action validation."""
    
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    SKIPPED = "skipped"


class EnhancedActionResult(BaseModel):
    """
    Enhanced action result with comprehensive debugging information.
    
    This replaces the simple ActionResult with detailed information needed
    for debugging failed actions and generating bug reports.
    """
    
    # Basic info
    action_id: UUID = Field(default_factory=uuid4)
    action_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validation phase
    validation_passed: bool
    validation_status: ValidationStatus
    validation_reasoning: str
    validation_confidence: float = Field(0.0, ge=0.0, le=1.0)
    
    # Grid selection phase
    grid_cell: str
    grid_coordinates: Tuple[int, int]
    offset_x: float = Field(0.5, ge=0.0, le=1.0)
    offset_y: float = Field(0.5, ge=0.0, le=1.0)
    coordinate_confidence: float = Field(0.0, ge=0.0, le=1.0)
    coordinate_reasoning: str
    
    # Screenshots and visual evidence
    screenshot_before: Optional[bytes] = None
    screenshot_after: Optional[bytes] = None
    grid_screenshot_before: Optional[bytes] = None  # With grid overlay
    grid_screenshot_highlighted: Optional[bytes] = None  # With selected cell highlighted
    
    # Browser state
    url_before: str
    url_after: str
    page_title_before: str
    page_title_after: str
    
    # Execution details
    execution_success: bool
    execution_time_ms: float
    execution_error: Optional[str] = None
    execution_error_type: Optional[str] = None
    execution_traceback: Optional[str] = None
    
    # AI reasoning and context
    test_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context about the test plan and current step"
    )
    ai_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="AI's analysis of the result"
    )
    
    # Retry information
    attempt_number: int = 1
    max_attempts: int = 3
    retry_reason: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class BugReport(BaseModel):
    """
    Comprehensive bug report for a failed test step.
    
    Generated when an action fails to provide detailed debugging information.
    """
    
    report_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Test information
    test_plan_id: UUID
    test_plan_name: str
    step_number: int
    step_description: str
    
    # What was attempted
    action_attempted: str
    expected_outcome: str
    actual_outcome: str
    
    # Visual evidence
    screenshots: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths to saved screenshots"
    )
    
    # AI reasoning
    ai_reasoning: Dict[str, str] = Field(
        default_factory=dict,
        description="AI's reasoning at each phase"
    )
    
    # Error details
    error_type: str
    error_message: str
    error_traceback: Optional[str] = None
    
    # Browser state
    browser_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Browser state at time of failure"
    )
    
    # Recommendations
    failure_analysis: str
    recommended_fixes: List[str] = Field(default_factory=list)
    
    # Categorization
    severity: str = "medium"  # low, medium, high, critical
    category: str = "unknown"  # ui_change, timing, network, validation, etc.
    
    
class EnhancedTestState(BaseModel):
    """
    Enhanced test state with detailed error tracking.
    
    Extends the basic TestState with comprehensive error information.
    """
    
    # Import base types to avoid circular imports
    from src.core.types import TestPlan, TestStep, TestStatus
    
    # Base state info
    test_plan: TestPlan
    current_step: Optional[TestStep] = None
    completed_steps: List[UUID] = Field(default_factory=list)
    failed_steps: List[UUID] = Field(default_factory=list)
    skipped_steps: List[UUID] = Field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Enhanced error tracking
    error_count: int = 0
    warning_count: int = 0
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed error information"
    )
    warnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Warning information"
    )
    
    # Bug reports
    bug_reports: List[BugReport] = Field(
        default_factory=list,
        description="Generated bug reports for failures"
    )
    
    # Execution history with enhanced results
    execution_history: List[EnhancedActionResult] = Field(
        default_factory=list,
        description="Detailed history of all actions"
    )
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    total_api_calls: int = 0
    total_browser_actions: int = 0
    total_screenshots: int = 0
    
    
class EnhancedTestStepResult(BaseModel):
    """Enhanced result of executing a single test step."""
    
    from src.core.types import TestStep, EvaluationResult
    
    step: TestStep
    success: bool
    action_result: Optional[EnhancedActionResult] = None
    evaluation: Optional[EvaluationResult] = None
    execution_mode: str = "visual"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional debugging info
    debug_info: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)