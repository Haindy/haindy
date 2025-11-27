"""
Enhanced data models for comprehensive action execution tracking.

These models provide detailed information for debugging and error reporting,
capturing the full lifecycle of action execution from validation through result analysis.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.core.types import ActionInstruction, GridCoordinate, TestStep, TestState


class ValidationResult(BaseModel):
    """Result of action validation phase."""
    
    valid: bool = Field(..., description="Whether the action is valid to execute")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Validation confidence score")
    reasoning: str = Field(..., description="Detailed explanation of validation decision")
    concerns: List[str] = Field(default_factory=list, description="List of validation concerns")
    suggestions: List[str] = Field(
        default_factory=list, 
        description="Alternative approaches if validation failed"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoordinateResult(BaseModel):
    """Result of coordinate determination phase."""
    
    grid_cell: str = Field(..., description="Grid cell identifier (e.g., 'M23')")
    grid_coordinates: tuple[int, int] = Field(..., description="Pixel coordinates (x, y)")
    offset_x: float = Field(..., ge=0.0, le=1.0, description="X offset within cell")
    offset_y: float = Field(..., ge=0.0, le=1.0, description="Y offset within cell")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Coordinate confidence score")
    reasoning: str = Field("", description="Explanation of coordinate selection")
    refined: bool = Field(False, description="Whether adaptive refinement was applied")
    refinement_details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Details about refinement process if applied"
    )


class ExecutionResult(BaseModel):
    """Result of action execution phase."""
    
    success: bool = Field(..., description="Whether the action executed successfully")
    execution_time_ms: float = Field(..., description="Execution duration in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    error_traceback: Optional[str] = Field(None, description="Full error traceback if available")
    browser_logs: List[str] = Field(default_factory=list, description="Browser console logs")
    network_activity: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Network requests during execution"
    )


class AIAnalysis(BaseModel):
    """AI analysis of action results."""
    
    success: bool = Field(..., description="AI assessment of action success")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence score")
    actual_outcome: str = Field(..., description="Description of what actually happened")
    matches_expected: bool = Field(..., description="Whether outcome matches expectations")
    ui_changes: List[str] = Field(
        default_factory=list, 
        description="List of observed UI changes"
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="AI recommendations for next steps"
    )
    anomalies: List[str] = Field(
        default_factory=list, 
        description="Unexpected behaviors or UI states detected"
    )


class BrowserState(BaseModel):
    """Browser state at a point in time."""
    
    url: str = Field(..., description="Current page URL")
    title: str = Field(..., description="Page title")
    viewport_size: tuple[int, int] = Field(..., description="Viewport dimensions (width, height)")
    screenshot: Optional[bytes] = Field(None, description="Screenshot data")
    screenshot_path: Optional[str] = Field(None, description="Path to saved screenshot")
    dom_ready_state: Optional[str] = Field(None, description="Document ready state")
    active_element: Optional[str] = Field(None, description="Currently focused element")


class ComputerToolTurn(BaseModel):
    """Record of a single Computer Use tool action and its execution details."""

    call_id: str = Field(..., description="OpenAI computer_call identifier")
    action_type: str = Field(..., description="Type of action requested by the model")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw action payload supplied by the model",
    )
    status: str = Field(
        "pending",
        description="Execution status (pending|executed|failed|skipped)",
    )
    error_message: Optional[str] = Field(
        None, description="Error captured during execution or validation"
    )
    screenshot_path: Optional[str] = Field(
        None, description="Path to screenshot captured after this action"
    )
    response_id: Optional[str] = Field(
        None, description="Response ID that produced this action"
    )
    pending_safety_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Safety checks returned alongside this action",
    )
    acknowledged: bool = Field(
        False, description="Whether pending safety checks were acknowledged"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the action was processed",
    )
    latency_ms: Optional[float] = Field(
        None, description="Latency of executing the action in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for observability"
    )


class SafetyEvent(BaseModel):
    """Represents a safety check event encountered during Computer Use execution."""

    call_id: str = Field(..., description="Call ID associated with the safety event")
    code: str = Field(..., description="Safety check code")
    message: str = Field(..., description="Safety check message")
    acknowledged: bool = Field(
        False, description="Whether the event was acknowledged by the system"
    )
    acknowledged_ids: List[str] = Field(
        default_factory=list,
        description="IDs of safety checks acknowledged in follow-up payloads",
    )
    response_id: Optional[str] = Field(
        None, description="Response ID where the safety event originated"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the safety event was recorded",
    )


class EnhancedActionResult(BaseModel):
    """
    Comprehensive result of action execution with full debugging information.
    
    This model captures the complete lifecycle of an action from validation
    through execution and analysis, providing rich context for debugging
    and error reporting.
    """
    
    # Identifiers
    action_id: UUID = Field(default_factory=uuid4)
    test_step_id: UUID = Field(..., description="ID of the test step being executed")
    
    # Timestamps
    timestamp_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp_end: Optional[datetime] = Field(None)
    
    # Original request
    test_step: TestStep = Field(..., description="Original test step")
    test_context: Dict[str, Any] = Field(..., description="Test execution context")
    
    # Validation phase
    validation: ValidationResult = Field(..., description="Validation phase results")
    
    # Coordinate determination
    coordinates: Optional[CoordinateResult] = Field(
        None, 
        description="Coordinate determination results"
    )
    
    # Browser states
    browser_state_before: Optional[BrowserState] = Field(
        None, 
        description="Browser state before action"
    )
    browser_state_after: Optional[BrowserState] = Field(
        None, 
        description="Browser state after action"
    )
    
    # Grid screenshots
    grid_screenshot_before: Optional[bytes] = Field(
        None, 
        description="Screenshot with grid overlay before action"
    )
    grid_screenshot_highlighted: Optional[bytes] = Field(
        None, 
        description="Screenshot with selected cell highlighted"
    )
    
    # Execution
    execution: Optional[ExecutionResult] = Field(
        None, 
        description="Execution phase results"
    )
    
    # AI Analysis
    ai_analysis: Optional[AIAnalysis] = Field(
        None, 
        description="AI analysis of results"
    )
    computer_actions: List[ComputerToolTurn] = Field(
        default_factory=list,
        description="Sequence of Computer Use tool actions executed",
    )
    safety_events: List[SafetyEvent] = Field(
        default_factory=list,
        description="Safety check events encountered during execution",
    )
    final_model_output: Optional[str] = Field(
        None,
        description="Final assistant message returned by the model after completion",
    )
    response_ids: List[str] = Field(
        default_factory=list,
        description="OpenAI response IDs involved in this action loop",
    )
    
    # Overall status
    overall_success: bool = Field(
        False, 
        description="Whether the entire action completed successfully"
    )
    failure_phase: Optional[str] = Field(
        None, 
        description="Which phase failed: validation|coordinates|execution|analysis"
    )
    
    def dict_for_compatibility(self) -> Dict[str, Any]:
        """
        Convert to dictionary format compatible with existing code.
        
        This method provides backward compatibility while we transition
        to the new model structure.
        """
        return {
            "action_type": "click",  # Default since we don't have action_type in new structure
            "timestamp": self.timestamp_start.isoformat(),
            "validation_passed": self.validation.valid,
            "validation_reasoning": self.validation.reasoning,
            "validation_confidence": self.validation.confidence,
            "grid_cell": self.coordinates.grid_cell if self.coordinates else "",
            "grid_coordinates": self.coordinates.grid_coordinates if self.coordinates else (0, 0),
            "offset_x": self.coordinates.offset_x if self.coordinates else 0.5,
            "offset_y": self.coordinates.offset_y if self.coordinates else 0.5,
            "coordinate_confidence": self.coordinates.confidence if self.coordinates else 0.0,
            "coordinate_reasoning": self.coordinates.reasoning if self.coordinates else "",
            "execution_success": self.execution.success if self.execution else False,
            "execution_time_ms": self.execution.execution_time_ms if self.execution else 0.0,
            "execution_error": self.execution.error_message if self.execution else None,
            "url_before": self.browser_state_before.url if self.browser_state_before else "",
            "url_after": self.browser_state_after.url if self.browser_state_after else "",
            "page_title_before": self.browser_state_before.title if self.browser_state_before else "",
            "page_title_after": self.browser_state_after.title if self.browser_state_after else "",
            "screenshot_before": self.browser_state_before.screenshot if self.browser_state_before else None,
            "screenshot_after": self.browser_state_after.screenshot if self.browser_state_after else None,
            "grid_screenshot_before": self.grid_screenshot_before,
            "grid_screenshot_highlighted": self.grid_screenshot_highlighted,
            "test_context": self.test_context,
            "ai_analysis": self.ai_analysis.model_dump() if self.ai_analysis else {},
            "computer_actions": [
                action.model_dump() for action in self.computer_actions
            ],
            "safety_events": [event.model_dump() for event in self.safety_events],
            "final_model_output": self.final_model_output,
            "response_ids": self.response_ids,
        }


class ActionPattern(BaseModel):
    """
    Reusable pattern for successful actions.
    
    Used for caching and optimizing repeated actions.
    """
    
    pattern_id: UUID = Field(default_factory=uuid4)
    action_type: str = Field(..., description="Type of action")
    target_description: str = Field(..., description="Description of target element")
    grid_coordinates: CoordinateResult = Field(..., description="Successful coordinates")
    playwright_command: Optional[str] = Field(
        None, 
        description="Recorded Playwright command for direct replay"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence")
    success_count: int = Field(0, description="Number of successful uses")
    failure_count: int = Field(0, description="Number of failed attempts")
    last_used: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    screenshot_hash: Optional[str] = Field(
        None, 
        description="Hash of screenshot for visual similarity matching"
    )


class BugReport(BaseModel):
    """
    Comprehensive bug report for failed test steps.
    
    Captures all relevant information for debugging test failures,
    including screenshots, AI reasoning, and execution details.
    """
    
    bug_id: UUID = Field(default_factory=uuid4, description="Unique bug identifier")
    test_step: TestStep = Field(..., description="The test step that failed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Failure details
    failure_type: str = Field(..., description="Type of failure: validation|execution|evaluation")
    error_message: str = Field(..., description="Primary error message")
    detailed_error: Optional[str] = Field(None, description="Detailed error with stack trace")
    
    # Execution context
    test_plan_name: str = Field(..., description="Name of the test plan")
    step_number: int = Field(..., description="Step number in the test plan")
    execution_mode: str = Field("visual", description="How the step was executed")
    
    # Visual evidence
    screenshot_before: Optional[bytes] = Field(None, description="Screenshot before action")
    screenshot_after: Optional[bytes] = Field(None, description="Screenshot after action")
    grid_screenshot: Optional[bytes] = Field(None, description="Grid overlay screenshot with highlight")
    
    # AI analysis
    validation_result: Optional[ValidationResult] = Field(None, description="Validation phase results")
    coordinate_result: Optional[CoordinateResult] = Field(None, description="Coordinate determination results")
    execution_result: Optional[ExecutionResult] = Field(None, description="Execution phase results")
    ai_analysis: Optional[AIAnalysis] = Field(None, description="AI evaluation of results")
    
    # Action details
    attempted_action: str = Field(..., description="What action was attempted")
    expected_outcome: str = Field(..., description="What was expected to happen")
    actual_outcome: str = Field(..., description="What actually happened")
    
    # Grid interaction
    grid_cell_targeted: Optional[str] = Field(None, description="Grid cell that was targeted")
    coordinates_used: Optional[GridCoordinate] = Field(None, description="Exact coordinates used")
    
    # Browser state
    url_before: Optional[str] = Field(None, description="URL before action")
    url_after: Optional[str] = Field(None, description="URL after action")
    page_title_before: Optional[str] = Field(None, description="Page title before action")
    page_title_after: Optional[str] = Field(None, description="Page title after action")
    
    # Debugging aids
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores at each phase"
    )
    ui_anomalies: List[str] = Field(
        default_factory=list,
        description="Detected UI anomalies or unexpected states"
    )
    suggested_fixes: List[str] = Field(
        default_factory=list,
        description="AI-suggested fixes or workarounds"
    )
    
    # Categorization
    severity: str = Field("medium", description="Bug severity: low|medium|high|critical")
    is_blocking: bool = Field(False, description="Whether this blocks test continuation")
    is_flaky: bool = Field(False, description="Whether this appears to be a flaky failure")
    
    def to_summary(self) -> str:
        """Generate a concise summary of the bug report."""
        severity_upper = self.severity.upper()
        blocking_text = " [BLOCKING]" if self.is_blocking else ""
        
        summary = f"BUG [{severity_upper}]{blocking_text}"
        
        if self.step_number:
            summary += f" Step {self.step_number}: {self.test_step.description}\n"
        else:
            summary += f" {self.test_step.description}\n"
        
        summary += f"  Error: {self.error_message}\n"
        
        if self.expected_outcome and self.actual_outcome:
            summary += f"  Expected: {self.expected_outcome}\n"
            summary += f"  Actual: {self.actual_outcome}\n"
        
        if self.ai_analysis:
            summary += f"  Confidence: {self.ai_analysis.confidence*100:.0f}%\n"
        elif self.confidence_scores.get('overall'):
            summary += f"  Confidence: {self.confidence_scores.get('overall', 0.0):.0%}\n"
        
        return summary


class EnhancedTestState(TestState):
    """Extended test state with enhanced tracking capabilities."""
    execution_history: List[EnhancedActionResult] = Field(
        default_factory=list,
        description="Detailed history of all action executions"
    )
    bug_reports: List[BugReport] = Field(
        default_factory=list,
        description="Collection of bug reports for failed steps"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics for the test execution"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for test execution"
    )
