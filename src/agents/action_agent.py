"""Computer-use-only ActionAgent implementation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from src.agents.computer_use import ComputerUseExecutionError, ComputerUseSession
from src.config.settings import get_settings
from src.core.enhanced_types import (
    AIAnalysis,
    ComputerToolTurn,
    EnhancedActionResult,
    EnvironmentState,
    ExecutionResult,
    ValidationResult,
)
from src.core.interfaces import AutomationDriver
from src.core.types import ActionInstruction, ActionType, TestCase, TestStatus, TestStep
from src.desktop.cache import CoordinateCache
from src.desktop.execution_replay import DriverActionError, normalize_driver_action
from src.monitoring.debug_logger import get_debug_logger
from src.monitoring.logger import get_logger
from src.runtime.environment import (
    coordinate_cache_path_for_environment,
    resolve_runtime_environment_from_context,
)

OBSERVE_ONLY_ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {"screenshot", "wait", "scroll"}
)

logger = get_logger(__name__)


@dataclass
class ActionAgentStepSession:
    """OpenAI Computer Use session state shared across one test step."""

    session: ComputerUseSession
    provider: str
    environment: str
    base_metadata: dict[str, Any]
    safety_identifier: str
    has_computer_use_action: bool = False
    usable: bool = True
    unusable_reason: str | None = None
    response_ids: list[str] = field(default_factory=list)


@dataclass
class StepSessionValidationResult:
    """Structured validation response returned from the step session."""

    verification: dict[str, Any]
    prompt: str
    raw_response: str
    response_ids: list[str] = field(default_factory=list)


class ActionAgent:
    """Executes test actions through the desktop Computer Use session."""

    def __init__(
        self,
        name: str = "ActionAgent",
        automation_driver: AutomationDriver | None = None,
    ) -> None:
        self.name = name
        self.automation_driver = automation_driver

        settings = get_settings()
        self.settings = settings
        self._coordinate_cache = CoordinateCache(settings.desktop_coordinate_cache_path)
        self._computer_use_model = getattr(settings, "computer_use_model", None)
        self._openai_client: AsyncOpenAI | None = None

        self.conversation_history: list[dict[str, Any]] = []

    def reset_conversation(self) -> None:
        """Reset conversation history for a new action."""
        self.conversation_history = []
        logger.debug("Conversation history reset for new action")

    def _get_openai_client(self) -> AsyncOpenAI:
        """Lazily create an OpenAI client when the CU provider needs one."""
        if self._openai_client is None:
            api_key = str(getattr(self.settings, "openai_api_key", "") or "").strip()
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set HAINDY_OPENAI_API_KEY."
                )
            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                max_retries=int(getattr(self.settings, "openai_max_retries", 3)),
            )
        return self._openai_client

    def supports_step_scoped_validation(self) -> bool:
        """Return True when this agent can reuse one CU session across a step."""
        return str(getattr(self.settings, "cu_provider", "")).strip().lower() in {
            "openai",
            "google",
        }

    async def begin_step_session(
        self,
        test_step: TestStep,
        test_context: dict[str, Any],
    ) -> ActionAgentStepSession | None:
        """Create a reusable Computer Use session for one test step."""
        if not self.supports_step_scoped_validation():
            return None

        debug_logger = get_debug_logger()
        context_lookup = test_context if isinstance(test_context, dict) else {}
        environment = self._resolve_environment(context_lookup)
        safety_identifier = self._resolve_safety_identifier(test_step, context_lookup)
        session = self._new_computer_use_session(
            debug_logger,
            environment=environment,
        )
        session.begin_step_scope()

        return ActionAgentStepSession(
            session=session,
            provider=session.provider,
            environment=environment,
            safety_identifier=safety_identifier,
            base_metadata={
                "step_number": test_step.step_number,
                "test_plan_name": context_lookup.get("test_plan_name")
                or context_lookup.get("plan_name"),
                "test_case_name": context_lookup.get("test_case_name")
                or context_lookup.get("case_name"),
                "step_goal": test_step.description,
                "environment": environment,
                "allow_safety_auto_approve": True,
                "safety_identifier": safety_identifier,
            },
        )

    async def end_step_session(
        self,
        step_session: ActionAgentStepSession | None,
    ) -> None:
        """Close the step-scoped Computer Use session if one is active."""
        if step_session is None:
            return
        await step_session.session.close()

    @staticmethod
    def _mark_step_session_unusable(
        step_session: ActionAgentStepSession | None,
        reason: str,
    ) -> None:
        if step_session is None:
            return
        step_session.usable = False
        step_session.unusable_reason = reason

    def _build_action_session_metadata(
        self,
        step_session: ActionAgentStepSession | None,
        *,
        test_step: TestStep,
        instruction: ActionInstruction,
        interaction_mode: str,
        current_url: str,
        context_lookup: dict[str, Any],
    ) -> tuple[dict[str, Any], str, str]:
        """Build per-action session metadata and return metadata/environment/safety."""
        environment = (
            step_session.environment
            if step_session is not None
            else self._resolve_environment(context_lookup)
        )
        safety_identifier = (
            step_session.safety_identifier
            if step_session is not None
            else self._resolve_safety_identifier(test_step, context_lookup)
        )
        metadata = (
            dict(step_session.base_metadata)
            if step_session is not None
            else {
                "step_number": test_step.step_number,
                "test_plan_name": context_lookup.get("test_plan_name")
                or context_lookup.get("plan_name"),
                "test_case_name": context_lookup.get("test_case_name")
                or context_lookup.get("case_name"),
                "step_goal": test_step.description,
                "environment": environment,
                "allow_safety_auto_approve": True,
                "safety_identifier": safety_identifier,
            }
        )
        metadata.update(
            {
                "target": instruction.target,
                "value": instruction.value,
                "expected_outcome": instruction.expected_outcome,
                "interaction_mode": interaction_mode,
                "response_reporting_scope": self._resolve_response_reporting_scope(
                    instruction
                ),
                "environment": environment,
                "safety_identifier": safety_identifier,
            }
        )
        app_package = str(context_lookup.get("app_package") or "").strip()
        app_activity = str(context_lookup.get("app_activity") or "").strip()
        if app_package:
            metadata["app_package"] = app_package
        if app_activity:
            metadata["app_activity"] = app_activity
        if current_url:
            metadata["current_url"] = current_url
        return metadata, environment, safety_identifier

    @staticmethod
    def _resolve_response_reporting_scope(instruction: ActionInstruction) -> str:
        """Classify how detailed the CU agent's final confirmation may be."""
        if instruction.action_type in {ActionType.ASSERT, ActionType.TYPE}:
            return "detailed"
        return "state_only"

    @staticmethod
    def _format_execution_history(execution_history: list[dict[str, Any]]) -> str:
        history_context: list[str] = []
        for hist_item in execution_history:
            status = hist_item.get("status")
            status_emoji = "✓" if status == TestStatus.PASSED else "✗"
            history_context.append(
                f"Step {hist_item.get('step_number')}: {hist_item.get('action')} - {status_emoji} {status}\n"
                f"  Result: {hist_item.get('actual_result', 'N/A')}"
            )
        return "\n".join(history_context) if history_context else "None"

    @staticmethod
    def _format_action_results(action_results: list[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for idx, action_data in enumerate(action_results, start=1):
            action = action_data.get("action", {})
            action_result = action_data.get("result", {})
            validation = action_result.get("validation", {})
            ai_analysis = action_result.get("ai_analysis", {})
            execution = action_result.get("execution", {})
            cu_outcome = action_result.get("outcome", "")

            action_detail = f"""Action {idx}: {action.get("description", "Unknown action")}
  Type: {action.get("type", "unknown")}
  Target: {action.get("target", "N/A")}
  Success: {action_result.get("success", False)}"""

            if cu_outcome and cu_outcome != "Action completed":
                action_detail += f"\n  CU agent observation: {cu_outcome}"

            action_detail += "\n\n  Validation Results:"
            if isinstance(validation, dict):
                for key, value in validation.items():
                    if key in {
                        "target_reference",
                        "pixel_coordinates",
                        "relative_x",
                        "relative_y",
                    }:
                        continue
                    action_detail += f"\n    {key}: {value}"

            if isinstance(ai_analysis, dict) and ai_analysis:
                action_detail += "\n  \n  AI Analysis:"
                action_detail += (
                    f"\n    Reasoning: {ai_analysis.get('reasoning', 'N/A')}"
                )
                action_detail += (
                    f"\n    Actual outcome: {ai_analysis.get('actual_outcome', 'N/A')}"
                )
                action_detail += (
                    f"\n    Confidence: {ai_analysis.get('confidence', 0.0)}"
                )

            if isinstance(execution, dict) and execution:
                action_detail += "\n  \n  Execution Details:"
                action_detail += (
                    f"\n    Duration: {execution.get('duration_ms', 'N/A')}ms"
                )
                if execution.get("error_message"):
                    action_detail += f"\n    Error: {execution.get('error_message')}"

            rendered.append(action_detail)

        return "\n".join(rendered)

    def _build_step_validation_prompt(
        self,
        *,
        test_case: TestCase,
        step: TestStep,
        action_results: list[dict[str, Any]],
        execution_history: list[dict[str, Any]],
        next_test_case: TestCase | None,
    ) -> str:
        """Build the final in-session step validation prompt."""
        return f"""We have finished executing one test step inside this same Computer Use conversation.

Original test case: "{test_case.name}"

Previous steps in this test case:
{self._format_execution_history(execution_history)}

Current step to validate:
Step {step.step_number}: {step.action}
Expected result: {step.expected_result}

Actions performed in this step:
{self._format_action_results(action_results)}

Use the full context already present in this conversation, including the screenshots and intermediate follow-up frames that were exchanged during these actions.
Do not ask for more screenshots, do not call tools, and do not restart the analysis from scratch.

IMPORTANT:
- The "CU agent observation" fields above are real-time descriptions captured during execution.
- Evaluate ONLY whether this current step achieved its stated expected result.
- Do NOT fail this step based on broader test-case goals or assertions assigned to earlier/later steps unless they are explicitly part of this step's expected result.
- If a CU agent observation includes extra details outside this step's scope, ignore those extras unless they are directly required by this step and supported by the visible evidence already in the conversation.
- Return a final verdict for the step, not for any single sub-action.

Determine:
1. Did this step achieve its intended purpose?
2. If it failed, is the failure a blocker for the next test case?
   Next test case: {next_test_case.name if next_test_case else "None (last test case)"}

Respond with JSON only:
{{
  "verdict": "PASS" or "FAIL",
  "reasoning": "Your analysis of why the step passed or failed",
  "actual_result": "Concise description of what actually happened",
  "confidence": 0.0-1.0,
  "is_blocker": true/false,
  "blocker_reasoning": "Why this would or would not block the next test case"
}}"""

    @staticmethod
    def _normalize_step_validation_result(
        payload: dict[str, Any] | None,
        *,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        """Normalize ActionAgent-produced step validation output."""
        result = dict(payload or {})
        result["verdict"] = str(result.get("verdict", "FAIL")).strip().upper()
        if result["verdict"] not in {"PASS", "FAIL"}:
            result["verdict"] = "FAIL"
        result["reasoning"] = str(
            result.get("reasoning") or failure_reason or "Step validation failed."
        )
        result["actual_result"] = str(
            result.get("actual_result") or failure_reason or "Unknown outcome"
        )
        try:
            result["confidence"] = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            result["confidence"] = 0.0
        result["confidence"] = max(0.0, min(result["confidence"], 1.0))
        result["is_blocker"] = bool(result.get("is_blocker", False))
        result["blocker_reasoning"] = str(result.get("blocker_reasoning") or "")
        return result

    async def validate_step_with_session(
        self,
        *,
        step_session: ActionAgentStepSession,
        test_case: TestCase,
        step: TestStep,
        action_results: list[dict[str, Any]],
        execution_history: list[dict[str, Any]],
        next_test_case: TestCase | None,
    ) -> StepSessionValidationResult:
        """Request the final step verdict from the active session."""
        prompt = self._build_step_validation_prompt(
            test_case=test_case,
            step=step,
            action_results=action_results,
            execution_history=execution_history,
            next_test_case=next_test_case,
        )

        if not step_session.usable:
            verification = self._normalize_step_validation_result(
                None,
                failure_reason=step_session.unusable_reason
                or "Computer Use session became unusable before final validation.",
            )
            return StepSessionValidationResult(
                verification=verification,
                prompt=prompt,
                raw_response="",
                response_ids=[],
            )

        try:
            reflection = await step_session.session.reflect_step(
                prompt=prompt,
                metadata={
                    **step_session.base_metadata,
                    "validation_phase": "step_reflection",
                },
            )
            raw_response = str(reflection.get("raw_text") or "")
            parsed = json.loads(raw_response) if raw_response else {}
            verification = self._normalize_step_validation_result(
                parsed if isinstance(parsed, dict) else None,
                failure_reason="Step reflection returned a non-object JSON payload.",
            )
            response_ids = [
                response_id
                for response_id in reflection.get("response_ids", [])
                if isinstance(response_id, str) and response_id
            ]
            step_session.response_ids.extend(response_ids)
            return StepSessionValidationResult(
                verification=verification,
                prompt=prompt,
                raw_response=raw_response,
                response_ids=response_ids,
            )
        except Exception as exc:
            self._mark_step_session_unusable(step_session, str(exc))
            verification = self._normalize_step_validation_result(
                None,
                failure_reason=f"Step validation failed in the ActionAgent session: {exc}",
            )
            return StepSessionValidationResult(
                verification=verification,
                prompt=prompt,
                raw_response="",
                response_ids=[],
            )

    @staticmethod
    def _slugify_identifier(value: str, max_length: int = 32) -> str:
        """Create a filesystem/API friendly slug."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
        slug = slug.strip("-")
        if max_length and len(slug) > max_length:
            slug = slug[:max_length]
        return slug

    def _resolve_safety_identifier(
        self,
        test_step: TestStep,
        test_context: dict[str, Any],
    ) -> str:
        """Derive a stable safety identifier for Computer Use requests."""
        components: list[str] = []
        debug_logger = get_debug_logger()
        if debug_logger and getattr(debug_logger, "test_run_id", None):
            components.append(
                self._slugify_identifier(debug_logger.test_run_id, max_length=24)
            )

        plan_name = (
            test_context.get("test_plan_name") or test_context.get("plan_name") or ""
        )
        if isinstance(plan_name, str) and plan_name.strip():
            components.append(self._slugify_identifier(plan_name))

        case_name = (
            test_context.get("test_case_name") or test_context.get("case_name") or ""
        )
        if isinstance(case_name, str) and case_name.strip():
            components.append(self._slugify_identifier(case_name))

        components.append(f"step{test_step.step_number}")
        identifier = (
            "-".join(filter(None, components)) or f"haindy-step-{test_step.step_number}"
        )
        return identifier[:64]

    @staticmethod
    def _build_computer_use_goal(
        test_step: TestStep,
        instruction: ActionInstruction,
    ) -> str:
        """Build the prompt passed to Computer Use."""
        if instruction.computer_use_prompt and instruction.computer_use_prompt.strip():
            return str(instruction.computer_use_prompt).strip()

        lines = [
            f"Action type: {instruction.action_type.value}",
            f"Step: {test_step.description}",
        ]
        if instruction.target:
            lines.append(f"Target: {instruction.target}")
        if instruction.value:
            lines.append(f"Value: {instruction.value}")
        lines.append(f"Expected outcome: {instruction.expected_outcome}")
        return "\n".join(lines).strip()

    async def _capture_environment_state(
        self,
        screenshot: bytes | None,
        debug_logger: Any,
        step_number: int | None,
        label: str,
        visual_frame: Any | None = None,
    ) -> EnvironmentState:
        """Capture current page state for debugging."""
        if not self.automation_driver:
            raise ComputerUseExecutionError("Automation driver is not available.")

        url = ""
        title = ""

        get_url = getattr(self.automation_driver, "get_page_url", None)
        get_title = getattr(self.automation_driver, "get_page_title", None)

        if callable(get_url):
            try:
                url = await get_url()
            except Exception:
                logger.debug(
                    "Unable to retrieve page URL during environment state capture",
                    exc_info=True,
                )

        if callable(get_title):
            try:
                title = await get_title()
            except Exception:
                logger.debug(
                    "Unable to retrieve page title during environment state capture",
                    exc_info=True,
                )

        (
            viewport_width,
            viewport_height,
        ) = await self.automation_driver.get_viewport_size()

        screenshot_path = None
        if screenshot and debug_logger:
            screenshot_path = debug_logger.save_screenshot(
                screenshot,
                name=f"computer_use_{label}",
                step_number=step_number,
            )

        return EnvironmentState(
            url=url or "",
            title=title or "",
            viewport_size=(viewport_width, viewport_height),
            screenshot=screenshot,
            screenshot_path=screenshot_path,
            frame_kind=(
                getattr(visual_frame, "kind", "keyframe")
                if visual_frame is not None
                else "keyframe"
            ),
            frame_id=getattr(visual_frame, "frame_id", None),
            parent_keyframe_id=getattr(visual_frame, "parent_keyframe_id", None),
            patch_bounds=(
                getattr(
                    getattr(visual_frame, "bounds", None), "as_tuple", lambda: None
                )()
                if visual_frame is not None
                and getattr(visual_frame, "kind", "keyframe") == "patch"
                else None
            ),
            target_bounds=(
                getattr(
                    getattr(visual_frame, "target_bounds", None),
                    "as_tuple",
                    lambda: None,
                )()
                if visual_frame is not None
                and getattr(visual_frame, "target_bounds", None) is not None
                else None
            ),
            diff_bounds=(
                getattr(
                    getattr(visual_frame, "diff_bounds", None), "as_tuple", lambda: None
                )()
                if visual_frame is not None
                and getattr(visual_frame, "diff_bounds", None) is not None
                else None
            ),
        )

    def _new_computer_use_session(
        self,
        debug_logger: Any,
        environment: str,
    ) -> ComputerUseSession:
        """Create a Computer Use session bound to the current driver/client."""
        if not self.automation_driver:
            raise ComputerUseExecutionError("Automation driver is not available.")

        provider = str(getattr(self.settings, "cu_provider", "")).strip().lower()
        model_override = self._computer_use_model if provider == "openai" else None
        client = self._get_openai_client() if provider == "openai" else None

        cache = self._coordinate_cache
        if hasattr(self.automation_driver, "coordinate_cache"):
            try:
                cache = self.automation_driver.coordinate_cache
            except Exception:
                cache = self._coordinate_cache
        else:
            cache = CoordinateCache(
                coordinate_cache_path_for_environment(self.settings, environment)
            )

        return ComputerUseSession(
            client=client,
            automation_driver=self.automation_driver,
            settings=self.settings,
            debug_logger=debug_logger,
            model=model_override,
            environment=environment,
            coordinate_cache=cache,
        )

    @staticmethod
    def _resolve_environment(test_context: dict[str, Any]) -> str:
        return resolve_runtime_environment_from_context(test_context).name

    @staticmethod
    def _extract_cache_metadata(
        turns: list[ComputerToolTurn],
        cache_label: str | None,
        cache_action: str,
    ) -> tuple[bool, tuple[int, int] | None, tuple[int, int] | None]:
        """Derive cache usage and coordinates from executed turns."""
        if not cache_label:
            return False, None, None

        cache_hit = False
        coordinates: tuple[int, int] | None = None
        resolution: tuple[int, int] | None = None

        for turn in turns:
            meta = getattr(turn, "metadata", {}) or {}
            if meta.get("cache_label") and meta.get("cache_label") != cache_label:
                continue
            if meta.get("cache_action") and meta.get("cache_action") != cache_action:
                continue

            cache_hit = bool(meta.get("cache_hit", False))

            x = meta.get("x") if meta.get("x") is not None else meta.get("start_x")
            y = meta.get("y") if meta.get("y") is not None else meta.get("start_y")
            if x is not None and y is not None:
                try:
                    coordinates = (int(x), int(y))
                except Exception:
                    coordinates = None

            res_raw = meta.get("resolution")
            if isinstance(res_raw, (list, tuple)) and len(res_raw) == 2:
                try:
                    resolution = (int(res_raw[0]), int(res_raw[1]))
                except Exception:
                    resolution = None

        return cache_hit, coordinates, resolution

    @staticmethod
    def _scroll_xy(direction: str, magnitude: int) -> tuple[int, int]:
        direction_norm = str(direction or "").strip().lower()
        amount = abs(int(magnitude))
        if direction_norm == "down":
            return (0, amount)
        if direction_norm == "up":
            return (0, -amount)
        if direction_norm == "right":
            return (amount, 0)
        if direction_norm == "left":
            return (-amount, 0)
        raise ValueError(f"Invalid scroll direction: {direction!r}")

    @staticmethod
    def _canonicalize_replay_keys(raw_keys: Any) -> str | None:
        """Normalize replay key payloads into a single driver-compatible string."""
        if raw_keys is None:
            return None
        if isinstance(raw_keys, (list, tuple)):
            pieces = [str(part).strip() for part in raw_keys if str(part).strip()]
            if not pieces:
                return None
            return "+".join(pieces)
        text = str(raw_keys).strip()
        return text or None

    @classmethod
    def _extract_driver_actions(
        cls, turns: list[ComputerToolTurn]
    ) -> list[dict[str, Any]]:
        """Translate Computer Use turns into replayable driver actions."""
        recorded: list[dict[str, Any]] = []
        for turn in turns:
            if getattr(turn, "status", None) != "executed":
                continue

            action_type = str(getattr(turn, "action_type", "") or "").strip().lower()
            params = getattr(turn, "parameters", {}) or {}
            meta = getattr(turn, "metadata", {}) or {}
            normalized_coords = bool(meta.get("normalized_coords", False))

            if action_type in {"click", "click_at", "move_mouse_and_click"}:
                x = meta.get("x")
                y = meta.get("y")
                if x is None or y is None:
                    continue
                recorded.append(
                    {
                        "type": "click",
                        "x": x,
                        "y": y,
                        "button": (params.get("button", "left") or "left"),
                        "click_count": params.get("click_count", 1),
                    }
                )
            elif action_type == "double_click":
                x = meta.get("x")
                y = meta.get("y")
                if x is None or y is None:
                    continue
                recorded.append(
                    {
                        "type": "click",
                        "x": x,
                        "y": y,
                        "button": "left",
                        "click_count": 2,
                    }
                )
            elif action_type == "right_click":
                x = meta.get("x")
                y = meta.get("y")
                if x is None or y is None:
                    continue
                recorded.append(
                    {
                        "type": "click",
                        "x": x,
                        "y": y,
                        "button": "right",
                        "click_count": 1,
                    }
                )
            elif action_type in {"move", "hover_at"}:
                x = meta.get("x")
                y = meta.get("y")
                if x is None or y is None:
                    continue
                recorded.append({"type": "move", "x": x, "y": y})
            elif action_type in {"drag", "drag_and_drop"}:
                start_x = meta.get("start_x")
                start_y = meta.get("start_y")
                end_x = meta.get("end_x")
                end_y = meta.get("end_y")
                if None in {start_x, start_y, end_x, end_y}:
                    continue
                recorded.append(
                    {
                        "type": "drag",
                        "start_x": start_x,
                        "start_y": start_y,
                        "end_x": end_x,
                        "end_y": end_y,
                    }
                )
            elif action_type == "scroll":
                scroll_x = meta.get("scroll_x")
                scroll_y = meta.get("scroll_y")
                if scroll_x is None and scroll_y is None:
                    direction = (
                        meta.get("scroll_direction") or params.get("direction") or ""
                    )
                    magnitude = meta.get("scroll_magnitude") or params.get("magnitude")
                    if magnitude is None:
                        continue
                    try:
                        dx, dy = cls._scroll_xy(str(direction), int(magnitude))
                    except Exception:
                        continue
                    recorded.append({"type": "scroll_by_pixels", "x": dx, "y": dy})
                else:
                    recorded.append(
                        {
                            "type": "scroll_by_pixels",
                            "x": scroll_x or 0,
                            "y": scroll_y or 0,
                        }
                    )
            elif action_type == "scroll_document":
                direction = (
                    meta.get("scroll_direction") or params.get("direction") or ""
                )
                magnitude = meta.get("scroll_magnitude") or params.get("magnitude")
                if magnitude is None:
                    continue
                try:
                    dx, dy = cls._scroll_xy(str(direction), int(magnitude))
                except Exception:
                    continue
                recorded.append({"type": "scroll_by_pixels", "x": dx, "y": dy})
            elif action_type == "scroll_at":
                x = meta.get("x")
                y = meta.get("y")
                if x is not None and y is not None:
                    recorded.append({"type": "move", "x": x, "y": y})
                direction = (
                    meta.get("scroll_direction") or params.get("direction") or ""
                )
                magnitude = meta.get("scroll_magnitude") or params.get("magnitude")
                if magnitude is None:
                    continue
                try:
                    dx, dy = cls._scroll_xy(str(direction), int(magnitude))
                except Exception:
                    continue
                recorded.append({"type": "scroll_by_pixels", "x": dx, "y": dy})
            elif action_type == "type":
                text_payload = (
                    params.get("text") or params.get("value") or params.get("input")
                )
                if text_payload is None:
                    continue
                recorded.append({"type": "type_text", "text": str(text_payload)})
            elif action_type == "type_text_at":
                x = meta.get("x")
                y = meta.get("y")
                if x is None or y is None:
                    continue
                text_payload = params.get("text")
                if text_payload is None:
                    continue
                press_enter_default = False if normalized_coords else True
                press_enter = bool(params.get("press_enter", press_enter_default))
                clear_before = bool(params.get("clear_before_typing", True))
                recorded.append(
                    {
                        "type": "click",
                        "x": x,
                        "y": y,
                        "button": "left",
                        "click_count": 1,
                    }
                )
                if clear_before:
                    recorded.append({"type": "press_key", "keys": "ctrl+a"})
                    recorded.append({"type": "press_key", "keys": "backspace"})
                recorded.append({"type": "type_text", "text": str(text_payload)})
                if press_enter:
                    recorded.append({"type": "press_key", "keys": "enter"})
            elif action_type in {"keypress", "key_combination"}:
                keys = params.get("key") or params.get("value") or params.get("keys")
                if keys is None:
                    continue
                normalized_keys = cls._canonicalize_replay_keys(keys)
                if not normalized_keys:
                    continue
                recorded.append({"type": "press_key", "keys": normalized_keys})
            elif action_type == "wait":
                duration_ms = meta.get("duration_ms") or params.get("duration_ms")
                if duration_ms is None:
                    continue
                recorded.append({"type": "wait", "duration_ms": duration_ms})
            elif action_type == "wait_5_seconds":
                recorded.append({"type": "wait", "duration_ms": 5000})
            elif action_type == "go_back":
                recorded.append({"type": "press_key", "keys": "alt+left"})
            elif action_type == "go_forward":
                recorded.append({"type": "press_key", "keys": "alt+right"})
            elif action_type == "navigate":
                url = params.get("url")
                if not url:
                    continue
                recorded.append({"type": "press_key", "keys": "ctrl+l"})
                recorded.append({"type": "type_text", "text": str(url)})
                recorded.append({"type": "press_key", "keys": "enter"})
            elif action_type == "search":
                recorded.append({"type": "press_key", "keys": "ctrl+l"})
                recorded.append(
                    {"type": "type_text", "text": "https://www.google.com/"}
                )
                recorded.append({"type": "press_key", "keys": "enter"})

        normalized: list[dict[str, Any]] = []
        for action in recorded:
            try:
                normalized.append(normalize_driver_action(action))
            except DriverActionError:
                logger.debug(
                    "ActionAgent: skipping invalid driver action",
                    exc_info=True,
                    extra={"action": action},
                )
        return normalized

    async def _execute_skip_navigation_workflow(
        self,
        test_step: TestStep,
        test_context: dict[str, Any],
    ) -> EnhancedActionResult:
        """Return a successful result when navigation is already satisfied."""
        instruction = test_step.action_instruction
        rationale = instruction.description if instruction else test_step.description
        rationale = rationale or "Navigation already satisfied."

        validation = ValidationResult(
            valid=True,
            confidence=0.9,
            reasoning=f"Navigation skipped: {rationale}",
        )
        execution = ExecutionResult(
            success=True,
            execution_time_ms=0.0,
            error_message=None,
        )
        ai_analysis = AIAnalysis(
            success=True,
            confidence=0.7,
            actual_outcome=rationale,
            matches_expected=True,
        )

        result = EnhancedActionResult(
            test_step_id=test_step.step_id,
            test_step=test_step,
            test_context=test_context,
            validation=validation,
            execution=execution,
            ai_analysis=ai_analysis,
            overall_success=True,
            failure_phase=None,
        )
        result.timestamp_end = datetime.now(timezone.utc)

        debug_logger = get_debug_logger()
        if debug_logger:
            ctx = test_context if isinstance(test_context, dict) else {}
            debug_logger.log_ai_interaction(
                agent_name=self.name,
                action_type="skip_navigation",
                prompt=rationale,
                response="Navigation already satisfied; no action taken.",
                screenshot_path=None,
                additional_context={
                    "test_case": ctx.get("test_case_id"),
                    "step_number": test_step.step_number,
                },
            )

        return result

    async def _execute_computer_tool_workflow(
        self,
        test_step: TestStep,
        test_context: dict[str, Any],
        screenshot: bytes | None = None,
        record_driver_actions: bool = False,
        step_session: ActionAgentStepSession | None = None,
    ) -> EnhancedActionResult:
        """Execute an action using the Computer Use tool and return a rich result."""
        if not self.automation_driver:
            raise ComputerUseExecutionError("Automation driver is not available.")

        instruction = test_step.action_instruction
        if not instruction:
            raise ComputerUseExecutionError(
                "Missing action instruction for Computer Use workflow."
            )

        debug_logger = get_debug_logger()
        initial_screenshot = screenshot
        if initial_screenshot is None:
            initial_screenshot = await self.automation_driver.screenshot()

        environment_state_before = await self._capture_environment_state(
            initial_screenshot,
            debug_logger,
            test_step.step_number,
            "before",
        )

        is_assert_step = instruction.action_type == ActionType.ASSERT
        interaction_mode = "observe_only" if is_assert_step else "execute"
        goal = self._build_computer_use_goal(test_step, instruction)
        if not goal:
            raise ComputerUseExecutionError("Missing Computer Use goal.")

        context_lookup = test_context if isinstance(test_context, dict) else {}
        context_for_result = (
            dict(test_context) if isinstance(test_context, dict) else {}
        )
        session_metadata, environment, safety_identifier = (
            self._build_action_session_metadata(
                step_session,
                test_step=test_step,
                instruction=instruction,
                interaction_mode=interaction_mode,
                current_url=environment_state_before.url,
                context_lookup=context_lookup,
            )
        )

        context_for_result["safety_identifier"] = safety_identifier
        context_for_result["interaction_mode"] = interaction_mode

        session = (
            step_session.session
            if step_session is not None
            else self._new_computer_use_session(
                debug_logger,
                environment=environment,
            )
        )

        self.conversation_history.append({"role": "user", "content": goal})

        cache_label: str | None = None
        cache_action = "click"
        if instruction.action_type in {ActionType.CLICK, ActionType.TYPE}:
            cache_label = (
                instruction.target or instruction.description or test_step.description
            )
        if isinstance(test_context, dict):
            cache_label = test_context.get("cache_label") or cache_label
            cache_action = test_context.get("cache_action") or cache_action

        allowed_actions: set[str] | None = None
        if is_assert_step:
            allowed_actions = set(OBSERVE_ONLY_ALLOWED_ACTIONS)

        start_ts = time.perf_counter()
        try:
            if step_session is not None and step_session.provider in {
                "openai",
                "google",
            }:
                session_result = await session.execute_step_action(
                    goal,
                    initial_screenshot,
                    session_metadata,
                    allowed_actions=allowed_actions,
                    environment=environment,
                    cache_label=cache_label,
                    cache_action=cache_action,
                )
            else:
                session_result = await session.run(
                    goal,
                    initial_screenshot,
                    session_metadata,
                    allowed_actions=allowed_actions,
                    environment=environment,
                    cache_label=cache_label,
                    cache_action=cache_action,
                )
        except Exception as exc:
            self._mark_step_session_unusable(step_session, str(exc))
            raise ComputerUseExecutionError(str(exc)) from exc

        duration_ms = (time.perf_counter() - start_ts) * 1000

        after_screenshot = (
            session_result.final_visual_frame.image_bytes
            if session_result.final_visual_frame is not None
            else None
        )
        if after_screenshot is None:
            after_screenshot = await self.automation_driver.screenshot()
        environment_state_after = await self._capture_environment_state(
            after_screenshot,
            debug_logger,
            test_step.step_number,
            "after",
            visual_frame=session_result.final_visual_frame,
        )

        failing_action = next(
            (
                action
                for action in session_result.actions
                if action.status != "executed"
            ),
            None,
        )

        execution_error: str | None = None
        if session_result.terminal_status == "failed":
            execution_error = (
                session_result.terminal_failure_reason
                or "Computer Use session terminated with a failure state."
            )
        elif failing_action and failing_action.error_message:
            execution_error = failing_action.error_message
        elif failing_action:
            execution_error = f"Computer action '{failing_action.action_type}' did not complete successfully."
        elif session_result.safety_events:
            execution_error = (
                session_result.safety_events[0].message
                or "Safety check prevented action execution."
            )

        success = execution_error is None
        if step_session is not None:
            step_session.has_computer_use_action = True
            step_session.response_ids.extend(
                response_id
                for response_id in session_result.response_ids
                if response_id
            )
            if session_result.terminal_status == "failed":
                self._mark_step_session_unusable(
                    step_session,
                    execution_error
                    or session_result.terminal_failure_reason
                    or "Computer Use session terminated with a failure state.",
                )

        execution_result = ExecutionResult(
            success=success,
            execution_time_ms=duration_ms,
            error_message=execution_error,
        )

        concerns: list[str] = []
        validation_reason = "Computer Use tool executed the requested action."
        if execution_error:
            concerns.append(execution_error)
            validation_reason = execution_error
        if session_result.safety_events:
            safety_message = session_result.safety_events[0].message
            if safety_message and safety_message not in concerns:
                concerns.append(safety_message)
            validation_reason = safety_message or validation_reason

        validation = ValidationResult(
            valid=success and not session_result.safety_events,
            confidence=0.75 if success else 0.25,
            reasoning=validation_reason,
            concerns=concerns,
        )

        ai_analysis: AIAnalysis | None = None
        if session_result.final_output:
            ai_analysis = AIAnalysis(
                success=success,
                confidence=0.6 if success else 0.4,
                actual_outcome=session_result.final_output,
                matches_expected=success,
            )
            self.conversation_history.append(
                {"role": "assistant", "content": session_result.final_output}
            )

        failed_action_count = sum(
            1 for action in session_result.actions if action.status != "executed"
        )
        if debug_logger:
            debug_logger.log_ai_interaction(
                agent_name=self.name,
                action_type="computer_use",
                prompt=goal,
                response=session_result.final_output or "",
                screenshot_path=environment_state_after.screenshot_path,
                additional_context={
                    "test_case": context_lookup.get("test_case_id"),
                    "step_number": test_step.step_number,
                    "response_ids": session_result.response_ids,
                    "terminal_status": session_result.terminal_status,
                    "terminal_failure_code": session_result.terminal_failure_code,
                    "overall_success": success,
                    "failed_action_count": failed_action_count,
                },
            )

        cache_hit, cache_coordinates, cache_resolution = self._extract_cache_metadata(
            session_result.actions,
            cache_label,
            cache_action,
        )
        driver_actions = (
            self._extract_driver_actions(session_result.actions)
            if record_driver_actions
            else []
        )

        result = EnhancedActionResult(
            test_step_id=test_step.step_id,
            test_step=test_step,
            test_context=context_for_result,
            validation=validation,
            execution=execution_result,
            environment_state_before=environment_state_before,
            environment_state_after=environment_state_after,
            ai_analysis=ai_analysis,
            overall_success=success,
            failure_phase=None if success else "execution",
            computer_actions=session_result.actions,
            safety_events=session_result.safety_events,
            final_model_output=session_result.final_output,
            response_ids=session_result.response_ids,
            cache_label=cache_label,
            cache_action=cache_action if cache_label else None,
            cache_hit=cache_hit,
            cache_coordinates=cache_coordinates,
            cache_resolution=cache_resolution,
            driver_actions=driver_actions,
        )
        result.timestamp_end = datetime.now(timezone.utc)
        return result

    async def execute_action(
        self,
        test_step: TestStep,
        test_context: dict[str, Any],
        screenshot: bytes | None = None,
        record_driver_actions: bool = False,
        step_session: ActionAgentStepSession | None = None,
    ) -> EnhancedActionResult:
        """Execute an action through Computer Use (except explicit skip-navigation)."""
        self.reset_conversation()

        instruction = test_step.action_instruction
        action_type = instruction.action_type if instruction else None

        is_skip_navigation = action_type == ActionType.SKIP_NAVIGATION
        if not is_skip_navigation:
            raw_action = str(test_step.action or "").strip().lower()
            is_skip_navigation = raw_action == ActionType.SKIP_NAVIGATION.value

        if is_skip_navigation:
            return await self._execute_skip_navigation_workflow(test_step, test_context)

        try:
            return await self._execute_computer_tool_workflow(
                test_step=test_step,
                test_context=test_context,
                screenshot=screenshot,
                record_driver_actions=record_driver_actions,
                step_session=step_session,
            )
        except ComputerUseExecutionError:
            logger.error(
                "Computer Use workflow failed; aborting action",
                extra={
                    "step_number": test_step.step_number,
                    "action_type": action_type.value if action_type else "unknown",
                },
                exc_info=True,
            )
            raise
