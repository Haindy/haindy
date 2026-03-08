"""Action execution collaborator for TestRunner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.core.types import ActionInstruction, ActionType, TestStep
from src.runtime.environment import (
    normalize_automation_backend,
    normalize_runtime_environment_name,
    normalize_target_type,
    resolve_runtime_environment,
)


@dataclass(frozen=True)
class StepActionExecutionRequest:
    action: dict[str, Any]
    step: TestStep
    runtime_environment: str
    current_test_plan_name: str | None
    current_test_case_name: str | None
    current_test_case_id: str | None
    state_context: dict[str, Any]
    recent_actions: list[dict[str, Any]]
    record_driver_actions: bool
    screenshot: bytes | None = None


@dataclass(frozen=True)
class StepActionExecutionResult:
    compatibility_result: dict[str, Any]
    action_data: dict[str, Any]


class TestRunnerExecutor:
    """Executes decomposed step actions and returns structured records."""

    def __init__(
        self,
        *,
        action_agent: Any,
        automation_driver: Any,
        artifacts: Any | None = None,
    ) -> None:
        self._action_agent = action_agent
        self._automation_driver = automation_driver
        self._artifacts = artifacts

    async def execute_action(
        self,
        request: StepActionExecutionRequest,
    ) -> StepActionExecutionResult:
        """Execute a single action and capture the compatibility payload."""
        action = request.action
        step = request.step
        action_type = action.get("type", "assert")
        action_data = {
            "action_id": str(uuid4()),
            "action_type": action_type,
            "target": action.get("target", ""),
            "value": action.get("value"),
            "description": action.get("description", ""),
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "timestamp_end": None,
            "ai_conversation": {
                "test_runner_interpretation": action.copy()
                if isinstance(action, dict)
                else None,
                "action_agent_execution": None,
            },
            "automation_calls": [],
            "result": None,
            "screenshots": {},
        }

        try:
            if not self._automation_driver:
                error_result = {
                    "success": False,
                    "error": "Action agent or automation driver not available",
                }
                action_data["result"] = error_result
                action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
                return StepActionExecutionResult(
                    compatibility_result=error_result,
                    action_data=action_data,
                )

            if action_type == ActionType.RESET_APP:
                reset_result = await self._execute_reset_app(request)
                action_data["result"] = reset_result
                action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
                return StepActionExecutionResult(
                    compatibility_result=reset_result,
                    action_data=action_data,
                )

            if not self._action_agent:
                error_result = {
                    "success": False,
                    "error": "Action agent or automation driver not available",
                }
                action_data["result"] = error_result
                action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
                return StepActionExecutionResult(
                    compatibility_result=error_result,
                    action_data=action_data,
                )

            if hasattr(self._automation_driver, "start_capture"):
                self._automation_driver.start_capture()

            runtime_environment = normalize_runtime_environment_name(
                request.runtime_environment
            )
            runtime_spec = resolve_runtime_environment(
                environment=step.environment,
                default=runtime_environment,
            )
            state_context = (
                request.state_context if isinstance(request.state_context, dict) else {}
            )
            automation_backend = normalize_automation_backend(
                state_context.get("automation_backend"),
                default=runtime_spec.automation_backend,
            )
            target_type = normalize_target_type(
                state_context.get("target_type"),
                default=runtime_spec.target_type,
            )
            timeout_ms = action.get("timeout")
            if timeout_ms is None and step.action_instruction is not None:
                timeout_ms = step.action_instruction.timeout
            if not isinstance(timeout_ms, int):
                timeout_ms = 5000

            instruction = ActionInstruction(
                action_type=ActionType(action_type),
                description=action.get("description", ""),
                target=action.get("target", ""),
                value=action.get("value"),
                expected_outcome=action.get("expected_outcome", step.expected_result),
                computer_use_prompt=action.get("computer_use_prompt"),
                timeout=timeout_ms,
            )
            action_step = TestStep(
                step_id=step.step_id,
                step_number=step.step_number,
                description=action.get("description", step.description),
                action=action.get("description", step.action),
                expected_result=action.get("expected_outcome", step.expected_result),
                action_instruction=instruction,
                dependencies=step.dependencies.copy(),
                optional=not action.get("critical", True),
                intent=step.intent,
                max_retries=step.max_retries,
                cache_label=step.cache_label,
                cache_action=step.cache_action,
                environment=runtime_spec.name,
                can_be_replayed=step.can_be_replayed,
                loop=step.loop,
                scroll_policy=step.scroll_policy,
                capture_clipboard=step.capture_clipboard,
                clipboard_output_key=step.clipboard_output_key,
            )

            test_context = {
                "test_plan_name": request.current_test_plan_name,
                "test_case_name": request.current_test_case_name,
                "test_case_id": request.current_test_case_id,
                "step_number": step.step_number,
                "action_description": action.get("description", ""),
                "recent_actions": request.recent_actions,
                "step_intent": step.intent.value,
                "automation_backend": automation_backend,
                "target_type": target_type,
                "environment": runtime_spec.name,
                "cache_label": (
                    step.cache_label
                    or action.get("target")
                    or action.get("description")
                ),
                "cache_action": step.cache_action or "click",
            }

            screenshot = request.screenshot
            if screenshot is None:
                screenshot = await self._automation_driver.screenshot()
            result = await self._action_agent.execute_action(
                test_step=action_step,
                test_context=test_context,
                screenshot=screenshot,
                record_driver_actions=request.record_driver_actions,
            )

            if hasattr(self._automation_driver, "stop_capture"):
                action_data["automation_calls"] = self._automation_driver.stop_capture()

            if hasattr(self._action_agent, "conversation_history"):
                action_data["ai_conversation"]["action_agent_execution"] = {
                    "messages": self._action_agent.conversation_history.copy(),
                    "screenshot_path": None,
                }
                self._action_agent.conversation_history = []

            action_data["result"] = {
                "success": (result.validation.valid if result.validation else False)
                and (result.execution.success if result.execution else False),
                "validation": result.validation.model_dump()
                if result.validation
                else None,
                "coordinates": result.coordinates.model_dump()
                if result.coordinates
                else None,
                "execution": result.execution.model_dump()
                if result.execution
                else None,
                "ai_analysis": result.ai_analysis.model_dump()
                if result.ai_analysis
                else None,
                "environment_state_before": (
                    {
                        k: v
                        for k, v in result.environment_state_before.model_dump().items()
                        if k != "screenshot"
                    }
                    if result.environment_state_before
                    else None
                ),
                "environment_state_after": (
                    {
                        k: v
                        for k, v in result.environment_state_after.model_dump().items()
                        if k != "screenshot"
                    }
                    if result.environment_state_after
                    else None
                ),
                "cache": {
                    "label": result.cache_label,
                    "action": result.cache_action,
                    "hit": result.cache_hit,
                    "coordinates": result.cache_coordinates,
                    "resolution": result.cache_resolution,
                },
                "driver_actions": result.driver_actions,
            }
            if (
                result.environment_state_before
                and result.environment_state_before.screenshot_path
            ):
                action_data["screenshots"]["before"] = (
                    result.environment_state_before.screenshot_path
                )
            if (
                result.environment_state_after
                and result.environment_state_after.screenshot_path
            ):
                action_data["screenshots"]["after"] = (
                    result.environment_state_after.screenshot_path
                )
            if (
                self._artifacts is not None
                and result.environment_state_after
                and result.environment_state_after.screenshot is not None
                and result.environment_state_after.screenshot_path
            ):
                self._artifacts.update_latest_snapshot(
                    result.environment_state_after.screenshot,
                    result.environment_state_after.screenshot_path,
                    f"action_{step.step_number}_after",
                )

            success = (result.validation.valid if result.validation else False) and (
                result.execution.success if result.execution else False
            )
            outcome = (
                result.ai_analysis.actual_outcome
                if result.ai_analysis
                else "Action completed"
            )
            compatibility_result = {
                "success": success,
                "action_type": action_type,
                "target": action.get("target", ""),
                "outcome": outcome,
                "confidence": result.ai_analysis.confidence
                if result.ai_analysis
                else 0.0,
                "error": result.execution.error_message
                if (result.execution and not success)
                else None,
                "cache_label": result.cache_label,
                "cache_action": result.cache_action,
                "cache_hit": result.cache_hit,
                "cache_coordinates": result.cache_coordinates,
                "cache_resolution": result.cache_resolution,
                "driver_actions": result.driver_actions,
            }
            action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            return StepActionExecutionResult(
                compatibility_result=compatibility_result,
                action_data=action_data,
            )

        except Exception as exc:
            if self._automation_driver and hasattr(
                self._automation_driver, "stop_capture"
            ):
                action_data["automation_calls"] = self._automation_driver.stop_capture()
            error_result = {
                "success": False,
                "action_type": action_type,
                "error": str(exc),
            }
            action_data["result"] = error_result
            action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            return StepActionExecutionResult(
                compatibility_result=error_result,
                action_data=action_data,
            )

    async def _execute_reset_app(
        self,
        request: StepActionExecutionRequest,
    ) -> dict[str, Any]:
        driver = self._automation_driver
        state_context = (
            request.state_context if isinstance(request.state_context, dict) else {}
        )
        entry_setup = state_context.get("entry_setup") or {}
        try:
            if hasattr(driver, "force_stop_app"):
                await driver.force_stop_app()
            if hasattr(driver, "clear_app_data"):
                await driver.clear_app_data()
            if hasattr(driver, "launch_app"):
                package_name = (
                    state_context.get("app_package")
                    or entry_setup.get("app_package")
                    or ""
                )
                activity_name = (
                    state_context.get("app_activity")
                    or entry_setup.get("app_activity")
                    or ""
                )
                if package_name:
                    await driver.launch_app(package_name, activity_name or None)
            await asyncio.sleep(2)
            return {"success": True, "result": "App reset to clean state"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
