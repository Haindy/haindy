"""Replay and coordinate cache helpers for TestRunner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.core.types import StepResult, TestCase, TestStatus, TestStep
from src.desktop.cache import CoordinateCache
from src.desktop.execution_replay import replay_driver_actions
from src.monitoring.logger import get_logger
from src.runtime.environment import (
    coordinate_cache_path_for_environment,
    normalize_runtime_environment_name,
)
from src.runtime.execution_replay_cache import (
    ExecutionReplayCache,
    ExecutionReplayCacheKey,
)

REPLAY_CACHED_ACTION_MIN_STABILIZATION_WAIT_MS = 2000
REPLAY_VALIDATION_ONLY_ACTION_TYPES: frozenset[str] = frozenset(
    {"assert", "skip_navigation", "wait", "screenshot"}
)
REPLAY_VALIDATION_MODEL_WAIT_BUDGET_MS = 30000
REPLAY_VALIDATION_MODEL_WAIT_FALLBACK_MS = 1000

CaptureReplayScreenshot = Callable[[str], Awaitable[tuple[bytes | None, str | None]]]
VerifyReplayOutcome = Callable[..., Awaitable[dict[str, Any]]]
CoerceModelBool = Callable[[Any], bool]

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReplayExecutionRequest:
    """Inputs required to attempt an execution replay hit."""

    step: TestStep
    test_case: TestCase
    step_result: StepResult
    screenshot_before: bytes | None
    execution_history: list[dict[str, Any]]
    next_test_case: TestCase | None
    current_test_plan_name: str
    current_runtime_environment: str
    plan_cache_key: str
    plan_fingerprint: str
    capture_replay_screenshot: CaptureReplayScreenshot
    verify_expected_outcome: VerifyReplayOutcome
    coerce_model_bool: CoerceModelBool
    trace: Any = None


@dataclass(frozen=True)
class ReplayExecutionResult:
    """Replay validation result returned to TestRunner."""

    action_record: dict[str, Any]
    actions_performed: list[dict[str, Any]]
    verification: dict[str, Any]
    replay_validation_wait_spent_ms: int
    replay_validation_wait_cycles: int
    replay_validation_wait_budget_remaining_ms: int
    fallback_to_cu: bool = False
    fallback_screenshot_bytes: bytes | None = None
    fallback_screenshot_path: str | None = None


@dataclass
class ExecutionReplayService:
    """Encapsulates replay cache and coordinate cache internals."""

    settings: Any
    automation_driver: Any
    execution_replay_cache: ExecutionReplayCache
    coordinate_cache: CoordinateCache

    def set_automation_driver(self, automation_driver: Any) -> None:
        self.automation_driver = automation_driver

    def set_coordinate_cache(self, coordinate_cache: CoordinateCache) -> None:
        self.coordinate_cache = coordinate_cache

    def coordinate_cache_for_environment(self, environment: str) -> CoordinateCache:
        if self.coordinate_cache and getattr(
            self.coordinate_cache, "cache_path", None
        ) == coordinate_cache_path_for_environment(self.settings, environment):
            return self.coordinate_cache
        return CoordinateCache(
            coordinate_cache_path_for_environment(self.settings, environment)
        )

    @staticmethod
    def is_validation_only_action_result(result: dict[str, Any]) -> bool:
        action_payload = result.get("action")
        if not isinstance(action_payload, dict):
            return False
        action_type = str(action_payload.get("type") or "").strip().lower()
        if not action_type:
            return False
        return action_type in REPLAY_VALIDATION_ONLY_ACTION_TYPES

    @classmethod
    def is_validation_only_step_result_set(
        cls,
        action_results: list[dict[str, Any]],
    ) -> bool:
        seen_action_type = False
        for result in action_results:
            action_payload = result.get("action")
            if not isinstance(action_payload, dict):
                continue
            action_type = str(action_payload.get("type") or "").strip().lower()
            if not action_type:
                continue
            seen_action_type = True
            if not cls.is_validation_only_action_result(result):
                return False
        return seen_action_type

    @staticmethod
    def driver_actions_for_replay(result: dict[str, Any]) -> list[dict[str, Any]]:
        driver_actions = result.get("driver_actions")
        if not isinstance(driver_actions, list):
            nested = result.get("result")
            if isinstance(nested, dict):
                nested_driver_actions = nested.get("driver_actions")
                if isinstance(nested_driver_actions, list):
                    driver_actions = nested_driver_actions
        if not isinstance(driver_actions, list):
            return []
        return [item for item in driver_actions if isinstance(item, dict)]

    def replay_enabled(self, step: TestStep) -> bool:
        if not self.settings.enable_execution_replay_cache:
            return False
        if getattr(step, "loop", False):
            return False
        return self.automation_driver is not None

    def replay_stabilization_wait_ms(self) -> int:
        configured = int(
            getattr(self.settings, "actions_computer_tool_stabilization_wait_ms", 0)
        )
        return max(configured, REPLAY_CACHED_ACTION_MIN_STABILIZATION_WAIT_MS)

    async def execution_replay_key(
        self,
        *,
        step: TestStep,
        test_case: TestCase,
        current_test_plan_name: str,
        current_runtime_environment: str,
        plan_cache_key: str,
        plan_fingerprint: str,
    ) -> ExecutionReplayCacheKey | None:
        if not self.automation_driver:
            return None
        try:
            (
                viewport_width,
                viewport_height,
            ) = await self.automation_driver.get_viewport_size()
        except Exception:
            return None
        environment_name = normalize_runtime_environment_name(
            step.environment,
            default=normalize_runtime_environment_name(current_runtime_environment),
        )
        keyboard_layout = getattr(self.automation_driver, "keyboard_layout", None) or (
            "android"
            if environment_name == "mobile_adb"
            else self.settings.desktop_keyboard_layout
        )
        return ExecutionReplayCacheKey(
            scenario=current_test_plan_name,
            step=plan_cache_key,
            environment=environment_name,
            resolution=(int(viewport_width), int(viewport_height)),
            keyboard_layout=str(keyboard_layout),
            plan_fingerprint=plan_fingerprint,
        )

    def lookup(
        self,
        key: ExecutionReplayCacheKey,
    ) -> Any:
        return self.execution_replay_cache.lookup(key)

    def invalidate(self, key: ExecutionReplayCacheKey) -> None:
        self.execution_replay_cache.invalidate(key)

    def store(
        self,
        key: ExecutionReplayCacheKey,
        actions: list[dict[str, Any]],
    ) -> None:
        self.execution_replay_cache.store(key, actions)

    async def try_execution_replay(
        self,
        request: ReplayExecutionRequest,
    ) -> ReplayExecutionResult | None:
        """Attempt a replay cache hit, validate it, and fall back on failure."""
        if not self.replay_enabled(request.step):
            return None

        key = await self.execution_replay_key(
            step=request.step,
            test_case=request.test_case,
            current_test_plan_name=request.current_test_plan_name,
            current_runtime_environment=request.current_runtime_environment,
            plan_cache_key=request.plan_cache_key,
            plan_fingerprint=request.plan_fingerprint,
        )
        if key is None:
            return None

        entry = self.lookup(key)
        if not entry:
            return None

        driver = self.automation_driver
        if driver is None:
            return None

        if request.trace:
            request.trace.record_cache_event(
                {
                    "type": "execution_replay_cache_hit",
                    "scenario": key.scenario,
                    "step": key.step,
                    "environment": key.environment,
                    "resolution": key.resolution,
                    "keyboard_layout": key.keyboard_layout,
                    "action_count": len(entry.actions),
                }
            )

        try:
            await replay_driver_actions(
                driver,
                entry.actions,
                stabilization_wait_ms=self.replay_stabilization_wait_ms(),
                action_timeout_seconds=max(
                    self.settings.actions_computer_tool_action_timeout_ms / 1000.0,
                    0.5,
                ),
            )
        except Exception as exc:
            logger.warning(
                "TestRunner: execution replay failed; invalidating cache",
                extra={"step": key.step, "error": str(exc)},
            )
            self._invalidate_replay_entry(
                key=key,
                trace=request.trace,
                reason="action_error",
                error=str(exc),
            )
            self._record_replay_fallback(key, trace=request.trace)
            return None

        replay_result = {
            "success": True,
            "outcome": f"Replayed {len(entry.actions)} cached driver action(s).",
            "driver_actions": entry.actions,
        }
        replay_action_record: dict[str, Any] = {
            "action_id": f"execution_replay_{uuid4()}",
            "action_type": "execution_replay",
            "target": "",
            "value": None,
            "description": f"Replay cached actions for {key.step}",
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
            "ai_conversation": {
                "test_runner_interpretation": None,
                "action_agent_execution": None,
            },
            "automation_calls": [],
            "result": replay_result,
            "screenshots": {
                "before": request.step_result.screenshot_before,
            },
        }

        screenshot_after = None
        screenshot_path = None
        screenshot_after, screenshot_path = await request.capture_replay_screenshot(
            "after"
        )
        if screenshot_path:
            request.step_result.screenshot_after = screenshot_path
            replay_action_record["screenshots"]["after"] = screenshot_path

        replay_action = {
            "action": {
                "type": "execution_replay",
                "description": replay_action_record["description"],
            },
            "result": replay_result,
            "full_data": {
                "action_type": "execution_replay",
                "result": replay_result,
            },
        }
        replay_actions_performed = [
            {
                "success": True,
                "action_type": "execution_replay",
                "outcome": replay_result["outcome"],
                "confidence": 1.0,
                "driver_actions": entry.actions,
            }
        ]

        replay_wait_spent_ms = 0
        replay_wait_cycles = 0
        replay_wait_budget_ms = REPLAY_VALIDATION_MODEL_WAIT_BUDGET_MS

        while True:
            verification = await request.verify_expected_outcome(
                test_case=request.test_case,
                step=request.step,
                action_results=[replay_action],
                screenshot_before=request.screenshot_before,
                screenshot_after=screenshot_after,
                execution_history=request.execution_history,
                next_test_case=request.next_test_case,
                replay_wait_budget_ms=replay_wait_budget_ms,
            )
            if verification.get("verdict") == "PASS":
                break

            request_additional_wait = request.coerce_model_bool(
                verification.get("request_additional_wait", False)
            )
            if not request_additional_wait or replay_wait_budget_ms <= 0:
                break

            recommended_wait_raw = verification.get("recommended_wait_ms", 0)
            try:
                recommended_wait_ms = int(recommended_wait_raw)
            except (TypeError, ValueError):
                recommended_wait_ms = 0

            wait_ms = recommended_wait_ms
            if wait_ms <= 0:
                wait_ms = min(
                    REPLAY_VALIDATION_MODEL_WAIT_FALLBACK_MS,
                    replay_wait_budget_ms,
                )
            wait_ms = max(0, min(wait_ms, replay_wait_budget_ms))
            if wait_ms <= 0:
                break

            logger.info(
                "Execution replay validation requested additional wait",
                extra={
                    "step": key.step,
                    "wait_ms": wait_ms,
                    "remaining_budget_before_wait_ms": replay_wait_budget_ms,
                    "wait_reasoning": verification.get("wait_reasoning", ""),
                },
            )
            await asyncio.sleep(wait_ms / 1000.0)
            replay_wait_budget_ms -= wait_ms
            replay_wait_spent_ms += wait_ms
            replay_wait_cycles += 1

            screenshot_after, screenshot_path = await request.capture_replay_screenshot(
                f"replay_wait_{replay_wait_cycles}_after"
            )
            if screenshot_path:
                request.step_result.screenshot_after = screenshot_path

        request.step_result.actions_performed = replay_actions_performed
        request.step_result.status = (
            TestStatus.PASSED
            if verification.get("verdict") == "PASS"
            else TestStatus.FAILED
        )
        request.step_result.actual_result = str(
            verification.get("actual_result") or "Unknown outcome"
        )
        request.step_result.error_message = (
            str(verification.get("reasoning") or "")
            if verification.get("verdict") == "FAIL"
            else None
        )
        request.step_result.confidence = float(verification.get("confidence", 0.0))

        if verification.get("verdict") == "FAIL":
            logger.info(
                "Execution replay validation failed; invalidating cache",
                extra={
                    "step": key.step,
                    "validation_reasoning": verification.get("reasoning"),
                    "replay_wait_spent_ms": replay_wait_spent_ms,
                    "replay_wait_cycles": replay_wait_cycles,
                },
            )
            fallback_screenshot_bytes = None
            fallback_screenshot_path = None
            (
                fallback_screenshot_bytes,
                fallback_screenshot_path,
            ) = await request.capture_replay_screenshot("replay_failure")
            if fallback_screenshot_path:
                request.step_result.screenshot_after = fallback_screenshot_path
                replay_action_record["screenshots"]["after"] = fallback_screenshot_path
            self._invalidate_replay_entry(
                key=key,
                trace=request.trace,
                reason="validation_failed",
                message=str(verification.get("reasoning") or ""),
                replay_wait_spent_ms=replay_wait_spent_ms,
                replay_wait_cycles=replay_wait_cycles,
            )
            self._record_replay_fallback(key, trace=request.trace)
            return ReplayExecutionResult(
                action_record=replay_action_record,
                actions_performed=replay_actions_performed,
                verification=verification,
                replay_validation_wait_spent_ms=replay_wait_spent_ms,
                replay_validation_wait_cycles=replay_wait_cycles,
                replay_validation_wait_budget_remaining_ms=replay_wait_budget_ms,
                fallback_to_cu=True,
                fallback_screenshot_bytes=fallback_screenshot_bytes,
                fallback_screenshot_path=fallback_screenshot_path,
            )

        if request.trace:
            request.trace.record_step(
                scenario_name=key.scenario,
                step=request.step,
                step_result=request.step_result,
                attempt=1,
                plan_cache_hit=None,
            )

        return ReplayExecutionResult(
            action_record=replay_action_record,
            actions_performed=replay_actions_performed,
            verification=verification,
            replay_validation_wait_spent_ms=replay_wait_spent_ms,
            replay_validation_wait_cycles=replay_wait_cycles,
            replay_validation_wait_budget_remaining_ms=replay_wait_budget_ms,
        )

    async def store_execution_replay(
        self,
        *,
        step: TestStep,
        test_case: TestCase,
        action_results: list[dict[str, Any]],
        current_test_plan_name: str,
        current_runtime_environment: str,
        plan_cache_key: str,
        plan_fingerprint: str,
    ) -> ExecutionReplayCacheKey | None:
        if not self.replay_enabled(step) or not action_results:
            return None
        if self.is_validation_only_step_result_set(action_results):
            return None
        recorded: list[dict[str, Any]] = []
        for result in action_results:
            recorded.extend(self.driver_actions_for_replay(result))
        if not recorded:
            return None
        key = await self.execution_replay_key(
            step=step,
            test_case=test_case,
            current_test_plan_name=current_test_plan_name,
            current_runtime_environment=current_runtime_environment,
            plan_cache_key=plan_cache_key,
            plan_fingerprint=plan_fingerprint,
        )
        if key is None:
            return None
        try:
            self.store(key, recorded)
        except Exception:
            logger.debug(
                "TestRunner: failed to store execution replay cache", exc_info=True
            )
            return None
        return key

    async def persist_coordinate_cache(
        self,
        *,
        action_results: list[dict[str, Any]],
        plan_cache_key: str | None,
        current_test_plan_name: str,
        trace: Any,
    ) -> None:
        if not self.coordinate_cache or not action_results:
            return
        for result in action_results:
            label = result.get("cache_label")
            coords = result.get("cache_coordinates")
            action = result.get("cache_action") or "click"
            if not label or not coords:
                continue
            resolution = result.get("cache_resolution")
            if not resolution:
                if not self.automation_driver:
                    continue
                try:
                    resolution = await self.automation_driver.get_viewport_size()
                except Exception:
                    continue
            x, y = coords
            self.coordinate_cache.add(label, action, x, y, resolution)
            if trace:
                trace.record_cache_event(
                    {
                        "type": "coordinate_cache_add",
                        "scenario": current_test_plan_name,
                        "step": plan_cache_key,
                        "cache_label": label,
                        "cache_action": action,
                        "x": x,
                        "y": y,
                        "resolution": resolution,
                    }
                )

    async def invalidate_coordinate_cache(
        self,
        *,
        action_results: list[dict[str, Any]],
        plan_cache_key: str | None,
        current_test_plan_name: str,
        trace: Any,
    ) -> None:
        if not self.coordinate_cache or not action_results:
            return
        hits = [
            result
            for result in action_results
            if result.get("cache_hit") and result.get("cache_label")
        ]
        if not hits:
            return
        for result in hits:
            label = result.get("cache_label")
            if not isinstance(label, str) or not label:
                continue
            action = result.get("cache_action") or "click"
            resolution = result.get("cache_resolution")
            if not resolution:
                if not self.automation_driver:
                    continue
                try:
                    resolution = await self.automation_driver.get_viewport_size()
                except Exception:
                    continue
            self.coordinate_cache.invalidate(label, action, resolution)
            if trace:
                trace.record_cache_event(
                    {
                        "type": "coordinate_cache_invalidate",
                        "scenario": current_test_plan_name,
                        "step": plan_cache_key,
                        "cache_label": label,
                        "cache_action": action,
                        "resolution": resolution,
                    }
                )

    def _invalidate_replay_entry(
        self,
        *,
        key: ExecutionReplayCacheKey,
        trace: Any,
        reason: str,
        **extra: Any,
    ) -> None:
        try:
            self.invalidate(key)
            if trace:
                trace.record_cache_event(
                    {
                        "type": "execution_replay_cache_invalidate",
                        "scenario": key.scenario,
                        "step": key.step,
                        "environment": key.environment,
                        "resolution": key.resolution,
                        "keyboard_layout": key.keyboard_layout,
                        "reason": reason,
                        **extra,
                    }
                )
        except Exception:
            logger.debug(
                "TestRunner: execution replay cache invalidation failed",
                exc_info=True,
            )

    @staticmethod
    def _record_replay_fallback(
        key: ExecutionReplayCacheKey,
        *,
        trace: Any,
    ) -> None:
        if trace:
            trace.record_cache_event(
                {
                    "type": "execution_replay_fallback_to_cu",
                    "scenario": key.scenario,
                    "step": key.step,
                }
            )
