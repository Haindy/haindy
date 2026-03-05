"""Replay and coordinate cache helpers for TestRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.types import TestCase, TestStep
from src.desktop.cache import CoordinateCache
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
        self.store(key, recorded)
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
