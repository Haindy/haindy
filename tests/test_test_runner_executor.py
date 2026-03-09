"""Focused tests for the TestRunner executor collaborator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.test_runner_executor import (
    StepActionExecutionRequest,
    TestRunnerExecutor,
)
from src.core.types import StepIntent, TestStep


class _ResetDriver:
    def __init__(self) -> None:
        self.force_stop_app = AsyncMock()
        self.clear_app_data = AsyncMock()
        self.launch_app = AsyncMock()


def _build_step() -> TestStep:
    return TestStep(
        step_number=1,
        description="Reset the app",
        action="Reset the app",
        expected_result="App is relaunched cleanly",
        intent=StepIntent.SETUP,
        environment="mobile_adb",
    )


@pytest.mark.asyncio
async def test_executor_reset_app_uses_state_context_package_metadata() -> None:
    driver = _ResetDriver()
    artifacts = SimpleNamespace(
        capture_screenshot=AsyncMock(
            return_value=SimpleNamespace(
                screenshot_bytes=b"reset-screenshot",
                screenshot_path="debug_screenshots/tcTC001_step1_reset_after.png",
            )
        )
    )
    executor = TestRunnerExecutor(
        action_agent=None,
        automation_driver=driver,
        artifacts=artifacts,
    )

    result = await executor.execute_action(
        StepActionExecutionRequest(
            action={
                "type": "reset_app",
                "description": "Reset app state",
                "critical": True,
            },
            step=_build_step(),
            runtime_environment="mobile_adb",
            current_test_plan_name="Plan",
            current_test_case_name="Case",
            current_test_case_id="TC001",
            state_context={
                "entry_setup": {
                    "app_package": "com.example.app",
                    "app_activity": "MainActivity",
                }
            },
            recent_actions=[],
            record_driver_actions=False,
        )
    )

    assert result.compatibility_result["success"] is True
    driver.force_stop_app.assert_awaited_once()
    driver.clear_app_data.assert_awaited_once()
    driver.launch_app.assert_awaited_once_with("com.example.app", "MainActivity")
    artifacts.capture_screenshot.assert_awaited_once_with(
        "tcTC001_step1_reset_after",
        origin="reset_app_1_after",
        update_latest=True,
    )
    assert result.action_data["screenshots"]["after"].endswith(
        "tcTC001_step1_reset_after.png"
    )


@pytest.mark.asyncio
async def test_executor_returns_error_when_agent_or_driver_missing() -> None:
    executor = TestRunnerExecutor(action_agent=None, automation_driver=None)

    result = await executor.execute_action(
        StepActionExecutionRequest(
            action={
                "type": "click",
                "target": "Submit",
                "description": "Click submit",
                "critical": True,
            },
            step=_build_step().model_copy(update={"environment": "desktop"}),
            runtime_environment="desktop",
            current_test_plan_name="Plan",
            current_test_case_name="Case",
            current_test_case_id="TC001",
            state_context={},
            recent_actions=[],
            record_driver_actions=False,
        )
    )

    assert result.compatibility_result == {
        "success": False,
        "error": "Action agent or automation driver not available",
    }
