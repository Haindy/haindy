"""Step execution helpers for TestRunner."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, cast

from src.agents.test_runner_executor import (
    StepActionExecutionRequest,
    TestRunnerExecutor,
)
from src.agents.test_runner_interpreter import (
    StepInterpretationRequest,
    TestRunnerInterpreter,
)
from src.core.types import StepResult, TestCase, TestCaseResult, TestStatus, TestStep
from src.monitoring.logger import get_logger
from src.runtime.execution_replay_service import (
    ExecutionReplayService,
    ReplayExecutionRequest,
)

logger = get_logger(__name__)

MAX_TURN_ERROR_PREFIX = "Computer Use max turns exceeded"
LOOP_ERROR_PREFIX = "Computer Use loop detected"

COMPUTER_USE_PROMPT_MANUAL = """Computer Use execution context:
- Executor: OpenAI Computer Use tool powered by GPT-5.2 with medium reasoning effort.
- Environment: Existing runtime UI session (desktop browser/app or Android mobile screenshot) with screenshot-driven interaction.
- Inputs: Each prompt is delivered with the latest screenshot and scenario metadata; do not capture screenshots yourself.

Prompt construction rules:
1. Begin with a concise imperative goal that states the desired outcome.
2. Identify the UI target(s) using the exact labels a human sees (no CSS/XPath speculation).
3. Provide any text to enter or keys to press when relevant.
4. Restate the expected outcome based only on what should be *immediately visible* after the action completes — for example, a screen transition, a success message, or a new UI state. Never ask the executor to navigate to a different screen, open a profile, or take any additional step to confirm results; all deeper verification is handled by a separate evaluation pass.
5. Instruct the executor to act directly without seeking confirmation from the user.
6. If the screenshot already shows the desired outcome (e.g. a toggle is already selected, a field is already filled, the expected screen is already visible), report success immediately without interacting. Do not tap or click elements that are already in the correct state.
7. Do not embed retry logic, fallback strategies, or alternative tap targets in the prompt. Retries are handled automatically by the execution infrastructure. Focus each prompt solely on *what* to do and *what success looks like*.
8. Tell it to rely on the provided screenshot for context and to scroll or refocus if elements are off-screen.
9. For observation-only (`assert`) actions, explicitly forbid interactions and request a visual verification summary instead.
10. Avoid backend assumptions, hidden DOM references, or multi-step checklists—each prompt should cover one cohesive action.
11. After the primary action completes (or fails), stop. Do not take additional navigation steps to verify account details, confirm identity, or validate data that is not immediately visible on screen.
12. When a step is about entering text into one specific field (e.g. typing a verification/OTP code, entering an email, filling a single input), do NOT instruct the executor to also tap a submit/confirm/send/reset button. Just fill the field and stop. Only include button-tap instructions when the step's explicit purpose is to submit the form or the step action text says to tap that button.

If no interaction is required (`skip_navigation`), leave the computer_use_prompt empty.""".strip()


class TestRunnerStepProcessor:
    """Owns step execution, interpretation, and replay glue for TestRunner."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def execute_test_step(
        self, step: TestStep, test_case: TestCase, case_result: TestCaseResult
    ) -> StepResult:
        """Execute a single test step with intelligent interpretation."""
        logger.info(
            "Executing test step",
            extra={
                "step_number": step.step_number,
                "action": step.action,
                "test_case": test_case.name,
            },
        )

        runner = self._runner
        runner._current_test_step = step
        runner._current_step_actions = []
        runner._current_step_data = {
            "step_number": step.step_number,
            "step_id": str(step.step_id),
            "step_description": step.action,
            "actions": runner._current_step_actions,
            "step_intent": step.intent.value,
        }
        current_case_actions = runner._current_test_case_actions
        assert current_case_actions is not None
        current_case_actions["steps"].append(runner._current_step_data)

        step_result = StepResult(
            step_id=step.step_id,
            step_number=step.step_number,
            status=TestStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            action=step.action,
            expected_result=step.expected_result,
            actual_result="",
        )
        screenshot_before: bytes | None = None
        screenshot_after: bytes | None = None
        attempt = 1
        plan_cache_hit = False
        replay_used = False

        try:
            if not runner._check_dependencies(step, case_result):
                step_result.status = TestStatus.SKIPPED
                step_result.actual_result = "Skipped due to unmet dependencies"
                return step_result

            before_capture = await runner._artifacts.capture_test_step_screenshot(
                test_case=test_case,
                step=step,
                suffix="before",
                origin=f"step_{step.step_number}_before",
                update_latest=True,
            )
            if before_capture:
                screenshot_before = before_capture.screenshot_bytes
                step_result.screenshot_before = before_capture.screenshot_path

            execution_history = [
                {
                    "step_number": prev_step.step_number,
                    "action": prev_step.action,
                    "status": prev_step.status,
                    "actual_result": prev_step.actual_result,
                }
                for prev_step in case_result.step_results
            ]

            next_test_case = None
            if runner._current_test_plan:
                current_idx = None
                for idx, current_case in enumerate(
                    runner._current_test_plan.test_cases
                ):
                    if current_case.case_id == test_case.case_id:
                        current_idx = idx
                        break
                if (
                    current_idx is not None
                    and current_idx < len(runner._current_test_plan.test_cases) - 1
                ):
                    next_test_case = runner._current_test_plan.test_cases[
                        current_idx + 1
                    ]

            replay_result = await runner._try_execution_replay(
                step=step,
                test_case=test_case,
                step_result=step_result,
                screenshot_before=screenshot_before,
                execution_history=execution_history,
                next_test_case=next_test_case,
            )
            if replay_result is not None:
                replay_used = True
                return cast(StepResult, replay_result)

            latest_action_results: list[dict[str, Any]] = []
            while True:
                actions, plan_cache_hit = await runner._interpret_step(
                    step, test_case, case_result, use_cache=(attempt == 1)
                )
                action_results: list[dict[str, Any]] = []
                forced_blocker_reason = None
                step_result.actions_performed = []
                step_session = None

                if (
                    runner.action_agent is not None
                    and hasattr(runner.action_agent, "begin_step_session")
                    and callable(runner.action_agent.begin_step_session)
                ):
                    step_session = await runner.action_agent.begin_step_session(
                        step,
                        {
                            **(
                                runner._test_state.context
                                if runner._test_state
                                and isinstance(runner._test_state.context, dict)
                                else {}
                            ),
                            "test_plan_name": runner._current_test_plan.name
                            if runner._current_test_plan
                            else None,
                            "test_case_name": runner._current_test_case.name
                            if runner._current_test_case
                            else None,
                            "test_case_id": runner._current_test_case.test_id
                            if runner._current_test_case
                            else None,
                            "step_number": step.step_number,
                            "environment": runner._environment,
                        },
                    )

                try:
                    for action in actions:
                        logger.debug(
                            "Executing sub-action",
                            extra={
                                "action_type": action["type"],
                                "description": action.get("description", ""),
                            },
                        )

                        action_result = await runner._execute_action(
                            action,
                            step,
                            record_driver_actions=runner._replay_enabled(step),
                            step_session=step_session,
                        )
                        step_result.actions_performed.append(action_result)
                        action_results.append(
                            {
                                "action": action,
                                "result": action_result,
                                "full_data": runner._current_step_actions[-1]
                                if runner._current_step_actions
                                else {},
                            }
                        )

                        if forced_blocker_reason is None:
                            error_text = action_result.get("error")
                            if not error_text:
                                full_data = action_results[-1]["full_data"]
                                if isinstance(full_data, dict):
                                    result_blob = full_data.get("result", {})
                                    if isinstance(result_blob, dict):
                                        exec_blob = result_blob.get("execution") or {}
                                        error_text = result_blob.get(
                                            "error"
                                        ) or exec_blob.get("error_message")
                            if (
                                error_text
                                and isinstance(error_text, str)
                                and (
                                    error_text.startswith(MAX_TURN_ERROR_PREFIX)
                                    or error_text.startswith(LOOP_ERROR_PREFIX)
                                )
                            ):
                                forced_blocker_reason = error_text
                                logger.error(
                                    "Action aborted due to Computer Use limit or loop",
                                    extra={
                                        "step_number": step.step_number,
                                        "action_description": action.get(
                                            "description", ""
                                        ),
                                        "reason": error_text,
                                    },
                                )
                                runner._current_step_data["blocker_reasoning"] = (
                                    error_text
                                )
                                runner._current_step_data["forced_blocker_reason"] = (
                                    error_text
                                )

                        if not action_result.get("success", False) and action.get(
                            "critical", True
                        ):
                            break

                    latest_action_results = action_results
                    runner._current_step_data["plan_cache_hit"] = plan_cache_hit

                    latest_after = runner._artifacts.latest_screenshot_bytes
                    latest_after_path = runner._artifacts.latest_screenshot_path
                    if latest_after is not None and latest_after_path is not None:
                        screenshot_after = latest_after
                        step_result.screenshot_after = latest_after_path
                    else:
                        if runner.automation_driver:
                            await asyncio.sleep(1)
                        after_capture = (
                            await runner._artifacts.capture_test_step_screenshot(
                                test_case=test_case,
                                step=step,
                                suffix="after",
                                origin=f"step_{step.step_number}_after",
                                update_latest=True,
                            )
                        )
                        if after_capture:
                            screenshot_after = after_capture.screenshot_bytes
                            step_result.screenshot_after = after_capture.screenshot_path

                    try:
                        if forced_blocker_reason:
                            verification = {
                                "verdict": "FAIL",
                                "reasoning": forced_blocker_reason,
                                "actual_result": forced_blocker_reason,
                                "confidence": 1.0,
                                "is_blocker": True,
                                "blocker_reasoning": forced_blocker_reason,
                            }
                            runner._current_step_data["verification_mode"] = (
                                "runner_short_circuit"
                            )
                        elif step_session is not None and getattr(
                            step_session, "has_computer_use_action", False
                        ):
                            validation_result = (
                                await runner.action_agent.validate_step_with_session(
                                    step_session=step_session,
                                    test_case=test_case,
                                    step=step,
                                    action_results=action_results,
                                    execution_history=execution_history,
                                    next_test_case=next_test_case,
                                )
                            )
                            verification = validation_result.verification
                            runner._current_step_data["verification_mode"] = (
                                "action_agent_session"
                            )
                            runner._current_step_data[
                                "action_agent_step_validation"
                            ] = {
                                "prompt": validation_result.prompt,
                                "response": validation_result.raw_response,
                                "response_ids": validation_result.response_ids,
                                "session_response_ids": getattr(
                                    step_session, "response_ids", []
                                ),
                            }
                        else:
                            verification = await runner._verify_expected_outcome(
                                test_case=test_case,
                                step=step,
                                action_results=action_results,
                                screenshot_before=screenshot_before,
                                screenshot_after=screenshot_after,
                                execution_history=execution_history,
                                next_test_case=next_test_case,
                            )
                            runner._current_step_data["verification_mode"] = "ai"

                        runner._current_step_data["verification_result"] = verification
                        step_result.status = (
                            TestStatus.PASSED
                            if verification["verdict"] == "PASS"
                            else TestStatus.FAILED
                        )
                        step_result.actual_result = verification["actual_result"]
                        step_result.error_message = (
                            verification["reasoning"]
                            if verification["verdict"] == "FAIL"
                            else None
                        )
                        step_result.confidence = verification.get("confidence", 0.0)

                        if verification["verdict"] == "FAIL":
                            runner._current_step_data["is_blocker"] = verification.get(
                                "is_blocker", False
                            )
                            runner._current_step_data["blocker_reasoning"] = (
                                verification.get("blocker_reasoning", "")
                            )

                    except Exception as exc:
                        logger.error(
                            "AI verification failed - marking test as failed",
                            extra={
                                "error": str(exc),
                                "step_number": step.step_number,
                                "test_case": test_case.name,
                            },
                        )
                        step_result.status = TestStatus.FAILED
                        step_result.actual_result = (
                            "Verification failed due to AI error"
                        )
                        step_result.error_message = (
                            f"AI verification failed: {str(exc)}"
                        )
                        step_result.confidence = 0.0
                        raise
                finally:
                    if (
                        step_session is not None
                        and runner.action_agent is not None
                        and hasattr(runner.action_agent, "end_step_session")
                        and callable(runner.action_agent.end_step_session)
                    ):
                        await runner.action_agent.end_step_session(step_session)

                if (
                    verification["verdict"] == "PASS"
                    or not plan_cache_hit
                    or attempt >= 2
                ):
                    break

                cache_key_raw = runner._current_step_data.get("plan_cache_key")
                cache_context_raw = runner._current_step_data.get("plan_cache_context")
                cache_key = cache_key_raw if isinstance(cache_key_raw, str) else None
                cache_context = (
                    cache_context_raw if isinstance(cache_context_raw, dict) else None
                )
                if cache_key and cache_context:
                    logger.warning(
                        "Cached plan failed; invalidating and retrying without cache",
                        extra={"step": cache_key, "attempt": attempt},
                    )
                    try:
                        runner._task_plan_cache.invalidate(cache_key, cache_context)
                        if runner._trace:
                            runner._trace.record_cache_event(
                                {
                                    "type": "task_plan_cache_invalidate",
                                    "scenario": runner._current_test_plan.name
                                    if runner._current_test_plan
                                    else "",
                                    "step": cache_key,
                                    "reason": "validation_failed_with_cached_plan",
                                }
                            )
                    except Exception:
                        logger.debug(
                            "Failed to invalidate task plan cache",
                            exc_info=True,
                        )

                attempt += 1

            cache_key_raw = runner._current_step_data.get("plan_cache_key")
            cache_context_raw = runner._current_step_data.get("plan_cache_context")
            cache_key = cache_key_raw if isinstance(cache_key_raw, str) else None
            cache_context = (
                cache_context_raw if isinstance(cache_context_raw, dict) else None
            )
            if step_result.status == TestStatus.PASSED:
                if cache_key and cache_context and not plan_cache_hit:
                    try:
                        runner._task_plan_cache.store(cache_key, cache_context, actions)
                        if runner._trace:
                            runner._trace.record_cache_event(
                                {
                                    "type": "task_plan_cache_store",
                                    "scenario": runner._current_test_plan.name
                                    if runner._current_test_plan
                                    else "",
                                    "step": cache_key,
                                }
                            )
                    except Exception:
                        logger.debug("Failed to store task plan cache", exc_info=True)

                await runner._store_execution_replay(
                    step, test_case, latest_action_results
                )
                await runner._persist_coordinate_cache(latest_action_results)
            else:
                await runner._invalidate_coordinate_cache(latest_action_results)

        except Exception as exc:
            logger.error(
                "Step execution failed",
                extra={"error": str(exc), "step_number": step.step_number},
            )
            step_result.status = TestStatus.FAILED
            step_result.actual_result = f"Error: {str(exc)}"
            step_result.error_message = str(exc)

        finally:
            step_result.completed_at = datetime.now(timezone.utc)
            if (
                step_result.screenshot_after is None
                and screenshot_after is None
                and screenshot_before is not None
                and step_result.screenshot_before
            ):
                runner._artifacts.update_latest_snapshot(
                    screenshot_before,
                    step_result.screenshot_before,
                    f"step_{step.step_number}_before",
                )

            runner._execution_history.append(
                {
                    "test_case": test_case.name,
                    "step": step.step_number,
                    "action": step.action,
                    "result": step_result.status.value,
                    "timestamp": step_result.completed_at,
                }
            )

            if runner._trace and not replay_used:
                runner._trace.record_step(
                    scenario_name=runner._current_test_plan.name
                    if runner._current_test_plan
                    else "",
                    step=step,
                    step_result=step_result,
                    attempt=attempt,
                    plan_cache_hit=plan_cache_hit,
                )

        return step_result

    async def interpret_step(
        self,
        step: TestStep,
        test_case: TestCase,
        case_result: TestCaseResult,
        use_cache: bool = True,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Interpret a test step and decompose it into executable actions."""
        runner = self._runner
        runner._interpreter = TestRunnerInterpreter(
            automation_driver=runner.automation_driver,
            get_interpretation_screenshot=runner._get_interpretation_screenshot,
            task_plan_cache=runner._task_plan_cache,
            trace=runner._trace,
            model_logger=runner._model_logger,
            model=runner.model,
            call_openai=runner.call_openai,
        )
        interpretation = await runner._interpreter.interpret_step(
            StepInterpretationRequest(
                step=step,
                test_case=test_case,
                case_result=case_result,
                current_test_plan=runner._current_test_plan,
                test_report=runner._test_report,
                execution_history=runner._execution_history,
                runtime_environment=runner._environment or "desktop",
                computer_use_prompt_manual=COMPUTER_USE_PROMPT_MANUAL,
                use_cache=use_cache,
            )
        )
        if hasattr(runner, "_current_step_data"):
            runner._current_step_data["interpretation_context"] = (
                interpretation.interpretation_context
            )
            runner._current_step_data["plan_cache_hit"] = interpretation.cache_hit
            runner._current_step_data["plan_cache_key"] = interpretation.plan_cache_key
            runner._current_step_data["plan_cache_context"] = (
                interpretation.plan_cache_context
            )
            runner._current_step_data["test_runner_interpretation"] = (
                interpretation.interpretation_record
            )
        return interpretation.actions, interpretation.cache_hit

    async def execute_action(
        self,
        action: dict[str, Any],
        step: TestStep,
        record_driver_actions: bool = False,
        step_session: Any | None = None,
    ) -> dict[str, Any]:
        """Execute a single decomposed action with comprehensive tracking."""
        runner = self._runner
        runner._executor = TestRunnerExecutor(
            action_agent=runner.action_agent,
            automation_driver=runner.automation_driver,
            artifacts=runner._artifacts,
        )
        execution = await runner._executor.execute_action(
            StepActionExecutionRequest(
                action=action,
                step=step,
                runtime_environment=runner._environment,
                current_test_plan_name=runner._current_test_plan.name
                if runner._current_test_plan
                else None,
                current_test_case_name=runner._current_test_case.name
                if runner._current_test_case
                else None,
                current_test_case_id=runner._current_test_case.test_id
                if runner._current_test_case
                else None,
                state_context=runner._test_state.context
                if runner._test_state and isinstance(runner._test_state.context, dict)
                else {},
                recent_actions=runner._execution_history[-3:],
                record_driver_actions=record_driver_actions,
                screenshot=runner._artifacts.latest_screenshot_bytes,
                step_session=step_session,
            )
        )
        if runner._current_step_actions is None:
            runner._current_step_actions = []
        runner._current_step_actions.append(execution.action_data)
        return cast(dict[str, Any], execution.compatibility_result)

    @staticmethod
    def plan_cache_key(step: TestStep, test_case: TestCase) -> str:
        return TestRunnerInterpreter.plan_cache_key(step, test_case)

    @staticmethod
    def is_validation_only_action_result(result: dict[str, Any]) -> bool:
        return ExecutionReplayService.is_validation_only_action_result(result)

    @classmethod
    def is_validation_only_step_result_set(
        cls, action_results: list[dict[str, Any]]
    ) -> bool:
        return ExecutionReplayService.is_validation_only_step_result_set(action_results)

    @staticmethod
    def driver_actions_for_replay(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Accept both wrapped and direct action-result payload shapes."""
        return ExecutionReplayService.driver_actions_for_replay(result)

    def replay_enabled(self, step: TestStep) -> bool:
        return bool(self._runner._replay_service.replay_enabled(step))

    def replay_stabilization_wait_ms(self) -> int:
        """Return stabilization wait used only for replayed macro actions."""
        return int(self._runner._replay_service.replay_stabilization_wait_ms())

    async def execution_replay_key(
        self, step: TestStep, test_case: TestCase
    ) -> Any | None:
        runner = self._runner
        return await runner._replay_service.execution_replay_key(
            step=step,
            test_case=test_case,
            current_test_plan_name=runner._current_test_plan.name
            if runner._current_test_plan
            else "",
            current_runtime_environment=runner._environment,
            plan_cache_key=runner._plan_cache_key(step, test_case),
            plan_fingerprint=runner._plan_fingerprint(),
        )

    async def try_execution_replay(
        self,
        *,
        step: TestStep,
        test_case: TestCase,
        step_result: StepResult,
        screenshot_before: bytes | None,
        execution_history: list[dict[str, Any]],
        next_test_case: TestCase | None,
    ) -> StepResult | None:
        runner = self._runner

        async def _capture_replay_screenshot(
            suffix: str,
        ) -> tuple[bytes | None, str | None]:
            capture = await runner._artifacts.capture_test_step_screenshot(
                test_case=test_case,
                step=step,
                suffix=suffix,
                origin=f"step_{step.step_number}_{suffix}",
                update_latest=True,
            )
            if capture is None:
                return None, None
            return capture.screenshot_bytes, capture.screenshot_path

        replay_result = await runner._replay_service.try_execution_replay(
            ReplayExecutionRequest(
                step=step,
                test_case=test_case,
                step_result=step_result,
                screenshot_before=screenshot_before,
                execution_history=execution_history,
                next_test_case=next_test_case,
                current_test_plan_name=runner._current_test_plan.name
                if runner._current_test_plan
                else "",
                current_runtime_environment=runner._environment,
                plan_cache_key=runner._plan_cache_key(step, test_case),
                plan_fingerprint=runner._plan_fingerprint(),
                capture_replay_screenshot=_capture_replay_screenshot,
                verify_expected_outcome=runner._verify_expected_outcome,
                coerce_model_bool=runner._coerce_model_bool,
                trace=runner._trace,
            )
        )
        if replay_result is None:
            return None

        if runner._current_step_actions is None:
            runner._current_step_actions = []
        runner._current_step_actions.append(replay_result.action_record)
        step_result.actions_performed = replay_result.actions_performed
        runner._current_step_data["verification_mode"] = "ai"
        runner._current_step_data["replay_validation_wait_spent_ms"] = (
            replay_result.replay_validation_wait_spent_ms
        )
        runner._current_step_data["replay_validation_wait_cycles"] = (
            replay_result.replay_validation_wait_cycles
        )
        runner._current_step_data["replay_validation_wait_budget_remaining_ms"] = (
            replay_result.replay_validation_wait_budget_remaining_ms
        )
        runner._current_step_data["verification_result"] = replay_result.verification
        if replay_result.fallback_to_cu:
            return None
        return step_result

    async def store_execution_replay(
        self,
        step: TestStep,
        test_case: TestCase,
        action_results: list[dict[str, Any]],
    ) -> None:
        runner = self._runner
        recorded_action_count = sum(
            len(self.driver_actions_for_replay(result)) for result in action_results
        )
        key = await runner._replay_service.store_execution_replay(
            step=step,
            test_case=test_case,
            action_results=action_results,
            current_test_plan_name=runner._current_test_plan.name
            if runner._current_test_plan
            else "",
            current_runtime_environment=runner._environment,
            plan_cache_key=runner._plan_cache_key(step, test_case),
            plan_fingerprint=runner._plan_fingerprint(),
        )
        if key and runner._trace:
            runner._trace.record_cache_event(
                {
                    "type": "execution_replay_cache_store",
                    "scenario": key.scenario,
                    "step": key.step,
                    "environment": key.environment,
                    "resolution": key.resolution,
                    "keyboard_layout": key.keyboard_layout,
                    "action_count": recorded_action_count,
                }
            )

    async def persist_coordinate_cache(
        self, action_results: list[dict[str, Any]]
    ) -> None:
        runner = self._runner
        plan_cache_key = runner._current_step_data.get("plan_cache_key")
        await runner._replay_service.persist_coordinate_cache(
            action_results=action_results,
            plan_cache_key=plan_cache_key if isinstance(plan_cache_key, str) else None,
            current_test_plan_name=runner._current_test_plan.name
            if runner._current_test_plan
            else "",
            trace=runner._trace,
        )

    async def invalidate_coordinate_cache(
        self, action_results: list[dict[str, Any]]
    ) -> None:
        runner = self._runner
        plan_cache_key = runner._current_step_data.get("plan_cache_key")
        await runner._replay_service.invalidate_coordinate_cache(
            action_results=action_results,
            plan_cache_key=plan_cache_key if isinstance(plan_cache_key, str) else None,
            current_test_plan_name=runner._current_test_plan.name
            if runner._current_test_plan
            else "",
            trace=runner._trace,
        )
