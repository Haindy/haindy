"""Step interpretation collaborator for TestRunner."""

from __future__ import annotations

import base64
import json
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from haindy.agents.structured_output_schemas import STEP_INTERPRETATION_RESPONSE_FORMAT
from haindy.core.types import (
    StepResult,
    TestCase,
    TestCaseResult,
    TestPlan,
    TestReport,
    TestStatus,
    TestStep,
)
from haindy.monitoring.logger import get_logger
from haindy.runtime.task_cache import TaskPlanCache
from haindy.runtime.trace import RunTraceWriter

logger = get_logger(__name__)


@dataclass(frozen=True)
class StepInterpretationRequest:
    step: TestStep
    test_case: TestCase
    case_result: TestCaseResult
    current_test_plan: TestPlan | None
    test_report: TestReport | None
    execution_history: list[dict[str, Any]]
    runtime_environment: str
    computer_use_prompt_manual: str
    use_cache: bool = True


@dataclass(frozen=True)
class StepInterpretationResult:
    actions: list[dict[str, Any]]
    cache_hit: bool
    plan_cache_key: str
    plan_cache_context: dict[str, Any]
    interpretation_context: dict[str, Any]
    interpretation_record: dict[str, Any]


class TestRunnerInterpreter:
    """Builds prompts and parses step action plans."""

    def __init__(
        self,
        *,
        automation_driver: Any,
        get_interpretation_screenshot: Callable[
            [TestStep, TestCase],
            Awaitable[tuple[bytes | None, str | None, str | None]],
        ],
        task_plan_cache: TaskPlanCache,
        trace: RunTraceWriter | None,
        model_logger: Any,
        model: str,
        call_model: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        self._automation_driver = automation_driver
        self._get_interpretation_screenshot = get_interpretation_screenshot
        self._task_plan_cache = task_plan_cache
        self._trace = trace
        self._model_logger = model_logger
        self._model = model
        self._call_model = call_model

    async def interpret_step(
        self,
        request: StepInterpretationRequest,
    ) -> StepInterpretationResult:
        """Interpret a step and return executable action definitions."""
        step = request.step
        test_case = request.test_case
        case_result = request.case_result

        recent_history = []
        for item in request.execution_history[-3:]:
            timestamp = item.get("timestamp")
            recent_history.append(
                {
                    "test_case": item.get("test_case", ""),
                    "step": item.get("step", 0),
                    "action": item.get("action", ""),
                    "result": item.get("result", ""),
                    "timestamp": (
                        timestamp.isoformat()
                        if isinstance(timestamp, datetime)
                        else str(timestamp or "")
                    ),
                }
            )
        recent_history_text = json.dumps(recent_history, indent=2)

        ordered_steps = self._ordered_case_steps(test_case)
        total_steps = len(ordered_steps)
        step_index = next(
            (
                idx
                for idx, (_, _, candidate) in enumerate(ordered_steps)
                if candidate.step_id == step.step_id
            ),
            max(0, step.step_number - 1),
        )

        previous_step_summary = "No previous steps in this test case."
        if step_index > 0:
            (
                previous_phase,
                previous_ordinal,
                previous_step,
            ) = ordered_steps[step_index - 1]
            previous_result = self._lookup_step_result(case_result, previous_step)
            previous_label = self._step_display_label(
                previous_phase,
                previous_ordinal,
                previous_step,
            )
            previous_step_summary = self._format_step_summary(
                previous_step,
                previous_result,
                label=previous_label,
            )

        next_step_summary = "This is the final step in this test case."
        if step_index < total_steps - 1:
            next_phase, next_ordinal, next_step = ordered_steps[step_index + 1]
            next_label = self._step_display_label(next_phase, next_ordinal, next_step)
            next_step_summary = self._format_step_summary(
                next_step,
                None,
                include_status=False,
                label=next_label,
            )

        previous_case_summary = self._previous_case_summary(
            test_case=test_case,
            current_test_plan=request.current_test_plan,
            test_report=request.test_report,
        )

        case_outline_lines = [
            f"{self._step_display_label(phase, ordinal, case_step)}: {case_step.action} "
            f"(intent: {case_step.intent.value}, expected: {case_step.expected_result})"
            + (" [CURRENT STEP]" if case_step.step_id == step.step_id else "")
            for phase, ordinal, case_step in ordered_steps
        ]
        prereq_lines = (
            [f"  - {prereq}" for prereq in test_case.prerequisites]
            if test_case.prerequisites
            else []
        )
        prereq_prefix = (
            "Preconditions:\n" + "\n".join(prereq_lines) + "\n\n"
            if prereq_lines
            else ""
        )
        case_outline_text = prereq_prefix + "\n".join(
            f"- {line}" for line in case_outline_lines
        )

        cache_context = {
            "test_case_id": test_case.test_id,
            "test_case_name": test_case.name,
            "step_number": step.step_number,
            "step_action": step.action,
            "expected_result": step.expected_result,
            "intent": step.intent.value,
            "previous_step_summary": previous_step_summary,
            "next_step_summary": next_step_summary,
            "previous_test_case": previous_case_summary,
            "case_outline": case_outline_lines,
        }
        step_cache_key = self.plan_cache_key(step, test_case)

        if request.use_cache:
            cached_actions = self._task_plan_cache.lookup(step_cache_key, cache_context)
            if cached_actions:
                logger.info(
                    "Using cached action plan for step",
                    extra={"step": step_cache_key, "action_count": len(cached_actions)},
                )
                if self._trace:
                    self._trace.record_cache_event(
                        {
                            "type": "task_plan_cache_hit",
                            "scenario": (
                                request.current_test_plan.name
                                if request.current_test_plan
                                else ""
                            ),
                            "step": step_cache_key,
                        }
                    )
                return StepInterpretationResult(
                    actions=cached_actions,
                    cache_hit=True,
                    plan_cache_key=step_cache_key,
                    plan_cache_context=cache_context,
                    interpretation_context=cache_context,
                    interpretation_record={
                        "prompt": None,
                        "response": {"actions": cached_actions},
                        "screenshot_path": None,
                        "screenshot_source": "plan_cache",
                        "context": cache_context,
                        "cache_key": step_cache_key,
                    },
                )

        (
            screenshot_bytes,
            screenshot_path,
            screenshot_source,
        ) = await self._get_interpretation_screenshot(
            step,
            test_case,
        )
        screenshot_b64 = (
            base64.b64encode(screenshot_bytes).decode("ascii")
            if screenshot_bytes
            else None
        )

        runtime_environment = request.runtime_environment or "desktop"
        viewport_hint = "unknown"
        if self._automation_driver:
            try:
                width, height = await self._automation_driver.get_viewport_size()
                viewport_hint = f"{width}x{height}"
            except Exception:
                viewport_hint = "unknown"
        interaction_hint = (
            "Android mobile application in screenshot-space coordinates"
            if runtime_environment == "mobile_adb"
            else "desktop/web UI in screenshot-space coordinates"
        )
        environment_specific_guidance = ""
        if runtime_environment == "mobile_adb":
            environment_specific_guidance = """
Mobile-specific constraints:
- This run targets Android mobile UI. Do not use desktop/browser navigation assumptions.
- Do NOT propose keyboard/browser shortcuts like Alt+Left, Alt+Right, Ctrl+L, Ctrl+Tab, or "browser back".
- Prefer tap/swipe interactions and Android-safe navigation.
- If back navigation is required, use `key_press` with value "back" or tap a visible in-app/system back control.
- If a setup step requires launching the app in a clean state with no active user session (e.g., "reset app", "clean state", "freshly launched", "no active session"), emit a single `reset_app` action instead of trying to navigate or sign out. Leave computer_use_prompt empty for this action type.
- If a setup step is explicitly about letting startup or loading finish (for example, waiting through a splash screen until a welcome/login screen appears), emit a single `wait` action with a condition-focused computer_use_prompt instead of compressing that behavior into `reset_app`.
""".strip()

        prompt = f"""You are the HAINDY Test Runner's interpretation agent. Use the current UI snapshot and scenario context to plan the minimal actions needed for the next step. You are preparing instructions for an automated Computer Use executor that will run them without further translation.

Run & Screenshot Context:
- Test case: {test_case.test_id} – {test_case.name}
- Test case description: {test_case.description}
- Step position: {step_index + 1} of {total_steps} (declared step number: {step.step_number}, intent: {step.intent.value})
- Runtime backend: {runtime_environment}
- Interaction mode: {interaction_hint}
- Viewport hint: {viewport_hint}
- Screenshot path: {screenshot_path or "unavailable"}
- Screenshot source: {screenshot_source or "unknown"}

Previous step summary:
{previous_step_summary}

Next step preview:
{next_step_summary}

Previous test case context:
{previous_case_summary}

Full test case outline:
{case_outline_text}

Recent execution history (most recent first):
{recent_history_text}

Computer Use executor manual (follow this precisely when writing prompts):
{request.computer_use_prompt_manual}

Guidelines:
1. Inspect the screenshot before planning navigation. If the required view is already visible, emit a single `skip_navigation` action that explains the evidence (leave computer_use_prompt empty in that case).
2. Provide high-level, outcome-focused actions. For text or form inputs, emit a single `type` action with the final value and let the Computer Use model handle focusing, clearing, or key presses—do not add helper clicks for the same control.
3. When the step is about startup stabilization or waiting for a visible ready state (for example, waiting for a splash/loading screen to clear), prefer a single `wait` action over `skip_navigation`. Use `wait` only when the step is explicitly about allowing the UI to become ready.
4. Only break a step into multiple actions when it truly touches different controls (e.g., separate date and time pickers). Otherwise, keep the entire outcome in one action so the executor can decide the mechanics.
5. Keep targets human-readable (no selectors) and ensure each action advances toward the expected result: {step.expected_result}.
6. Use the previous/next step context to stay aligned with the intended flow.
7. Every action except `skip_navigation` and `reset_app` must include a `computer_use_prompt` that is ready to send directly to the Computer Use model—no additional wrapping will be added later.
8. You are planning actions for the step marked [CURRENT STEP] ONLY. Do not plan actions for any other step. Even if the screenshot appears to show a later step's target already populated or completed, still execute the current step's action on the correct target — the visual state may reflect autofill, prior test state, or an incorrect field.
9. When the step action names a specific UI element, label, option, or role (e.g., "Athlete", "Coach or Team Admin", "Join Team", "Skip for now"), copy that exact text into the computer_use_prompt. Do not substitute synonyms, shorten the label, or infer a different option. If the step says "Athlete", the prompt must say "Athlete", not "role card" or "Coach".

{environment_specific_guidance}

Action schema for each entry (JSON object):
- type: One of [navigate, click, type, wait, assert, key_press, scroll_to_element, scroll_by_pixels, scroll_to_top, scroll_to_bottom, scroll_horizontal, skip_navigation, reset_app].
  • Use `wait` when the step is explicitly about allowing loading or startup to finish, or waiting until a specific visible ready state appears. Keep the target/description human-readable and make the `computer_use_prompt` state exactly what visible condition should end the wait.
  • Use `skip_navigation` only when navigation is already satisfied; do not provide a value.
  • Use `reset_app` (mobile_adb only) when a setup step requires a completely clean app state — no active session, no cached data. This action force-stops the app, clears all its data, and relaunches it. No computer_use_prompt is needed; leave it empty.
- target: Human description of the element or high-level goal.
- value: Required only when the action type needs input (navigate URL, type text, key_press key, scroll_by_pixels amount).
- description: Outcome-focused explanation so the Action Agent knows what success looks like.
- critical: Whether failure should halt remaining actions (true/false).
- expected_outcome (optional): Override the step-level expected result only if needed.
- computer_use_prompt: String containing the final directive for the Computer Use model, constructed according to the manual above. Required unless type is `skip_navigation` or `reset_app`.

Respond with a JSON object containing an "actions" array where every item follows this schema exactly."""

        message_content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if screenshot_b64:
            message_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                }
            )

        interpretation_context_payload = {
            "screenshot_path": screenshot_path,
            "screenshot_source": screenshot_source,
            "previous_step_summary": previous_step_summary,
            "next_step_summary": next_step_summary,
            "previous_test_case": previous_case_summary,
            "case_outline": case_outline_lines,
            "recent_history": recent_history,
        }

        logger.info(
            "Interpreting step with AI",
            extra={
                "test_case": test_case.test_id,
                "step_number": step.step_number,
                "action": step.action,
                "expected_result": step.expected_result,
                "intent": step.intent.value,
                "prompt_length": len(prompt),
                "screenshot_path": screenshot_path,
                "screenshot_source": screenshot_source,
            },
        )

        try:
            response = await self._call_model(
                messages=[{"role": "user", "content": message_content}],
                response_format=STEP_INTERPRETATION_RESPONSE_FORMAT,
                log_agent="test_runner.interpret_step",
                log_metadata={
                    "step_number": step.step_number,
                    "test_case": test_case.name,
                    "cache_key": step_cache_key,
                },
            )

            logger.debug(
                "OpenAI API call successful",
                extra={
                    "response_type": type(response).__name__,
                    "response_keys": (
                        list(response.keys()) if isinstance(response, dict) else None
                    ),
                },
            )

            content = response.get("content", {})
            if isinstance(content, list):
                # Some models (e.g. Google Gemini) return the actions array
                # directly as a top-level JSON array instead of wrapping it.
                content = {"actions": content}
            elif not isinstance(content, dict):
                raise TypeError(f"Expected dict content but got {type(content)}")

            actions = content.get("actions", [])
            if not actions:
                raise ValueError(
                    f"AI failed to provide actions for step {step.step_number}: {step.action}"
                )

            logger.debug(
                "Step interpretation successful",
                extra={
                    "test_case": test_case.test_id,
                    "step": step.step_number,
                    "original_action": step.action,
                    "decomposed_actions": len(actions),
                },
            )

            return StepInterpretationResult(
                actions=actions,
                cache_hit=False,
                plan_cache_key=step_cache_key,
                plan_cache_context=cache_context,
                interpretation_context=interpretation_context_payload,
                interpretation_record={
                    "prompt": prompt,
                    "response": response.get("content", {}),
                    "screenshot_path": screenshot_path,
                    "screenshot_source": screenshot_source,
                    "context": interpretation_context_payload,
                },
            )
        except Exception as exc:
            logger.error(
                "Failed to interpret step with AI",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "step": step.step_number,
                    "action": step.action,
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    @staticmethod
    def plan_cache_key(step: TestStep, test_case: TestCase) -> str:
        return f"{test_case.test_id}:{step.step_number}:{step.action}".strip()

    @staticmethod
    def _format_step_summary(
        step_obj: TestStep,
        result_obj: StepResult | None,
        *,
        include_status: bool = True,
        label: str | None = None,
    ) -> str:
        base = (
            f"{label or f'Step {step_obj.step_number}'}: {step_obj.action}\n"
            f"  - Expected: {step_obj.expected_result}"
        )
        if include_status:
            if result_obj:
                status_value = result_obj.status.value.upper()
                actual = result_obj.actual_result or "Not recorded"
                base += f"\n  - Status: {status_value}\n  - Actual: {actual}"
            else:
                base += "\n  - Status: NOT_EXECUTED\n  - Actual: N/A"
        return base

    @staticmethod
    def _ordered_case_steps(
        test_case: TestCase,
    ) -> list[tuple[str, int, TestStep]]:
        ordered_steps: list[tuple[str, int, TestStep]] = []
        for phase, steps in (
            ("setup", test_case.setup_steps),
            ("main", test_case.steps),
            ("cleanup", test_case.cleanup_steps),
        ):
            for ordinal, case_step in enumerate(steps, start=1):
                ordered_steps.append((phase, ordinal, case_step))
        return ordered_steps

    @staticmethod
    def _step_display_label(
        phase: str,
        ordinal: int,
        step: TestStep,
    ) -> str:
        if phase == "setup":
            return f"Setup Step {ordinal}"
        if phase == "cleanup":
            return f"Cleanup Step {ordinal}"
        return f"Step {step.step_number}"

    @staticmethod
    def _lookup_step_result(
        case_result: TestCaseResult,
        step: TestStep,
    ) -> StepResult | None:
        for result_group in (
            case_result.setup_step_results,
            case_result.step_results,
            case_result.cleanup_step_results,
        ):
            for step_result in result_group:
                if step_result.step_id == step.step_id:
                    return step_result
        return None

    def _previous_case_summary(
        self,
        *,
        test_case: TestCase,
        current_test_plan: TestPlan | None,
        test_report: TestReport | None,
    ) -> str:
        if not test_report:
            return "No previous test cases or steps."

        previous_case_result: TestCaseResult | None = None
        for test_case_result in test_report.test_cases:
            if test_case_result.case_id == test_case.case_id:
                break
            previous_case_result = test_case_result

        if not previous_case_result:
            return "No previous test cases or steps."

        last_step_definition: TestStep | None = None
        if current_test_plan:
            for candidate_case in current_test_plan.test_cases:
                if candidate_case.case_id == previous_case_result.case_id:
                    if candidate_case.steps:
                        last_step_definition = candidate_case.steps[-1]
                    break

        last_result = (
            previous_case_result.step_results[-1]
            if previous_case_result.step_results
            else None
        )
        if last_step_definition:
            return (
                f"Previous test case '{previous_case_result.name}' ended on "
                f"{self._format_step_summary(last_step_definition, last_result)}"
            )

        status_value = (
            previous_case_result.status.value.upper()
            if isinstance(previous_case_result.status, TestStatus)
            else str(previous_case_result.status)
        )
        return (
            f"Previous test case '{previous_case_result.name}' completed "
            f"with overall status {status_value}."
        )
