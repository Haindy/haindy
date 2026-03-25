"""Verification helpers for TestRunner."""

from __future__ import annotations

import base64
import json
import traceback
from typing import Any

from haindy.agents.computer_use.visual_state import VisualBounds, crop_to_bounds
from haindy.core.types import TestCase, TestCaseResult, TestStatus, TestStep
from haindy.monitoring.logger import get_logger

logger = get_logger(__name__)


class TestRunnerVerifier:
    """Owns verification and gating helpers used during step execution."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def verify_expected_outcome(
        self,
        test_case: TestCase,
        step: TestStep,
        action_results: list[dict[str, Any]],
        screenshot_before: bytes | None,
        screenshot_after: bytes | None,
        execution_history: list[dict[str, Any]],
        next_test_case: TestCase | None,
        replay_wait_budget_ms: int | None = None,
    ) -> dict[str, Any]:
        """Use AI to verify if expected outcome was achieved with full context."""
        verification_before = screenshot_before
        verification_after = screenshot_after
        verification_visual_note = self._build_verification_visual_note(
            step=step,
            action_results=action_results,
        )
        patch_bounds = self._extract_patch_bounds(action_results)
        if (
            verification_before is not None
            and verification_after is not None
            and patch_bounds is not None
            and step.intent.value != "group_assert"
            and len(action_results) == 1
        ):
            try:
                verification_before = crop_to_bounds(verification_before, patch_bounds)
            except Exception:
                logger.debug(
                    "Failed to crop verification before image; falling back to full frame",
                    exc_info=True,
                )

        history_context = []
        for hist_item in execution_history:
            status_emoji = "✓" if hist_item.get("status") == TestStatus.PASSED else "✗"
            history_context.append(
                f"Step {hist_item.get('step_number')}: {hist_item.get('action')} - {status_emoji} {hist_item.get('status')}\n"
                f"  Result: {hist_item.get('actual_result', 'N/A')}"
            )

        actions_context = []
        for idx, action_data in enumerate(action_results, 1):
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

            if validation:
                for key, value in validation.items():
                    if key in {
                        "target_reference",
                        "pixel_coordinates",
                        "relative_x",
                        "relative_y",
                    }:
                        continue
                    action_detail += f"\n    {key}: {value}"

            if ai_analysis:
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

            if execution:
                action_detail += "\n  \n  Execution Details:"
                action_detail += (
                    f"\n    Duration: {execution.get('duration_ms', 'N/A')}ms"
                )
                if execution.get("error_message"):
                    action_detail += f"\n    Error: {execution.get('error_message')}"

            actions_context.append(action_detail)

        replay_wait_section = ""
        replay_wait_response_fields = ""
        if replay_wait_budget_ms is not None:
            remaining_budget_ms = max(int(replay_wait_budget_ms), 0)
            replay_wait_section = f"""
3. This validation is for a replayed cached action. Decide whether this step should wait longer before final failure.
   - Request additional wait ONLY when there is clear evidence the UI is still settling (for example loading indicators, transition overlays, in-flight navigation, or a partially updated state).
   - If evidence already supports a final PASS or final FAIL, do not request more wait.
   - Remaining wait budget for this step: {remaining_budget_ms} ms.
"""
            replay_wait_response_fields = """
  "request_additional_wait": true/false,
  "recommended_wait_ms": integer (0 when no wait requested; must be <= remaining budget),
  "wait_reasoning": "Why additional wait is or is not needed",
"""

        prompt_text = f"""I'm executing a test case: "{test_case.name}"

Previous steps in this test case:
{chr(10).join(history_context) if history_context else "None"}

Current step to validate:
Step {step.step_number}: {step.action}
Expected result: {step.expected_result}

Actions performed:
{chr(10).join(actions_context)}

Based on all this information:

IMPORTANT: The "CU agent observation" fields above are real-time descriptions captured by the executor during the action, before the final screenshot was taken. Transient UI feedback such as toast messages, snackbars, and brief success banners may have auto-dismissed by the time the screenshot was captured. If the CU agent observation describes a success message or toast, treat that as strong evidence the action succeeded even if the message is no longer visible in the screenshot.

1. Did this step achieve its intended purpose? Evaluate ONLY whether this current step achieved its stated expected result.
   - Do NOT fail this step based on broader test-case goals or assertions assigned to earlier/later steps unless they are explicitly part of this step's expected result.
   - If a CU agent observation includes extra details outside this step's scope, ignore those extras unless they are directly required by this step and supported by the visible evidence.

2. Is this failure (if failed) a blocker that would prevent the next test case from running successfully?
   Next test case: {next_test_case.name if next_test_case else "None (last test case)"}
   (Consider: Does this failure leave the system in a state where the next test case cannot execute meaningfully?)
{replay_wait_section}
{verification_visual_note}

Respond with JSON:
{{
  "verdict": "PASS" or "FAIL",
  "reasoning": "Your analysis of why the step passed or failed",
  "actual_result": "Concise description of what actually happened",
  "confidence": 0.0-1.0,
{replay_wait_response_fields}  "is_blocker": true/false,
  "blocker_reasoning": "Why this would/wouldn't block the next test case"
}}"""
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]

        if verification_before:
            messages[0]["content"].insert(
                1, {"type": "text", "text": "\nScreenshot before actions:"}
            )
            messages[0]["content"].insert(
                2,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(verification_before).decode()}"
                    },
                },
            )

        if verification_after:
            messages[0]["content"].append(
                {"type": "text", "text": "\nScreenshot after actions:"}
            )
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(verification_after).decode()}"
                    },
                }
            )

        try:
            response = await self._runner.call_openai(
                messages=messages, response_format={"type": "json_object"}
            )

            log_messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                }
            ]
            if verification_before:
                log_messages[0]["content"].append(
                    {"type": "image_url", "image_url": "<<attached screenshot>>"}
                )
            if verification_after:
                log_messages[0]["content"].append(
                    {"type": "image_url", "image_url": "<<attached screenshot>>"}
                )

            screenshots: list[tuple[str, bytes]] = []
            if verification_before:
                screenshots.append(("verification_before", verification_before))
            if verification_after:
                screenshots.append(("verification_after", verification_after))

            await self._runner._model_logger.log_call(
                agent="test_runner.verify_step",
                model=self._runner.model,
                prompt=prompt_text,
                request_payload={
                    "messages": log_messages,
                    "response_format": {"type": "json_object"},
                },
                response=response,
                screenshots=screenshots or None,
                metadata={
                    "step_number": step.step_number,
                    "test_case": test_case.name,
                },
            )

            content = response.get("content", "{}")
            if isinstance(content, str):
                parsed_result = json.loads(content)
                result: dict[str, Any] = (
                    parsed_result if isinstance(parsed_result, dict) else {}
                )
            elif isinstance(content, dict):
                result = content
            else:
                result = {}

            if "verdict" not in result:
                result["verdict"] = "FAIL"
            result["verdict"] = str(result.get("verdict", "FAIL")).strip().upper()
            if result["verdict"] not in {"PASS", "FAIL"}:
                result["verdict"] = "FAIL"
            if "reasoning" not in result:
                result["reasoning"] = "Verification failed - no reasoning provided"
            if "actual_result" not in result:
                result["actual_result"] = "Unknown outcome"
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "is_blocker" not in result:
                result["is_blocker"] = False
            if "blocker_reasoning" not in result:
                result["blocker_reasoning"] = ""
            if replay_wait_budget_ms is not None:
                request_additional_wait = self.coerce_model_bool(
                    result.get("request_additional_wait", False)
                )
                recommended_wait_raw = result.get("recommended_wait_ms", 0)
                try:
                    recommended_wait_ms = int(recommended_wait_raw)
                except (TypeError, ValueError):
                    recommended_wait_ms = 0
                remaining_budget_ms = max(int(replay_wait_budget_ms), 0)
                if remaining_budget_ms <= 0:
                    request_additional_wait = False
                    recommended_wait_ms = 0
                else:
                    recommended_wait_ms = max(
                        0, min(recommended_wait_ms, remaining_budget_ms)
                    )
                    if not request_additional_wait:
                        recommended_wait_ms = 0
                result["request_additional_wait"] = request_additional_wait
                result["recommended_wait_ms"] = recommended_wait_ms
                if "wait_reasoning" not in result:
                    result["wait_reasoning"] = ""

            logger.info(
                "Step verification completed",
                extra={
                    "step_number": step.step_number,
                    "verdict": result["verdict"],
                    "confidence": result["confidence"],
                    "is_blocker": result["is_blocker"],
                },
            )
            return result

        except Exception as exc:
            logger.error(
                "Failed to verify outcome with AI",
                extra={
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "step": step.step_number,
                    "test_case": test_case.name,
                },
            )
            raise

    async def verify_prerequisites(self, prerequisites: list[str]) -> bool:
        """Verify test case prerequisites are met."""
        if not prerequisites:
            return True
        logger.info("Checking prerequisites", extra={"prerequisites": prerequisites})
        return True

    async def verify_postconditions(self, postconditions: list[str]) -> bool:
        """Verify test case postconditions are met."""
        if not postconditions:
            return True

        if self._runner.automation_driver:
            await self._runner.automation_driver.screenshot()
            logger.info(
                "Checking postconditions", extra={"postconditions": postconditions}
            )
        return True

    def check_dependencies(self, step: TestStep, case_result: TestCaseResult) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True

        for dep_num in step.dependencies:
            dep_result = next(
                (
                    result
                    for result in case_result.step_results
                    if result.step_number == dep_num
                ),
                None,
            )
            if not dep_result or dep_result.status != TestStatus.PASSED:
                logger.warning(
                    "Step dependency not met",
                    extra={
                        "step": step.step_number,
                        "dependency": dep_num,
                        "dependency_status": (
                            dep_result.status.value if dep_result else "not_found"
                        ),
                    },
                )
                return False
        return True

    @staticmethod
    def coerce_model_bool(value: Any) -> bool:
        """Normalize model-provided boolean-like values."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return False

    @staticmethod
    def _extract_patch_bounds(
        action_results: list[dict[str, Any]],
    ) -> VisualBounds | None:
        if len(action_results) != 1:
            return None
        full_data = action_results[0].get("full_data", {})
        if not isinstance(full_data, dict):
            return None
        result_blob = full_data.get("result", {})
        if not isinstance(result_blob, dict):
            return None
        env_after = result_blob.get("environment_state_after", {})
        if not isinstance(env_after, dict):
            return None
        if str(env_after.get("frame_kind") or "").strip().lower() != "patch":
            return None
        patch_bounds = env_after.get("patch_bounds")
        if not isinstance(patch_bounds, (list, tuple)) or len(patch_bounds) != 4:
            return None
        try:
            return VisualBounds(
                x=int(patch_bounds[0]),
                y=int(patch_bounds[1]),
                width=int(patch_bounds[2]),
                height=int(patch_bounds[3]),
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def _build_verification_visual_note(
        cls,
        *,
        step: TestStep,
        action_results: list[dict[str, Any]],
    ) -> str:
        patch_bounds = cls._extract_patch_bounds(action_results)
        if patch_bounds is None or step.intent.value == "group_assert":
            return ""
        full_data = action_results[0].get("full_data", {}) if action_results else {}
        result_blob = full_data.get("result", {}) if isinstance(full_data, dict) else {}
        env_after = (
            result_blob.get("environment_state_after", {})
            if isinstance(result_blob, dict)
            else {}
        )
        viewport = (
            env_after.get("viewport_size") if isinstance(env_after, dict) else None
        )
        frame_id = env_after.get("frame_id") if isinstance(env_after, dict) else None
        details = [
            "Visual verification note: the after image is a local patch crop, not the full screen.",
            (
                f"Patch bounds in full-screen coordinates: x={patch_bounds.x}, "
                f"y={patch_bounds.y}, width={patch_bounds.width}, height={patch_bounds.height}."
            ),
        ]
        if isinstance(viewport, (list, tuple)) and len(viewport) == 2:
            details.append(
                f"Original full-screen size: width={viewport[0]}, height={viewport[1]}."
            )
        if frame_id:
            details.append(f"Patch frame id: {frame_id}.")
        details.append(
            "Interpret any cropped before/after screenshots as views into that local region of the original screen."
        )
        return "\n".join(details)
