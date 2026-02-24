"""Situational setup agent for desktop-first execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import SITUATIONAL_SYSTEM_PROMPT
from src.core.interfaces import AutomationDriver
from src.core.types import ActionInstruction, ActionType, StepIntent, TestStep
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.agents.action_agent import ActionAgent


@dataclass
class SetupInstructions:
    """Structured setup instructions derived from context."""

    web_url: str = ""
    app_name: str = ""
    launch_command: str = ""
    maximize: bool = True


@dataclass
class EntrypointAction:
    """One visual entrypoint action to execute via Action Agent."""

    action_type: ActionType = ActionType.CLICK
    description: str = ""
    target: str | None = None
    value: str | None = None
    expected_outcome: str = "Entrypoint action completed."
    computer_use_prompt: str | None = None


@dataclass
class SituationalAssessment:
    """Result of context adequacy and setup parsing."""

    target_type: str = "desktop_app"
    sufficient: bool = False
    missing_items: list[str] = field(default_factory=list)
    setup: SetupInstructions = field(default_factory=SetupInstructions)
    entry_actions: list[EntrypointAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_blocking_questions(self) -> list[str]:
        if self.missing_items:
            return [f"Missing required context: {item}" for item in self.missing_items]
        return ["Context file is insufficient for entrypoint setup."]


class SituationalAgent(BaseAgent):
    """Determines if context is executable and prepares the entrypoint state."""

    def __init__(self, name: str = "SituationalAgent", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.system_prompt = SITUATIONAL_SYSTEM_PROMPT

    async def assess_context(
        self,
        requirements: str,
        context_text: str,
    ) -> SituationalAssessment:
        payload = await self._call_model_assessment(requirements, context_text)
        assessment = self._parse_assessment(payload, requirements, context_text)
        if not assessment.sufficient:
            logger.warning(
                "Situational assessment rejected context",
                extra={"missing_items": assessment.missing_items},
            )
        return assessment

    async def prepare_entrypoint(
        self,
        automation_driver: AutomationDriver,
        assessment: SituationalAssessment,
        action_agent: ActionAgent | None = None,
    ) -> None:
        await automation_driver.start()

        if action_agent is None:
            raise RuntimeError(
                "Situational entrypoint setup requires ActionAgent for visual execution."
            )

        entry_actions = assessment.entry_actions or self._derive_default_entry_actions(
            target_type=assessment.target_type,
            setup=assessment.setup,
            source_text="",
        )
        if not entry_actions:
            logger.info("No situational entrypoint actions were provided; continuing.")
            return

        for step_number, entry_action in enumerate(entry_actions, start=1):
            instruction = ActionInstruction(
                action_type=entry_action.action_type,
                description=entry_action.description,
                target=entry_action.target,
                value=entry_action.value,
                expected_outcome=entry_action.expected_outcome,
                computer_use_prompt=entry_action.computer_use_prompt,
            )
            test_step = TestStep(
                step_number=step_number,
                description=entry_action.description,
                action=entry_action.description,
                expected_result=entry_action.expected_outcome,
                action_instruction=instruction,
                intent=StepIntent.SETUP,
            )

            screenshot: bytes | None = None
            if entry_action.action_type != ActionType.SKIP_NAVIGATION:
                screenshot = await automation_driver.screenshot()

            result = await action_agent.execute_action(
                test_step=test_step,
                test_context={
                    "phase": "situational_entrypoint",
                    "target_type": assessment.target_type,
                    "step_number": step_number,
                },
                screenshot=screenshot,
                record_driver_actions=False,
            )
            if not result.overall_success:
                error_message = ""
                if result.execution and result.execution.error_message:
                    error_message = f" ({result.execution.error_message})"
                raise RuntimeError(
                    f"Entrypoint action failed at step {step_number}: "
                    f"{entry_action.description}{error_message}"
                )

    async def _call_model_assessment(
        self,
        requirements: str,
        context_text: str,
    ) -> dict[str, Any]:
        prompt = (
            "REQUIREMENTS:\n"
            f"{requirements.strip()}\n\n"
            "EXECUTION CONTEXT:\n"
            f"{context_text.strip()}\n"
        )
        try:
            response = await self.call_openai(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.get("content", {})
            if isinstance(content, str):
                return json.loads(content)
            if isinstance(content, dict):
                return content
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Situational assessment model call failed; using heuristic fallback",
                extra={"error": str(exc)},
            )
        return {}

    def _parse_assessment(
        self,
        payload: dict[str, Any],
        requirements: str,
        context_text: str,
    ) -> SituationalAssessment:
        source_text = f"{requirements.strip()}\n{context_text.strip()}".strip()

        if not payload:
            return self._heuristic_assessment(source_text)

        target_type = str(payload.get("target_type") or "").strip().lower()
        if target_type not in {"web", "desktop_app"}:
            target_type = "web" if self._extract_url(source_text) else "desktop_app"

        setup_payload = payload.get("setup", {}) if isinstance(payload.get("setup"), dict) else {}
        setup = SetupInstructions(
            web_url=str(setup_payload.get("web_url") or "").strip(),
            app_name=str(setup_payload.get("app_name") or "").strip(),
            launch_command=str(setup_payload.get("launch_command") or "").strip(),
            maximize=self._normalize_bool(setup_payload.get("maximize"), default=True),
        )
        missing_items = self._filter_non_visual_missing_items(
            self._ensure_string_list(payload.get("missing_items"))
        )
        notes = self._ensure_string_list(payload.get("notes"))
        entry_actions = self._parse_entry_actions(payload.get("entry_actions"))
        if not entry_actions:
            entry_actions = self._derive_default_entry_actions(
                target_type=target_type,
                setup=setup,
                source_text=source_text,
            )

        if target_type == "web" and not setup.web_url:
            extracted_url = self._extract_url(source_text)
            if extracted_url:
                setup.web_url = extracted_url
            else:
                missing_items.append("web_url")

        if target_type == "desktop_app" and not entry_actions:
            notes.append(
                "No explicit entry actions were provided; Action Agent will infer visually."
            )

        missing_items = sorted({item.strip() for item in missing_items if item.strip()})

        return SituationalAssessment(
            target_type=target_type,
            sufficient=not missing_items,
            missing_items=missing_items,
            setup=setup,
            entry_actions=entry_actions,
            notes=notes,
        )

    def _heuristic_assessment(self, source_text: str) -> SituationalAssessment:
        target_type = "web" if self._extract_url(source_text) else "desktop_app"
        setup = SetupInstructions(
            web_url=self._extract_url(source_text) or "",
            app_name=self._extract_app_name(source_text) or "",
            launch_command=self._extract_launch_command(source_text) or "",
            maximize=not self._contains_no_maximize_instruction(source_text),
        )
        entry_actions = self._derive_default_entry_actions(
            target_type=target_type,
            setup=setup,
            source_text=source_text,
        )
        missing_items: list[str] = []
        if target_type == "web" and not setup.web_url:
            missing_items.append("web_url")
        return SituationalAssessment(
            target_type=target_type,
            sufficient=not missing_items,
            missing_items=missing_items,
            setup=setup,
            entry_actions=entry_actions,
            notes=[],
        )

    def _derive_default_entry_actions(
        self,
        target_type: str,
        setup: SetupInstructions,
        source_text: str,
    ) -> list[EntrypointAction]:
        if target_type == "web" and setup.web_url:
            return [
                EntrypointAction(
                    action_type=ActionType.NAVIGATE,
                    description=f"Open {setup.web_url}",
                    target=setup.web_url,
                    value=setup.web_url,
                    expected_outcome=f"{setup.web_url} is open and ready for testing.",
                    computer_use_prompt=(
                        f"Open {setup.web_url} in the active browser and stop when the page "
                        "is fully visible for testing."
                    ),
                )
            ]

        if target_type != "desktop_app":
            return []

        app_name = setup.app_name or self._extract_app_name(source_text) or "the target application"
        launch_hint = setup.launch_command or self._extract_launch_command(source_text)
        prompt_lines = [
            f"Open {app_name} and bring it to the foreground using visible desktop UI actions only.",
            "Do not use programmatic window manager controls or deterministic OS identifiers.",
            "Use launcher/search/dock navigation if needed.",
            "Stop when the app is visible and ready for the first test action.",
        ]
        if launch_hint:
            prompt_lines.insert(
                1,
                f"If a terminal path is required, use this launch hint: {launch_hint}.",
            )

        return [
            EntrypointAction(
                action_type=ActionType.CLICK,
                description=f"Bring {app_name} to the test-ready entrypoint",
                target=app_name,
                expected_outcome=f"{app_name} is visible and ready for testing.",
                computer_use_prompt=" ".join(prompt_lines),
            )
        ]

    @staticmethod
    def _coerce_action_type(raw: Any) -> ActionType:
        if isinstance(raw, ActionType):
            return raw
        normalized = str(raw or "").strip().lower()
        try:
            return ActionType(normalized)
        except ValueError:
            return ActionType.CLICK

    def _parse_entry_actions(self, raw: Any) -> list[EntrypointAction]:
        if not isinstance(raw, list):
            return []

        actions: list[EntrypointAction] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue

            action_type = self._coerce_action_type(item.get("action_type"))
            description = str(item.get("description") or "").strip()
            if not description:
                description = f"Entrypoint step {index}"

            expected_outcome = str(item.get("expected_outcome") or "").strip()
            if not expected_outcome:
                expected_outcome = "Entrypoint action completed."

            target_raw = item.get("target")
            target = str(target_raw).strip() if target_raw not in {None, ""} else None

            value_raw = item.get("value")
            value = str(value_raw).strip() if value_raw not in {None, ""} else None

            prompt_raw = item.get("computer_use_prompt")
            computer_use_prompt = (
                str(prompt_raw).strip() if prompt_raw not in {None, ""} else None
            )

            actions.append(
                EntrypointAction(
                    action_type=action_type,
                    description=description,
                    target=target,
                    value=value,
                    expected_outcome=expected_outcome,
                    computer_use_prompt=computer_use_prompt,
                )
            )
        return actions

    @staticmethod
    def _filter_non_visual_missing_items(missing_items: list[str]) -> list[str]:
        blocked_markers = (
            "window title",
            "task switcher",
            "wm_class",
            "process name",
            "deterministic",
            "wmctrl",
            "xdotool",
            "focus it",
        )
        filtered: list[str] = []
        for item in missing_items:
            lowered = item.lower()
            if any(marker in lowered for marker in blocked_markers):
                continue
            filtered.append(item)
        return filtered

    @staticmethod
    def _ensure_string_list(raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return [str(raw).strip()] if str(raw).strip() else []

    @staticmethod
    def _normalize_bool(value: Any, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"false", "0", "no", "off"}:
                return False
            if normalized in {"true", "1", "yes", "on"}:
                return True
        return default

    @staticmethod
    def _extract_url(text: str) -> str | None:
        match = re.search(r"https?://[^\s<>'\"()]+", text or "")
        if not match:
            return None
        candidate = match.group(0).rstrip(".,;:!?)\"']")
        parsed = urlparse(candidate)
        if not parsed.scheme or not parsed.netloc:
            return None
        return candidate

    @staticmethod
    def _extract_launch_command(text: str) -> str | None:
        patterns = [
            r"(?:launch[_\s-]*command|start[_\s-]*command)\s*[:=]\s*(.+)",
            r"(?:run|launch)\s+command\s*[:=]\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_app_name(text: str) -> str | None:
        patterns = [
            r"(?:app[_\s-]*name|application[_\s-]*name|window[_\s-]*name)\s*[:=]\s*(.+)",
            r"(?:focus|open)\s+([A-Za-z0-9._ -]{3,})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _contains_no_maximize_instruction(text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            marker in lowered
            for marker in (
                "do not maximize",
                "don't maximize",
                "without maximizing",
                "leave window size",
            )
        )
