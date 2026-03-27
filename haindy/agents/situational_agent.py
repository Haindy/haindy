"""Situational setup agent for desktop-first execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from haindy.agents.base_agent import BaseAgent
from haindy.agents.structured_output_schemas import (
    SITUATIONAL_ASSESSMENT_RESPONSE_FORMAT,
)
from haindy.config.agent_prompts import SITUATIONAL_SYSTEM_PROMPT
from haindy.config.settings import get_settings
from haindy.core.interfaces import AutomationDriver
from haindy.core.types import ActionInstruction, ActionType, StepIntent, TestStep
from haindy.monitoring.debug_logger import get_debug_logger
from haindy.monitoring.logger import get_logger
from haindy.runtime.situational_cache import (
    SituationalCache,
    build_situational_cache_key_payload,
    hash_situational_cache_key,
)
from haindy.utils.model_logging import ModelCallLogger, get_model_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from haindy.agents.action_agent import ActionAgent


def _prompt_fingerprint(prompt: str | None) -> str:
    return sha256((prompt or "").encode("utf-8")).hexdigest()


def _agent_signature(agent: Any) -> dict[str, Any]:
    modalities = getattr(agent, "modalities", None) or []
    return {
        "name": str(getattr(agent, "name", "")),
        "model": str(getattr(agent, "model", "")),
        "temperature": getattr(agent, "temperature", None),
        "reasoning_level": str(getattr(agent, "reasoning_level", "")),
        "modalities": sorted(str(item) for item in modalities),
        "system_prompt_sha256": _prompt_fingerprint(
            getattr(agent, "system_prompt", None)
        ),
    }


@dataclass
class SetupInstructions:
    """Structured setup instructions derived from context."""

    web_url: str = ""
    app_name: str = ""
    launch_command: str = ""
    maximize: bool = True
    adb_serial: str = ""
    app_package: str = ""
    app_activity: str = ""
    adb_commands: list[str] = field(default_factory=list)
    ios_udid: str = ""
    bundle_id: str = ""


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

    def __init__(
        self,
        name: str = "SituationalAgent",
        model_logger: ModelCallLogger | None = None,
        situational_cache: SituationalCache | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.system_prompt = SITUATIONAL_SYSTEM_PROMPT
        settings = get_settings()
        self._model_logger = model_logger or get_model_logger(
            settings.model_log_path,
            max_screenshots=getattr(settings, "max_screenshots", None),
        )
        self._situational_cache = self._resolve_situational_cache(
            settings,
            situational_cache=situational_cache,
        )

    @staticmethod
    def _resolve_situational_cache(
        settings: Any,
        situational_cache: SituationalCache | None = None,
    ) -> SituationalCache | None:
        if situational_cache is not None:
            return situational_cache
        if not settings.enable_situational_cache:
            return None
        return SituationalCache(settings.situational_cache_path)

    async def assess_context(
        self,
        requirements: str,
        context_text: str,
    ) -> SituationalAssessment:
        cache = self._situational_cache
        cache_key_hash: str | None = None

        if cache is not None:
            cache_payload = build_situational_cache_key_payload(
                requirements=requirements,
                context_text=context_text,
                situational_signature=_agent_signature(self),
            )
            cache_key_hash = hash_situational_cache_key(cache_payload)
            cached_entry = cache.lookup(cache_key_hash)
            if cached_entry is not None:
                try:
                    assessment = self._assessment_from_cache_payload(
                        cached_entry.assessment_payload
                    )
                    logger.info(
                        "Situational cache hit for execution context validation",
                        extra={
                            "cache_key_hash": cache_key_hash,
                            "sufficient": assessment.sufficient,
                        },
                    )
                    return assessment
                except Exception:
                    cache.invalidate(cache_key_hash)
                    logger.warning(
                        "Situational cache entry invalid; falling back to model execution",
                        exc_info=True,
                        extra={"cache_key_hash": cache_key_hash},
                    )

        payload = await self._call_model_assessment(requirements, context_text)
        assessment = self._parse_assessment(payload, requirements, context_text)

        if cache is not None and cache_key_hash is not None and payload:
            try:
                cache.store(
                    key_hash=cache_key_hash,
                    assessment_payload=self._assessment_to_cache_payload(assessment),
                )
            except Exception:
                logger.debug("Failed to store situational cache entry", exc_info=True)

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
        if assessment.target_type == "mobile_adb":
            configure_target = getattr(automation_driver, "configure_target", None)
            if callable(configure_target):
                await configure_target(
                    adb_serial=assessment.setup.adb_serial or None,
                    app_package=assessment.setup.app_package or None,
                    app_activity=assessment.setup.app_activity or None,
                )
            await automation_driver.start()
            await self._prepare_mobile_entrypoint(automation_driver, assessment.setup)
            return

        if assessment.target_type == "mobile_ios":
            configure_target = getattr(automation_driver, "configure_target", None)
            if callable(configure_target):
                await configure_target(
                    udid=assessment.setup.ios_udid or None,
                    bundle_id=assessment.setup.bundle_id or None,
                )
            await automation_driver.start()
            await self._prepare_ios_entrypoint(automation_driver, assessment.setup)
            return

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
                logger.warning(
                    "Entrypoint action failed; continuing with best-effort setup",
                    extra={
                        "step_number": step_number,
                        "description": entry_action.description,
                        "error": error_message or "unknown",
                    },
                )
                continue

    async def _prepare_mobile_entrypoint(
        self,
        automation_driver: AutomationDriver,
        setup: SetupInstructions,
    ) -> None:
        run_adb_commands = getattr(automation_driver, "run_adb_commands", None)
        if setup.adb_commands and callable(run_adb_commands):
            await run_adb_commands(setup.adb_commands)

        launch_app = getattr(automation_driver, "launch_app", None)
        if callable(launch_app) and setup.app_package and not setup.adb_commands:
            await launch_app(
                app_package=setup.app_package,
                app_activity=setup.app_activity or None,
            )

        await automation_driver.screenshot()

    async def _prepare_ios_entrypoint(
        self,
        automation_driver: AutomationDriver,
        setup: SetupInstructions,
    ) -> None:
        launch_app = getattr(automation_driver, "launch_app", None)
        if callable(launch_app) and setup.bundle_id:
            await launch_app(bundle_id=setup.bundle_id)

        await automation_driver.screenshot()

    @staticmethod
    def _assessment_to_cache_payload(
        assessment: SituationalAssessment,
    ) -> dict[str, Any]:
        return {
            "target_type": assessment.target_type,
            "sufficient": assessment.sufficient,
            "missing_items": list(assessment.missing_items),
            "setup": {
                "web_url": assessment.setup.web_url,
                "app_name": assessment.setup.app_name,
                "launch_command": assessment.setup.launch_command,
                "maximize": assessment.setup.maximize,
                "adb_serial": assessment.setup.adb_serial,
                "app_package": assessment.setup.app_package,
                "app_activity": assessment.setup.app_activity,
                "adb_commands": list(assessment.setup.adb_commands),
                "ios_udid": assessment.setup.ios_udid,
                "bundle_id": assessment.setup.bundle_id,
            },
            "entry_actions": [
                {
                    "action_type": action.action_type.value,
                    "description": action.description,
                    "target": action.target,
                    "value": action.value,
                    "expected_outcome": action.expected_outcome,
                    "computer_use_prompt": action.computer_use_prompt,
                }
                for action in assessment.entry_actions
            ],
            "notes": list(assessment.notes),
        }

    def _assessment_from_cache_payload(
        self, payload: dict[str, Any]
    ) -> SituationalAssessment:
        if not isinstance(payload, dict):
            raise ValueError("Assessment payload must be a dictionary.")

        target_type = str(payload.get("target_type") or "").strip().lower()
        if target_type not in {"web", "desktop_app", "mobile_adb", "mobile_ios"}:
            raise ValueError(f"Unsupported cached target_type: {target_type!r}")

        setup_payload = (
            payload.get("setup", {}) if isinstance(payload.get("setup"), dict) else {}
        )
        setup = SetupInstructions(
            web_url=str(setup_payload.get("web_url") or "").strip(),
            app_name=str(setup_payload.get("app_name") or "").strip(),
            launch_command=str(setup_payload.get("launch_command") or "").strip(),
            maximize=self._normalize_bool(setup_payload.get("maximize"), default=True),
            adb_serial=str(setup_payload.get("adb_serial") or "").strip(),
            app_package=str(setup_payload.get("app_package") or "").strip(),
            app_activity=str(setup_payload.get("app_activity") or "").strip(),
            adb_commands=self._ensure_command_list(setup_payload.get("adb_commands")),
            ios_udid=str(setup_payload.get("ios_udid") or "").strip(),
            bundle_id=str(setup_payload.get("bundle_id") or "").strip(),
        )

        return SituationalAssessment(
            target_type=target_type,
            sufficient=self._normalize_bool(payload.get("sufficient"), default=False),
            missing_items=self._ensure_string_list(payload.get("missing_items")),
            setup=setup,
            entry_actions=self._parse_entry_actions(payload.get("entry_actions")),
            notes=self._ensure_string_list(payload.get("notes")),
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
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response: dict[str, Any] | None = None
        parsed_payload: dict[str, Any] = {}
        error_message: str | None = None
        logged_exception = False
        try:
            response = await self.call_model(
                messages=messages,
                response_format=SITUATIONAL_ASSESSMENT_RESPONSE_FORMAT,
                temperature=0.1,
                log_agent="situational.assessment",
                log_metadata={"phase": "situational_assessment"},
            )
            content = response.get("content", {})
            if isinstance(content, str):
                parsed_payload = json.loads(content)
            elif isinstance(content, dict):
                parsed_payload = content
            else:
                error_message = (
                    "Situational assessment returned unsupported content type: "
                    f"{type(content).__name__}"
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            error_message = str(exc)
            logged_exception = True
            logger.warning(
                "Situational assessment model call failed; using heuristic fallback",
                extra={"error": str(exc)},
            )

        if error_message and not logged_exception:
            logger.warning(
                "Situational assessment model payload unusable; using heuristic fallback",
                extra={"error": error_message},
            )

        await self._log_assessment_interaction(
            prompt=prompt,
            response_payload=response,
            parsed_payload=parsed_payload,
            error_message=error_message,
        )
        return parsed_payload

    async def _log_assessment_interaction(
        self,
        *,
        prompt: str,
        response_payload: dict[str, Any] | None,
        parsed_payload: dict[str, Any],
        error_message: str | None,
    ) -> None:
        """Persist situational debug details for run forensics."""
        response_for_log: dict[str, Any]
        if response_payload is None:
            response_for_log = {"content": parsed_payload or {}, "error": error_message}
        else:
            response_for_log = response_payload

        debug_logger = get_debug_logger()
        if not debug_logger:
            return

        metadata: dict[str, Any] = {
            "phase": "situational_assessment",
            "fallback_used": bool(error_message),
            "error": error_message,
        }
        if parsed_payload.get("target_type"):
            metadata["target_type"] = parsed_payload.get("target_type")

        response_content = response_for_log.get("content", response_for_log)
        try:
            if isinstance(response_content, str):
                response_text = response_content
            else:
                response_text = json.dumps(response_content, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive string fallback
            response_text = str(response_content)

        if error_message:
            response_text = (
                f"{response_text}\n\nFallback note: {error_message}"
                if response_text
                else f"Fallback note: {error_message}"
            )

        try:
            debug_logger.log_ai_interaction(
                agent_name=self.name,
                action_type="situational_assessment",
                prompt=prompt,
                response=response_text,
                additional_context=metadata,
            )
        except Exception:  # pragma: no cover - logging should never be fatal
            logger.debug(
                "Failed to persist situational assessment debug interaction",
                exc_info=True,
            )

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
        if target_type not in {"web", "desktop_app", "mobile_adb", "mobile_ios"}:
            if self._extract_url(source_text):
                target_type = "web"
            elif self._looks_like_ios_context(source_text):
                target_type = "mobile_ios"
            elif self._looks_like_mobile_context(source_text):
                target_type = "mobile_adb"
            else:
                target_type = "desktop_app"

        setup_payload = (
            payload.get("setup", {}) if isinstance(payload.get("setup"), dict) else {}
        )
        setup = SetupInstructions(
            web_url=str(setup_payload.get("web_url") or "").strip(),
            app_name=str(setup_payload.get("app_name") or "").strip(),
            launch_command=str(setup_payload.get("launch_command") or "").strip(),
            maximize=self._normalize_bool(setup_payload.get("maximize"), default=True),
            adb_serial=str(setup_payload.get("adb_serial") or "").strip(),
            app_package=str(setup_payload.get("app_package") or "").strip(),
            app_activity=str(setup_payload.get("app_activity") or "").strip(),
            adb_commands=self._ensure_command_list(setup_payload.get("adb_commands")),
            ios_udid=str(setup_payload.get("ios_udid") or "").strip(),
            bundle_id=str(setup_payload.get("bundle_id") or "").strip(),
        )
        model_missing_items = self._filter_non_visual_missing_items(
            self._ensure_string_list(payload.get("missing_items"))
        )
        notes = self._ensure_string_list(payload.get("notes"))
        missing_items, notes = self._normalize_model_missing_items(
            target_type=target_type,
            missing_items=model_missing_items,
            notes=notes,
            context_text=context_text,
        )
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

        if target_type == "mobile_adb":
            if not setup.adb_serial:
                setup.adb_serial = self._extract_adb_serial(source_text) or ""
            if not setup.app_package:
                setup.app_package = self._extract_app_package(source_text) or ""
            if not setup.app_activity:
                setup.app_activity = self._extract_app_activity(source_text) or ""
            if not setup.adb_commands:
                setup.adb_commands = self._extract_adb_commands(source_text)

            has_structured_mobile_setup = bool(setup.adb_serial and setup.app_package)
            has_mobile_command_path = bool(setup.adb_commands)
            if not has_structured_mobile_setup and not has_mobile_command_path:
                missing_items.append(
                    "Provide adb_serial + app_package or adb_commands that discover the device and open the app"
                )
            elif not has_structured_mobile_setup:
                notes.append(
                    "Using adb_commands path for mobile entrypoint discovery/setup."
                )

        if target_type == "desktop_app" and not entry_actions:
            notes.append(
                "No explicit entry actions were provided; Action Agent will infer visually."
            )

        missing_items = sorted({item.strip() for item in missing_items if item.strip()})
        if target_type == "desktop_app" and missing_items:
            notes.extend(
                f"Non-blocking context gap for desktop entrypoint: {item}"
                for item in missing_items
            )
            missing_items = []

        return SituationalAssessment(
            target_type=target_type,
            sufficient=not missing_items,
            missing_items=missing_items,
            setup=setup,
            entry_actions=entry_actions,
            notes=notes,
        )

    def _heuristic_assessment(self, source_text: str) -> SituationalAssessment:
        if self._extract_url(source_text):
            target_type = "web"
        elif self._looks_like_ios_context(source_text):
            target_type = "mobile_ios"
        elif self._looks_like_mobile_context(source_text):
            target_type = "mobile_adb"
        else:
            target_type = "desktop_app"
        setup = SetupInstructions(
            web_url=self._extract_url(source_text) or "",
            app_name=self._extract_app_name(source_text) or "",
            launch_command=self._extract_launch_command(source_text) or "",
            maximize=not self._contains_no_maximize_instruction(source_text),
            adb_serial=self._extract_adb_serial(source_text) or "",
            app_package=self._extract_app_package(source_text) or "",
            app_activity=self._extract_app_activity(source_text) or "",
            adb_commands=self._extract_adb_commands(source_text),
        )
        entry_actions = self._derive_default_entry_actions(
            target_type=target_type,
            setup=setup,
            source_text=source_text,
        )
        missing_items: list[str] = []
        if target_type == "web" and not setup.web_url:
            missing_items.append("web_url")
        if target_type == "mobile_adb":
            has_structured_mobile_setup = bool(setup.adb_serial and setup.app_package)
            has_mobile_command_path = bool(setup.adb_commands)
            if not has_structured_mobile_setup and not has_mobile_command_path:
                missing_items.append(
                    "Provide adb_serial + app_package or adb_commands that discover the device and open the app"
                )
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

        if target_type == "mobile_adb":
            return []

        if target_type == "mobile_ios":
            # For iOS, use visual entry actions just like desktop/web.
            # If a URL is present navigate to it; otherwise open the home screen app.
            url = setup.web_url or self._extract_url(source_text)
            if url:
                return [
                    EntrypointAction(
                        action_type=ActionType.NAVIGATE,
                        description=f"Open Safari and navigate to {url}",
                        target=url,
                        value=url,
                        expected_outcome=f"{url} is open and visible in Safari.",
                        computer_use_prompt=(
                            f"Open Safari on the iOS device and navigate to {url}. "
                            "If Safari is not already open, tap its icon on the home screen. "
                            "Then tap the address bar, type the URL, and go. "
                            "Stop when the page is fully loaded and visible."
                        ),
                    )
                ]
            app_name = setup.app_name or (
                setup.bundle_id.split(".")[-1].capitalize()
                if setup.bundle_id
                else "the target app"
            )
            return [
                EntrypointAction(
                    action_type=ActionType.CLICK,
                    description=f"Open {app_name} on the iOS device",
                    target=app_name,
                    expected_outcome=f"{app_name} is open and ready for testing.",
                    computer_use_prompt=(
                        f"Open {app_name} on the iOS device. "
                        "Tap its icon on the home screen or in the dock. "
                        "Stop when the app is fully visible and ready for the first test action."
                    ),
                )
            ]

        if target_type != "desktop_app":
            return []

        app_name = (
            setup.app_name
            or self._extract_app_name(source_text)
            or "the target application"
        )
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

    def _normalize_model_missing_items(
        self,
        *,
        target_type: str,
        missing_items: list[str],
        notes: list[str],
        context_text: str,
    ) -> tuple[list[str], list[str]]:
        context_lower = (context_text or "").lower()
        has_out_of_scope_directive = "out of scope" in context_lower
        has_literal_value_directive = any(
            marker in context_lower
            for marker in (
                "use it literally",
                "take it literally",
                "take values literally",
            )
        )

        retained: list[str] = []
        augmented_notes = list(notes)
        for item in missing_items:
            lowered = item.lower()
            should_demote = False

            if self._is_speculative_missing_item(lowered):
                should_demote = True
            elif has_out_of_scope_directive and "out of scope" in lowered:
                should_demote = True
            elif has_literal_value_directive and any(
                token in lowered
                for token in (
                    "appears to be",
                    "not the",
                    "invalid",
                    "wrong",
                )
            ):
                should_demote = True
            elif not self._is_launch_critical_missing_item(
                target_type=target_type, item=lowered
            ):
                should_demote = True

            if should_demote:
                augmented_notes.append(f"Non-blocking context gap: {item}")
            else:
                retained.append(item)

        return retained, augmented_notes

    @staticmethod
    def _is_speculative_missing_item(item: str) -> bool:
        speculative_markers = (
            "clarification",
            "confirmation",
            "confirm ",
            "approval",
            "appears to be",
            "seems",
            "likely",
            "probably",
            "precondition requires",
        )
        return any(marker in item for marker in speculative_markers)

    @staticmethod
    def _is_launch_critical_missing_item(*, target_type: str, item: str) -> bool:
        if target_type == "mobile_adb":
            mobile_markers = (
                "adb_serial",
                "adb serial",
                "adb_commands",
                "adb command",
                "app_package",
                "app package",
                "package name",
                "device serial",
                "discover the device and open the app",
            )
            return any(marker in item for marker in mobile_markers)

        if target_type == "mobile_ios":
            ios_markers = (
                "ios_udid",
                "bundle_id",
                "bundle id",
                "udid",
            )
            return any(marker in item for marker in ios_markers)

        if target_type == "web":
            if "web_url" in item:
                return True
            if "url" in item and not any(
                marker in item for marker in ("deep link", "branch", "invitation")
            ):
                return True
            return False

        return False

    @staticmethod
    def _ensure_string_list(raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return [str(raw).strip()] if str(raw).strip() else []

    @staticmethod
    def _ensure_command_list(raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            commands = [str(item).strip() for item in raw if str(item).strip()]
            return [cmd for cmd in commands if cmd.startswith("adb")]
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("adb"):
                return [text]
        return []

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

    @staticmethod
    def _looks_like_mobile_context(text: str) -> bool:
        lowered = (text or "").lower()
        markers = (
            "android",
            "adb",
            "emulator",
            "mobile app",
            "physical device",
            "package:",
            "app_package",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _looks_like_ios_context(text: str) -> bool:
        lowered = (text or "").lower()
        markers = (
            "iphone",
            "ipad",
            "mobile_ios",
            "ios simulator",
            "xcode simulator",
            "idb",
            "bundle_id",
            "udid",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _extract_adb_serial(text: str) -> str | None:
        patterns = [
            r"(?:adb[_\s-]*serial|device[_\s-]*serial)\s*[:=]\s*([^\s]+)",
            r"\b(emulator-\d{4,5})\b",
            r"\b([A-Za-z0-9._:-]+)\s+device\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_app_package(text: str) -> str | None:
        patterns = [
            r"(?:app[_\s-]*package|package[_\s-]*name)\s*[:=]\s*([A-Za-z0-9_.]+)",
            r"\bpackage\s*[:=]\s*([A-Za-z0-9_.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_app_activity(text: str) -> str | None:
        patterns = [
            r"(?:app[_\s-]*activity|activity)\s*[:=]\s*([A-Za-z0-9_.$/]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_adb_commands(text: str) -> list[str]:
        commands: list[str] = []
        for line in (text or "").splitlines():
            candidate = line.strip()
            if candidate.startswith("adb "):
                commands.append(candidate)
        return commands
