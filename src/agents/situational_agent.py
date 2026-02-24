"""Situational setup agent for desktop-first execution."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import SITUATIONAL_SYSTEM_PROMPT
from src.core.interfaces import AutomationDriver
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SetupInstructions:
    """Structured setup instructions derived from context."""

    web_url: str = ""
    app_name: str = ""
    launch_command: str = ""
    maximize: bool = True


@dataclass
class SituationalAssessment:
    """Result of context adequacy and setup parsing."""

    target_type: str = "desktop_app"
    sufficient: bool = False
    missing_items: list[str] = field(default_factory=list)
    setup: SetupInstructions = field(default_factory=SetupInstructions)
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
        assessment = self._parse_assessment(payload, context_text)
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
    ) -> None:
        await automation_driver.start()

        setup = assessment.setup
        if setup.launch_command:
            await self._run_shell_command(setup.launch_command)
            await asyncio.sleep(0.75)

        if setup.app_name:
            await self._focus_window(setup.app_name)
            await asyncio.sleep(0.2)

        if setup.maximize:
            await self._maximize_active_window()
            await asyncio.sleep(0.2)

        if assessment.target_type == "web" and setup.web_url:
            await automation_driver.navigate(setup.web_url)

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
        context_text: str,
    ) -> SituationalAssessment:
        if not payload:
            return self._heuristic_assessment(context_text)

        target_type = str(payload.get("target_type") or "").strip().lower()
        if target_type not in {"web", "desktop_app"}:
            target_type = "web" if self._extract_url(context_text) else "desktop_app"

        setup_payload = payload.get("setup", {}) if isinstance(payload.get("setup"), dict) else {}
        setup = SetupInstructions(
            web_url=str(setup_payload.get("web_url") or "").strip(),
            app_name=str(setup_payload.get("app_name") or "").strip(),
            launch_command=str(setup_payload.get("launch_command") or "").strip(),
            maximize=self._normalize_bool(setup_payload.get("maximize"), default=True),
        )
        missing_items = self._ensure_string_list(payload.get("missing_items"))
        notes = self._ensure_string_list(payload.get("notes"))
        sufficient = bool(payload.get("sufficient"))

        if target_type == "web" and not setup.web_url:
            extracted_url = self._extract_url(context_text)
            if extracted_url:
                setup.web_url = extracted_url
            else:
                missing_items.append("web_url")
                sufficient = False

        if target_type == "desktop_app" and not (setup.launch_command or setup.app_name):
            guessed = self._extract_app_name(context_text)
            if guessed:
                setup.app_name = guessed
            else:
                missing_items.append("desktop application name or launch command")
                sufficient = False

        missing_items = sorted({item.strip() for item in missing_items if item.strip()})
        if missing_items:
            sufficient = False

        return SituationalAssessment(
            target_type=target_type,
            sufficient=sufficient,
            missing_items=missing_items,
            setup=setup,
            notes=notes,
        )

    def _heuristic_assessment(self, context_text: str) -> SituationalAssessment:
        target_type = "web" if self._extract_url(context_text) else "desktop_app"
        setup = SetupInstructions(
            web_url=self._extract_url(context_text) or "",
            app_name=self._extract_app_name(context_text) or "",
            launch_command=self._extract_launch_command(context_text) or "",
            maximize=not self._contains_no_maximize_instruction(context_text),
        )
        missing_items: list[str] = []
        if target_type == "web" and not setup.web_url:
            missing_items.append("web_url")
        if target_type == "desktop_app" and not (setup.launch_command or setup.app_name):
            missing_items.append("desktop application name or launch command")
        return SituationalAssessment(
            target_type=target_type,
            sufficient=not missing_items,
            missing_items=missing_items,
            setup=setup,
            notes=[],
        )

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

    async def _run_shell_command(self, command: str) -> None:
        args = shlex.split(command)
        if not args:
            return
        process = await asyncio.create_subprocess_exec(*args)
        await process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Launch command failed ({process.returncode}): {command}")

    async def _focus_window(self, app_name: str) -> None:
        await self._run_best_effort(
            ["xdotool", "search", "--name", app_name, "windowactivate", "--sync"]
        )
        await self._run_best_effort(["wmctrl", "-a", app_name])

    async def _maximize_active_window(self) -> None:
        await self._run_best_effort(
            [
                "wmctrl",
                "-r",
                ":ACTIVE:",
                "-b",
                "add,maximized_vert,maximized_horz",
            ]
        )
        await self._run_best_effort(
            ["xdotool", "getactivewindow", "windowsize", "100%", "100%"]
        )

    async def _run_best_effort(self, cmd: list[str]) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.communicate()
        except FileNotFoundError:
            return
