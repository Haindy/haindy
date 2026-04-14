"""Awareness Agent for tool-call explore mode."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from haindy.agents.base_agent import BaseAgent
from haindy.agents.structured_output_schemas import (
    AWARENESS_ASSESSMENT_RESPONSE_FORMAT,
)
from haindy.config.agent_prompts import AWARENESS_AGENT_SYSTEM_PROMPT
from haindy.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AwarenessTodoItem:
    """One item in the explore TODO list."""

    action: str
    status: str


@dataclass(frozen=True)
class AwarenessAssessment:
    """Structured result from one awareness pass."""

    decision: str
    response: str
    current_focus: str | None
    todo: list[AwarenessTodoItem]
    observations: list[str]


class AwarenessAgent(BaseAgent):
    """Perceive the current UI and decide the next explore action."""

    def __init__(self, name: str = "AwarenessAgent", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.system_prompt = AWARENESS_AGENT_SYSTEM_PROMPT
        self.temperature = kwargs.get("temperature", 0.2)

    async def bootstrap(
        self,
        *,
        goal: str,
        screenshot: bytes,
        context: dict[str, Any] | None = None,
    ) -> AwarenessAssessment:
        """Build the first explore assessment from the initial screenshot."""

        return await self._assess(
            goal=goal,
            screenshot=screenshot,
            todo=[],
            observations=[],
            last_action_summary=None,
            context=context,
        )

    async def assess(
        self,
        *,
        goal: str,
        screenshot: bytes,
        todo: list[AwarenessTodoItem],
        observations: list[str],
        last_action_summary: str | None,
        context: dict[str, Any] | None = None,
    ) -> AwarenessAssessment:
        """Update the explore plan after one action."""

        return await self._assess(
            goal=goal,
            screenshot=screenshot,
            todo=todo,
            observations=observations,
            last_action_summary=last_action_summary,
            context=context,
        )

    async def _assess(
        self,
        *,
        goal: str,
        screenshot: bytes,
        todo: list[AwarenessTodoItem],
        observations: list[str],
        last_action_summary: str | None,
        context: dict[str, Any] | None,
    ) -> AwarenessAssessment:
        screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
        prompt = self._build_prompt(
            goal=goal,
            todo=todo,
            observations=observations,
            last_action_summary=last_action_summary,
            context=context,
        )

        logger.info(
            "Running awareness assessment",
            extra={
                "goal_length": len(goal),
                "todo_items": len(todo),
                "observations": len(observations),
            },
        )

        response = await self.call_model(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    ],
                }
            ],
            response_format=AWARENESS_ASSESSMENT_RESPONSE_FORMAT,
            log_agent="awareness_agent.assess",
            log_metadata={"goal": goal},
        )
        return self._parse_response(response)

    def _build_prompt(
        self,
        *,
        goal: str,
        todo: list[AwarenessTodoItem],
        observations: list[str],
        last_action_summary: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        todo_lines = (
            "\n".join(f"- [{item.status}] {item.action}" for item in todo)
            if todo
            else "- none yet"
        )
        observation_lines = (
            "\n".join(f"- {item}" for item in observations)
            if observations
            else "- none"
        )
        context_lines = (
            "\n".join(f"- {key}: {value}" for key, value in context.items())
            if context
            else "- none"
        )
        last_action = last_action_summary or "No previous action has been executed yet."

        return f"""Goal:
{goal}

Current TODO list:
{todo_lines}

Current observations:
{observation_lines}

Last action result:
{last_action}

Execution context:
{context_lines}

Return a JSON object with:
- decision: one of continue, goal_reached, stuck, aborted
- response: a concise natural-language summary for the caller
- current_focus: what the agent is currently trying to do, or null if terminal
- todo: the full updated TODO list
- observations: the full updated observations list

Decision rules:
- continue: there is at least one plausible next UI action
- goal_reached: the screenshot shows the goal is satisfied
- stuck: the visible UI leaves no reasonable next step to make progress
- aborted: the screenshot shows an external interruption or loss of control (wrong app in focus, launcher, emulator restart, unrelated system screen)
"""

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> AwarenessAssessment:
        content = response.get("content", {})
        if isinstance(content, str):
            payload = json.loads(content)
        elif isinstance(content, dict):
            payload = content
        else:
            raise ValueError("Awareness assessment response missing JSON payload.")

        todo: list[AwarenessTodoItem] = []
        for item in payload.get("todo", []):
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip()
            status = str(item.get("status") or "").strip()
            if not action or not status:
                continue
            todo.append(AwarenessTodoItem(action=action, status=status))

        return AwarenessAssessment(
            decision=str(payload.get("decision") or "stuck").strip(),
            response=str(payload.get("response") or "Exploration ended.").strip(),
            current_focus=(
                str(payload.get("current_focus")).strip()
                if payload.get("current_focus") is not None
                else None
            ),
            todo=todo,
            observations=[
                str(item).strip()
                for item in payload.get("observations", [])
                if str(item).strip()
            ],
        )
