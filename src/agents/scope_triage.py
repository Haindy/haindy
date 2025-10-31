"""Scope Triage Agent implementation.

Runs before the Test Planner to normalize scope, exclusions, ambiguities,
and blocking questions from mixed requirement bundles.
"""

import re
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import SCOPE_TRIAGE_SYSTEM_PROMPT
from src.core.types import ScopeTriageResult
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class ScopeTriageAgent(BaseAgent):
    """Agent responsible for extracting scoped testing inputs from raw requirements."""

    def __init__(self, name: str = "ScopeTriage", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.system_prompt = SCOPE_TRIAGE_SYSTEM_PROMPT
        # Lower temperature for deterministic extraction
        self.temperature = kwargs.get("temperature", 0.15)

    async def triage_scope(
        self,
        requirements: str,
        context: Optional[Dict[str, str]] = None,
    ) -> ScopeTriageResult:
        """
        Analyze the requirements bundle and extract a curated testing scope.

        Args:
            requirements: Raw requirements text, PRD, or instructions
            context: Optional additional context (environment, credentials, etc.)

        Returns:
            ScopeTriageResult: Structured scope summary
        """
        logger.info(
            "Running scope triage",
            extra={"requirements_length": len(requirements), "has_context": context is not None},
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_triage_message(requirements, context)},
        ]

        response = await self.call_openai(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )

        triage_result = self._parse_triage_response(response)
        triage_result = self._post_process_result(triage_result, requirements)

        logger.info(
            "Scope triage completed",
            extra={
                "has_exclusions": bool(triage_result.explicit_exclusions),
                "ambiguous_count": len(triage_result.ambiguous_points),
                "blocking_count": len(triage_result.blocking_questions),
            },
        )

        return triage_result

    def _build_triage_message(
        self,
        requirements: str,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        """Construct the user message for the triage pass."""
        message_lines: List[str] = [
            "Review the following requirements bundle and extract the testing scope.",
            "",
            "Return a JSON object that matches the keys requested.",
            "Honor any explicit URLs, credentials, or scope instructions without re-asking for them.",
            "Assume testers can create or modify data as needed unless the requirements forbid it.",
            "Treat items marked TBD as notes rather than blockers unless they directly prevent execution.",
            "",
            "REQUIREMENTS PACKAGE:",
            requirements,
        ]

        if context:
            message_lines.append("")
            message_lines.append("ADDITIONAL CONTEXT:")
            for key, value in context.items():
                message_lines.append(f"- {key}: {value}")

        return "\n".join(message_lines)

    def _parse_triage_response(self, response: Dict) -> ScopeTriageResult:
        """Parse the JSON response into a ScopeTriageResult."""
        import json

        content = response.get("content", {})
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Scope triage response was not valid JSON: {exc}") from exc
        elif isinstance(content, dict):
            payload = content
        else:
            raise ValueError("Scope triage response missing JSON payload.")

        def _ensure_list(value: Optional[object]) -> List[str]:
            if not value:
                return []
            if isinstance(value, list):
                return [str(item) for item in value if str(item).strip()]
            return [str(value)]

        in_scope_value = payload.get("in_scope")
        if isinstance(in_scope_value, list):
            in_scope = "\n".join(str(item).strip() for item in in_scope_value if str(item).strip())
        else:
            in_scope = str(in_scope_value) if in_scope_value is not None else ""
        explicit_exclusions = _ensure_list(payload.get("explicit_exclusions"))
        ambiguous_points = _ensure_list(payload.get("ambiguous_points"))
        blocking_questions = _ensure_list(payload.get("blocking_questions"))

        return ScopeTriageResult(
            in_scope=in_scope,
            explicit_exclusions=explicit_exclusions,
            ambiguous_points=ambiguous_points,
            blocking_questions=blocking_questions,
        )

    def _post_process_result(
        self,
        result: ScopeTriageResult,
        requirements: str,
    ) -> ScopeTriageResult:
        """Apply heuristics so obvious instructions aren't re-flagged as blockers."""
        requirements_lower = requirements.lower()
        urls = re.findall(r"https?://[^\s)]+", requirements_lower)

        if urls:
            result.blocking_questions = [
                q
                for q in result.blocking_questions
                if not re.search(r"\b(url|host|login|path)\b", q.lower())
            ]

        # Drop ambiguity notes that simply speculate about seeded data.
        cleaned_ambiguities: List[str] = []
        for point in result.ambiguous_points:
            lowered = point.lower()
            if "seed" in lowered:
                continue
            cleaned_ambiguities.append(point)
        result.ambiguous_points = cleaned_ambiguities

        return result
