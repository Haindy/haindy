"""
Unit tests for the Scope Triage Agent.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.scope_triage import ScopeTriageAgent
from src.core.types import ScopeTriageResult


class TestScopeTriageAgent:
    """Tests for ScopeTriageAgent behavior."""

    @pytest.fixture
    def agent(self) -> ScopeTriageAgent:
        """Create a triage agent with a mocked OpenAI client."""
        agent = ScopeTriageAgent()
        agent._client = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_triage_scope_parses_response(self, agent: ScopeTriageAgent) -> None:
        """Ensure triage_scope returns a structured ScopeTriageResult."""
        mock_payload = {
            "in_scope": "Only test the admin portal bundle creation flows.",
            "explicit_exclusions": ["Do not touch the FMC frontend."],
            "ambiguous_points": ["Clarify whether translation workflows must be validated."],
            "blocking_questions": [],
        }

        agent.call_openai = AsyncMock(return_value={"content": json.dumps(mock_payload)})

        result = await agent.triage_scope("Test requirements text")

        assert isinstance(result, ScopeTriageResult)
        assert result.in_scope.startswith("Only test the admin portal")
        assert result.explicit_exclusions == ["Do not touch the FMC frontend."]
        assert len(result.ambiguous_points) == 1
        assert not result.has_blockers()

    @pytest.mark.asyncio
    async def test_triage_scope_handles_scalar_arrays(self, agent: ScopeTriageAgent) -> None:
        """Lists should be coerced even when the model returns scalars."""
        mock_payload = {
            "in_scope": "Full scope permitted.",
            "explicit_exclusions": "",
            "ambiguous_points": "No ambiguities provided.",
            "blocking_questions": "Need staging URL before executing.",
        }

        agent.call_openai = AsyncMock(return_value={"content": json.dumps(mock_payload)})

        result = await agent.triage_scope("requirements")

        assert result.explicit_exclusions == []
        assert result.ambiguous_points == ["No ambiguities provided."]
        assert result.blocking_questions == ["Need staging URL before executing."]
        assert result.has_blockers()
        assert "BLOCKING QUESTIONS" in result.blocking_report()
