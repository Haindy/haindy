"""Shared runtime agent-construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents import (
    ActionAgent,
    ScopeTriageAgent,
    SituationalAgent,
    TestPlannerAgent,
    TestRunner,
)
from src.config.settings import Settings, get_settings
from src.core.interfaces import AutomationDriver


@dataclass(frozen=True)
class PlanningAgentBundle:
    """Agents required for context assessment and planning."""

    scope_triage: ScopeTriageAgent
    test_planner: TestPlannerAgent
    situational_agent: SituationalAgent


@dataclass(frozen=True)
class RuntimeAgentBundle:
    """Full coordinator runtime agent set."""

    scope_triage: ScopeTriageAgent
    test_planner: TestPlannerAgent
    test_runner: TestRunner
    action_agent: ActionAgent
    situational_agent: SituationalAgent

    def as_dict(self) -> dict[str, Any]:
        """Expose the coordinator's canonical agent map."""
        return {
            "scope_triage": self.scope_triage,
            "test_planner": self.test_planner,
            "test_runner": self.test_runner,
            "action_agent": self.action_agent,
            "situational_agent": self.situational_agent,
        }


class AgentFactory:
    """Create configured agents from the shared settings model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create_planning_agents(self) -> PlanningAgentBundle:
        """Build the planning-only agent set."""
        return PlanningAgentBundle(
            scope_triage=ScopeTriageAgent(
                name="ScopeTriage",
                **self._agent_kwargs("scope_triage"),
            ),
            test_planner=TestPlannerAgent(
                name="TestPlanner",
                **self._agent_kwargs("test_planner"),
            ),
            situational_agent=SituationalAgent(
                name="SituationalAgent",
                **self._agent_kwargs("situational_agent"),
            ),
        )

    def create_runtime_agents(
        self,
        *,
        automation_driver: AutomationDriver | None = None,
    ) -> RuntimeAgentBundle:
        """Build the full coordinator agent set."""
        planning_agents = self.create_planning_agents()
        action_agent = ActionAgent(
            name="ActionAgent",
            automation_driver=automation_driver,
            **self._agent_kwargs("action_agent"),
        )
        test_runner = TestRunner(
            name="TestRunner",
            automation_driver=automation_driver,
            action_agent=action_agent,
            **self._agent_kwargs("test_runner"),
        )
        return RuntimeAgentBundle(
            scope_triage=planning_agents.scope_triage,
            test_planner=planning_agents.test_planner,
            test_runner=test_runner,
            action_agent=action_agent,
            situational_agent=planning_agents.situational_agent,
        )

    def _agent_kwargs(self, agent_name: str) -> dict[str, Any]:
        config = self._settings.get_agent_model_config(agent_name)
        return {
            "model": config.model,
            "temperature": config.temperature,
            "reasoning_level": config.reasoning_level,
            "modalities": config.modalities,
        }
