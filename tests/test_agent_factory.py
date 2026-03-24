"""Tests for shared runtime agent factory helpers."""

from __future__ import annotations

from dataclasses import dataclass

from haindy.runtime.agent_factory import AgentFactory


@dataclass
class _AgentConfig:
    model: str
    temperature: float
    reasoning_level: str
    modalities: set[str]


class _SettingsStub:
    def get_agent_model_config(self, agent_name: str) -> _AgentConfig:
        return _AgentConfig(
            model=f"{agent_name}-model",
            temperature=0.2,
            reasoning_level="medium",
            modalities={"text"},
        )


class _PlanningAgentStub:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _ActionAgentStub(_PlanningAgentStub):
    pass


class _TestRunnerStub(_PlanningAgentStub):
    pass


def test_factory_creates_planning_bundle_with_shared_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "haindy.runtime.agent_factory.ScopeTriageAgent", _PlanningAgentStub
    )
    monkeypatch.setattr(
        "haindy.runtime.agent_factory.TestPlannerAgent", _PlanningAgentStub
    )
    monkeypatch.setattr(
        "haindy.runtime.agent_factory.SituationalAgent", _PlanningAgentStub
    )

    factory = AgentFactory(settings=_SettingsStub())

    bundle = factory.create_planning_agents()

    assert bundle.scope_triage.kwargs["name"] == "ScopeTriage"
    assert bundle.scope_triage.kwargs["model"] == "scope_triage-model"
    assert bundle.test_planner.kwargs["name"] == "TestPlanner"
    assert bundle.test_planner.kwargs["model"] == "test_planner-model"
    assert bundle.situational_agent.kwargs["name"] == "SituationalAgent"
    assert bundle.situational_agent.kwargs["model"] == "situational_agent-model"


def test_factory_creates_runtime_bundle_and_wires_action_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        "haindy.runtime.agent_factory.ScopeTriageAgent", _PlanningAgentStub
    )
    monkeypatch.setattr(
        "haindy.runtime.agent_factory.TestPlannerAgent", _PlanningAgentStub
    )
    monkeypatch.setattr(
        "haindy.runtime.agent_factory.SituationalAgent", _PlanningAgentStub
    )
    monkeypatch.setattr("haindy.runtime.agent_factory.ActionAgent", _ActionAgentStub)
    monkeypatch.setattr("haindy.runtime.agent_factory.TestRunner", _TestRunnerStub)

    driver = object()
    factory = AgentFactory(settings=_SettingsStub())

    bundle = factory.create_runtime_agents(automation_driver=driver)

    assert bundle.action_agent.kwargs["automation_driver"] is driver
    assert "model" not in bundle.action_agent.kwargs
    assert "reasoning_level" not in bundle.action_agent.kwargs
    assert "modalities" not in bundle.action_agent.kwargs
    assert bundle.test_runner.kwargs["automation_driver"] is driver
    assert bundle.test_runner.kwargs["action_agent"] is bundle.action_agent
    assert bundle.as_dict()["action_agent"] is bundle.action_agent
    assert bundle.as_dict()["test_runner"] is bundle.test_runner
