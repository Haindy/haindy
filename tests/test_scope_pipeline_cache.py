"""Tests for scope triage + planning cache integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from haindy.core.types import ScopeTriageResult, TestCase, TestPlan, TestStep
from haindy.error_handling.exceptions import ScopeTriageBlockedError
from haindy.orchestration.scope_pipeline import run_scope_triage_and_plan
from haindy.runtime.planning_cache import PlanningCache


def _sample_test_plan(name: str = "Cached Plan") -> TestPlan:
    return TestPlan(
        name=name,
        description="Sample description",
        requirements_source="Requirements source",
        test_cases=[
            TestCase(
                test_id="TC001",
                name="Sample Case",
                description="Case description",
                steps=[
                    TestStep(
                        step_number=1,
                        description="Do action",
                        action="Click login",
                        expected_result="Login succeeds",
                    )
                ],
            )
        ],
    )


def _planner_stub(test_plan: TestPlan) -> SimpleNamespace:
    return SimpleNamespace(
        name="TestPlanner",
        model="gpt-5.4",
        temperature=0.35,
        reasoning_level="high",
        modalities={"text"},
        system_prompt="planner prompt",
        create_test_plan=AsyncMock(return_value=test_plan),
        persist_test_plan=MagicMock(),
    )


def _triage_stub(result: ScopeTriageResult) -> SimpleNamespace:
    return SimpleNamespace(
        name="ScopeTriage",
        model="gpt-5.4",
        temperature=0.15,
        reasoning_level="high",
        modalities={"text"},
        system_prompt="triage prompt",
        triage_scope=AsyncMock(return_value=result),
    )


@pytest.mark.asyncio
async def test_cache_hit_skips_models_and_refreshes_plan_identity(tmp_path) -> None:
    cache = PlanningCache(tmp_path / "planning_cache.json")
    triage_result = ScopeTriageResult(
        in_scope="Test login only",
        explicit_exclusions=["Do not test reports"],
    )
    planner = _planner_stub(_sample_test_plan())
    triage_agent = _triage_stub(triage_result)
    requirements = "Test login flow"
    context = {"execution_context": "web app context"}

    first_plan, first_triage = await run_scope_triage_and_plan(
        requirements=requirements,
        planner=planner,
        triage_agent=triage_agent,
        context=context,
        planning_cache=cache,
    )

    triage_agent.triage_scope.reset_mock()
    planner.create_test_plan.reset_mock()
    planner.persist_test_plan.reset_mock()

    second_plan, second_triage = await run_scope_triage_and_plan(
        requirements=requirements,
        planner=planner,
        triage_agent=triage_agent,
        context=context,
        planning_cache=cache,
    )

    triage_agent.triage_scope.assert_not_awaited()
    planner.create_test_plan.assert_not_awaited()
    planner.persist_test_plan.assert_called_once()

    assert second_plan.plan_id != first_plan.plan_id
    assert second_triage.model_dump() == first_triage.model_dump()


@pytest.mark.asyncio
async def test_cache_replays_blocker_without_model_call(tmp_path) -> None:
    cache = PlanningCache(tmp_path / "planning_cache.json")
    triage_result = ScopeTriageResult(
        in_scope="",
        blocking_questions=["Missing URL"],
    )
    planner = _planner_stub(_sample_test_plan())
    triage_agent = _triage_stub(triage_result)
    requirements = "Test admin panel"
    context = {"execution_context": "missing target details"}

    with pytest.raises(ScopeTriageBlockedError):
        await run_scope_triage_and_plan(
            requirements=requirements,
            planner=planner,
            triage_agent=triage_agent,
            context=context,
            planning_cache=cache,
        )

    triage_agent.triage_scope.reset_mock()
    planner.create_test_plan.reset_mock()

    with pytest.raises(ScopeTriageBlockedError) as exc_info:
        await run_scope_triage_and_plan(
            requirements=requirements,
            planner=planner,
            triage_agent=triage_agent,
            context=context,
            planning_cache=cache,
        )

    triage_agent.triage_scope.assert_not_awaited()
    planner.create_test_plan.assert_not_awaited()
    assert exc_info.value.triage_result is not None
    assert exc_info.value.triage_result.blocking_questions == ["Missing URL"]


@pytest.mark.asyncio
async def test_cache_key_context_change_causes_cache_miss(tmp_path) -> None:
    cache = PlanningCache(tmp_path / "planning_cache.json")
    triage_result = ScopeTriageResult(
        in_scope="Test login",
        explicit_exclusions=[],
        ambiguous_points=[],
        blocking_questions=[],
    )
    planner = _planner_stub(_sample_test_plan("Plan A"))
    triage_agent = _triage_stub(triage_result)
    requirements = "Test login flow"
    context = {"execution_context": "full context"}

    await run_scope_triage_and_plan(
        requirements=requirements,
        planner=planner,
        triage_agent=triage_agent,
        context=context,
        cache_key_context={"execution_context": "context-v1"},
        planning_cache=cache,
    )

    await run_scope_triage_and_plan(
        requirements=requirements,
        planner=planner,
        triage_agent=triage_agent,
        context=context,
        cache_key_context={"execution_context": "context-v2"},
        planning_cache=cache,
    )

    assert triage_agent.triage_scope.await_count == 2
    assert planner.create_test_plan.await_count == 2
