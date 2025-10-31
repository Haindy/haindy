"""
Shared helpers for the two-pass scope triage and test planning workflow.
"""

from typing import Any, Dict, Optional, Tuple

from src.agents.scope_triage import ScopeTriageAgent
from src.agents.test_planner import TestPlannerAgent
from src.core.types import ScopeTriageResult, TestPlan
from src.error_handling.exceptions import ScopeTriageBlockedError
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


async def run_scope_triage_and_plan(
    requirements: str,
    planner: TestPlannerAgent,
    triage_agent: ScopeTriageAgent,
    context: Optional[Dict[str, str]] = None,
    planner_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[TestPlan, ScopeTriageResult]:
    """
    Execute the two-pass planning pipeline.

    Args:
        requirements: Raw requirements or PRD bundle supplied by the user
        planner: Initialized TestPlannerAgent
        triage_agent: Initialized ScopeTriageAgent
        context: Optional additional context (credentials, URLs, etc.)

    Returns:
        Tuple consisting of the generated TestPlan and the ScopeTriageResult

    Raises:
        ScopeTriageBlockedError: When blocking questions prevent safe planning
    """
    triage_result = await triage_agent.triage_scope(
        requirements=requirements,
        context=context,
    )

    if triage_result.has_blockers():
        logger.warning(
            "Scope triage identified blocking questions; aborting planning",
            extra={"blocking_count": len(triage_result.blocking_questions)},
        )
        raise ScopeTriageBlockedError(triage_result=triage_result)

    curated_scope = triage_result.build_planner_brief()
    create_kwargs = planner_kwargs or {}
    test_plan = await planner.create_test_plan(
        requirements=requirements,
        context=context,
        curated_scope=curated_scope,
        ambiguous_points=triage_result.ambiguous_points,
        **create_kwargs,
    )

    logger.info(
        "Two-pass planning completed",
        extra={
            "plan_id": str(test_plan.plan_id),
            "ambiguous_points": len(triage_result.ambiguous_points),
            "exclusions": len(triage_result.explicit_exclusions),
        },
    )

    return test_plan, triage_result
