"""
Shared helpers for the two-pass scope triage and test planning workflow.
"""

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import uuid4

from src.agents.scope_triage import ScopeTriageAgent
from src.agents.test_planner import TestPlannerAgent
from src.config.settings import get_settings
from src.core.types import ScopeTriageResult, TestPlan
from src.error_handling.exceptions import ScopeTriageBlockedError
from src.monitoring.logger import get_logger
from src.runtime.planning_cache import (
    PlanningCache,
    build_planning_cache_key_payload,
    hash_planning_cache_key,
)

logger = get_logger(__name__)


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


def _resolve_planning_cache(
    planning_cache: PlanningCache | None,
) -> PlanningCache | None:
    if planning_cache is not None:
        return planning_cache

    settings = get_settings()
    if not settings.enable_planning_cache:
        return None
    return PlanningCache(settings.planning_cache_path)


def _refresh_cached_plan_identity(test_plan: TestPlan) -> TestPlan:
    refreshed_plan = TestPlan.model_validate(test_plan.model_dump(mode="json"))
    refreshed_plan.plan_id = uuid4()
    refreshed_plan.created_at = datetime.now(timezone.utc)
    return refreshed_plan


async def run_scope_triage_and_plan(
    requirements: str,
    planner: TestPlannerAgent,
    triage_agent: ScopeTriageAgent,
    context: dict[str, Any] | None = None,
    planner_kwargs: dict[str, Any] | None = None,
    cache_key_context: dict[str, Any] | None = None,
    planning_cache: PlanningCache | None = None,
) -> tuple[TestPlan, ScopeTriageResult]:
    """
    Execute the two-pass planning pipeline.

    Args:
        requirements: Raw requirements or PRD bundle supplied by the user
        planner: Initialized TestPlannerAgent
        triage_agent: Initialized ScopeTriageAgent
        context: Optional additional context (credentials, URLs, etc.)
        planner_kwargs: Optional kwargs forwarded to TestPlannerAgent.create_test_plan
        cache_key_context: Optional context override used only for cache keying
        planning_cache: Optional explicit planning cache instance

    Returns:
        Tuple consisting of the generated TestPlan and the ScopeTriageResult

    Raises:
        ScopeTriageBlockedError: When blocking questions prevent safe planning
    """
    create_kwargs = planner_kwargs or {}
    cache = _resolve_planning_cache(planning_cache)
    cache_key_hash: str | None = None

    if cache is not None:
        cache_payload = build_planning_cache_key_payload(
            requirements=requirements,
            context=cache_key_context if cache_key_context is not None else context,
            planner_kwargs=create_kwargs,
            triage_signature=_agent_signature(triage_agent),
            planner_signature=_agent_signature(planner),
        )
        cache_key_hash = hash_planning_cache_key(cache_payload)
        cached_entry = cache.lookup(cache_key_hash)

        if cached_entry is not None:
            try:
                triage_result = ScopeTriageResult.model_validate(
                    cached_entry.triage_payload
                )
                if cached_entry.has_blockers:
                    logger.info(
                        "Planning cache hit for blocker triage result",
                        extra={"cache_key_hash": cache_key_hash},
                    )
                    raise ScopeTriageBlockedError(triage_result=triage_result)

                if cached_entry.test_plan_payload is None:
                    cache.invalidate(cache_key_hash)
                    logger.warning(
                        "Planning cache entry missing plan payload; invalidated",
                        extra={"cache_key_hash": cache_key_hash},
                    )
                else:
                    cached_plan = TestPlan.model_validate(
                        cached_entry.test_plan_payload
                    )
                    refreshed_plan = _refresh_cached_plan_identity(cached_plan)
                    persist_fn = getattr(planner, "persist_test_plan", None)
                    if callable(persist_fn):
                        persist_fn(refreshed_plan)
                    logger.info(
                        "Planning cache hit for triage and test plan",
                        extra={
                            "cache_key_hash": cache_key_hash,
                            "plan_id": str(refreshed_plan.plan_id),
                        },
                    )
                    return refreshed_plan, triage_result
            except ScopeTriageBlockedError:
                raise
            except Exception:
                cache.invalidate(cache_key_hash)
                logger.warning(
                    "Planning cache entry invalid; falling back to model execution",
                    exc_info=True,
                    extra={"cache_key_hash": cache_key_hash},
                )

    triage_result = await triage_agent.triage_scope(
        requirements=requirements,
        context=context,
    )

    if triage_result.has_blockers():
        if cache is not None and cache_key_hash is not None:
            cache.store(
                key_hash=cache_key_hash,
                triage_payload=triage_result.model_dump(mode="json"),
                test_plan_payload=None,
                has_blockers=True,
            )
        logger.warning(
            "Scope triage identified blocking questions; aborting planning",
            extra={"blocking_count": len(triage_result.blocking_questions)},
        )
        raise ScopeTriageBlockedError(triage_result=triage_result)

    curated_scope = triage_result.build_planner_brief()
    test_plan = await planner.create_test_plan(
        requirements=requirements,
        context=context,
        curated_scope=curated_scope,
        ambiguous_points=triage_result.ambiguous_points,
        **create_kwargs,
    )

    if cache is not None and cache_key_hash is not None:
        cache.store(
            key_hash=cache_key_hash,
            triage_payload=triage_result.model_dump(mode="json"),
            test_plan_payload=test_plan.model_dump(mode="json"),
            has_blockers=False,
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
