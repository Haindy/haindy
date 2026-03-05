"""Runtime helpers for caching, replay, and tracing."""

from src.runtime.environment import (
    RuntimeEnvironmentSpec,
    coordinate_cache_path_for_environment,
    normalize_automation_backend,
    normalize_runtime_environment_name,
    normalize_target_type,
    resolve_runtime_environment,
    resolve_runtime_environment_from_context,
    runtime_environment_spec,
)
from src.runtime.evidence import EvidenceManager
from src.runtime.execution_replay_cache import (
    ExecutionReplayCache,
    ExecutionReplayCacheKey,
    ExecutionReplayEntry,
)
from src.runtime.planning_cache import (
    PlanningCache,
    build_planning_cache_key_payload,
    hash_planning_cache_key,
)
from src.runtime.situational_cache import (
    SituationalCache,
    build_situational_cache_key_payload,
    hash_situational_cache_key,
)
from src.runtime.task_cache import TaskPlanCache
from src.runtime.trace import RunTraceWriter, load_model_calls_for_run

__all__ = [
    "EvidenceManager",
    "ExecutionReplayCache",
    "ExecutionReplayCacheKey",
    "ExecutionReplayEntry",
    "RuntimeEnvironmentSpec",
    "coordinate_cache_path_for_environment",
    "normalize_automation_backend",
    "normalize_runtime_environment_name",
    "normalize_target_type",
    "PlanningCache",
    "build_planning_cache_key_payload",
    "hash_planning_cache_key",
    "resolve_runtime_environment",
    "resolve_runtime_environment_from_context",
    "runtime_environment_spec",
    "SituationalCache",
    "build_situational_cache_key_payload",
    "hash_situational_cache_key",
    "TaskPlanCache",
    "RunTraceWriter",
    "load_model_calls_for_run",
]
