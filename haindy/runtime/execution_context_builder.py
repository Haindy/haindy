"""Shared execution-context assembly for planning and runtime flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haindy.agents.situational_agent import SituationalAssessment
from haindy.runtime.environment import (
    normalize_automation_backend,
    resolve_runtime_environment,
)


@dataclass(frozen=True)
class ExecutionContextBundle:
    """Canonical context payloads derived from a situational assessment."""

    planning_context: dict[str, Any]
    planning_cache_key_context: dict[str, Any]
    test_context: dict[str, Any]
    initial_url: str | None


def build_execution_context_bundle(
    *,
    context_text: str,
    assessment: SituationalAssessment,
    automation_backend: str,
) -> ExecutionContextBundle:
    """Build the planning and execution payloads from one source of truth."""
    normalized_backend = normalize_automation_backend(automation_backend)
    runtime_spec = resolve_runtime_environment(
        automation_backend=normalized_backend,
        target_type=assessment.target_type,
    )

    planning_setup = _build_setup_payload(assessment, stringify_maximize=True)
    test_setup = _build_setup_payload(assessment, stringify_maximize=False)

    planning_context = {
        "execution_context": context_text,
        "target_type": runtime_spec.target_type,
        "automation_backend": normalized_backend,
        **planning_setup,
    }
    test_context = {
        "execution_context": context_text,
        "target_type": runtime_spec.target_type,
        "automation_backend": normalized_backend,
        "entry_setup": test_setup,
        "setup_notes": list(assessment.notes),
    }

    return ExecutionContextBundle(
        planning_context=planning_context,
        planning_cache_key_context={"execution_context": context_text},
        test_context=test_context,
        initial_url=assessment.setup.web_url or None,
    )


def _build_setup_payload(
    assessment: SituationalAssessment,
    *,
    stringify_maximize: bool,
) -> dict[str, Any]:
    maximize: bool | str = assessment.setup.maximize
    if stringify_maximize:
        maximize = str(maximize)

    return {
        "web_url": assessment.setup.web_url,
        "app_name": assessment.setup.app_name,
        "launch_command": assessment.setup.launch_command,
        "maximize": maximize,
        "adb_serial": assessment.setup.adb_serial,
        "app_package": assessment.setup.app_package,
        "app_activity": assessment.setup.app_activity,
        "adb_commands": assessment.setup.adb_commands,
    }
