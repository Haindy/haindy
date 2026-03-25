"""Shared structured-output schemas for non-CU agent calls."""

from __future__ import annotations

from typing import Any

from haindy.core.types import ActionType
from haindy.models.structured_output import build_json_schema_response_format


def _string_schema() -> dict[str, Any]:
    return {"type": "string"}


def _integer_schema() -> dict[str, Any]:
    return {"type": "integer"}


def _number_schema() -> dict[str, Any]:
    return {"type": "number"}


def _boolean_schema() -> dict[str, Any]:
    return {"type": "boolean"}


def _array_schema(item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item_schema}


def _object_schema(
    properties: dict[str, Any],
    required: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or properties.keys()),
        "additionalProperties": False,
    }


_STRING_LIST_SCHEMA = _array_schema(_string_schema())
_INTEGER_LIST_SCHEMA = _array_schema(_integer_schema())
_ACTION_TYPE_ENUM = [action_type.value for action_type in ActionType]

_ENTRY_ACTION_SCHEMA = _object_schema(
    {
        "action_type": {"type": "string", "enum": _ACTION_TYPE_ENUM},
        "description": _string_schema(),
        "target": _string_schema(),
        "value": _string_schema(),
        "expected_outcome": _string_schema(),
        "computer_use_prompt": _string_schema(),
    }
)

_SITUATIONAL_SETUP_SCHEMA = _object_schema(
    {
        "web_url": _string_schema(),
        "app_name": _string_schema(),
        "launch_command": _string_schema(),
        "maximize": _boolean_schema(),
        "adb_serial": _string_schema(),
        "app_package": _string_schema(),
        "app_activity": _string_schema(),
        "adb_commands": _STRING_LIST_SCHEMA,
        "ios_udid": _string_schema(),
        "bundle_id": _string_schema(),
    }
)

_TEST_PLAN_STEP_SCHEMA = _object_schema(
    {
        "step_number": _integer_schema(),
        "action": _string_schema(),
        "expected_result": _string_schema(),
        "intent": {
            "type": "string",
            "enum": ["setup", "validation", "group_assert"],
        },
        "dependencies": _INTEGER_LIST_SCHEMA,
        "optional": _boolean_schema(),
    }
)

_TEST_CASE_SCHEMA = _object_schema(
    {
        "test_id": _string_schema(),
        "name": _string_schema(),
        "description": _string_schema(),
        "prerequisites": _STRING_LIST_SCHEMA,
        "setup_steps": _array_schema(_TEST_PLAN_STEP_SCHEMA),
        "steps": _array_schema(_TEST_PLAN_STEP_SCHEMA),
        "cleanup_steps": _array_schema(_TEST_PLAN_STEP_SCHEMA),
        "postconditions": _STRING_LIST_SCHEMA,
        "tags": _STRING_LIST_SCHEMA,
    }
)

_INTERPRETER_ACTION_SCHEMA = _object_schema(
    {
        "type": {"type": "string", "enum": _ACTION_TYPE_ENUM},
        "target": _string_schema(),
        "value": _string_schema(),
        "description": _string_schema(),
        "critical": _boolean_schema(),
        "expected_outcome": _string_schema(),
        "computer_use_prompt": _string_schema(),
    }
)

SCOPE_TRIAGE_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_scope_triage_v1",
    _object_schema(
        {
            "in_scope": _string_schema(),
            "explicit_exclusions": _STRING_LIST_SCHEMA,
            "ambiguous_points": _STRING_LIST_SCHEMA,
            "blocking_questions": _STRING_LIST_SCHEMA,
        }
    ),
)

SITUATIONAL_ASSESSMENT_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_situational_assessment_v1",
    _object_schema(
        {
            "target_type": {
                "type": "string",
                "enum": ["web", "desktop_app", "mobile_adb", "mobile_ios"],
            },
            "sufficient": _boolean_schema(),
            "missing_items": _STRING_LIST_SCHEMA,
            "setup": _SITUATIONAL_SETUP_SCHEMA,
            "entry_actions": _array_schema(_ENTRY_ACTION_SCHEMA),
            "notes": _STRING_LIST_SCHEMA,
        }
    ),
)

TEST_PLAN_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_test_plan_v1",
    _object_schema(
        {
            "name": _string_schema(),
            "description": _string_schema(),
            "requirements_source": _string_schema(),
            "test_cases": _array_schema(_TEST_CASE_SCHEMA),
            "tags": _STRING_LIST_SCHEMA,
            "estimated_duration_seconds": _integer_schema(),
        }
    ),
)

TEST_SCENARIOS_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_test_scenarios_v1",
    _object_schema(
        {
            "scenarios": _array_schema(
                _object_schema(
                    {
                        "name": _string_schema(),
                        "description": _string_schema(),
                        "priority": _string_schema(),
                        "type": _string_schema(),
                    }
                )
            )
        }
    ),
)

STEP_INTERPRETATION_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_step_interpretation_v1",
    _object_schema({"actions": _array_schema(_INTERPRETER_ACTION_SCHEMA)}),
)

STEP_VERIFICATION_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_step_verification_v1",
    _object_schema(
        {
            "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
            "reasoning": _string_schema(),
            "actual_result": _string_schema(),
            "confidence": _number_schema(),
            "request_additional_wait": _boolean_schema(),
            "recommended_wait_ms": _integer_schema(),
            "wait_reasoning": _string_schema(),
            "is_blocker": _boolean_schema(),
            "blocker_reasoning": _string_schema(),
        }
    ),
)

BUG_CLASSIFICATION_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_bug_classification_v1",
    _object_schema(
        {
            "error_type": {
                "type": "string",
                "enum": [
                    "element_not_found",
                    "assertion_failed",
                    "timeout",
                    "navigation_error",
                    "api_error",
                    "validation_error",
                    "unknown_error",
                ],
            },
            "severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low"],
            },
            "bug_description": _string_schema(),
            "reasoning": _string_schema(),
        }
    ),
)

BUG_PLAN_ASSESSMENT_RESPONSE_FORMAT = build_json_schema_response_format(
    "haindy_bug_plan_assessment_v1",
    _object_schema(
        {
            "severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low"],
            },
            "should_block": _boolean_schema(),
            "blocker_reason": _string_schema(),
            "notes": _string_schema(),
            "recommended_actions": _STRING_LIST_SCHEMA,
        }
    ),
)
