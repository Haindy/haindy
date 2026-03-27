"""Helpers for provider-neutral structured-output response formats."""

from __future__ import annotations

from typing import Any

LEGACY_JSON_OBJECT_SCHEMA_NAME = "haindy_response"


def build_json_schema_response_format(
    name: str,
    schema: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Build the provider-neutral structured-output response format envelope."""
    normalized_name = str(name or "").strip() or LEGACY_JSON_OBJECT_SCHEMA_NAME
    return {
        "type": "json_schema",
        "json_schema": {
            "name": normalized_name,
            "schema": schema,
            "strict": bool(strict),
        },
    }


def build_legacy_json_object_schema() -> dict[str, Any]:
    """Return a permissive object schema for json_object compatibility paths."""
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


def response_format_expects_json(response_format: dict[str, Any] | None) -> bool:
    """Return True when the response format requests JSON content."""
    format_type = response_format.get("type") if response_format else None
    return format_type in {"json_object", "json_schema"}


def extract_json_schema_definition(
    response_format: dict[str, Any] | None,
    *,
    fallback_name: str = LEGACY_JSON_OBJECT_SCHEMA_NAME,
) -> dict[str, Any] | None:
    """Normalize response_format into a named JSON schema payload when possible."""
    if not response_format:
        return None

    format_type = response_format.get("type")
    if format_type == "json_object":
        return {
            "name": fallback_name,
            "schema": build_legacy_json_object_schema(),
            "strict": True,
        }

    if format_type != "json_schema":
        return None

    raw_schema = response_format.get("json_schema")
    if not isinstance(raw_schema, dict):
        return None

    wrapped_schema = raw_schema.get("schema")
    if isinstance(wrapped_schema, dict):
        schema = wrapped_schema
        name = str(raw_schema.get("name") or fallback_name).strip() or fallback_name
        strict = bool(raw_schema.get("strict", True))
        return {"name": name, "schema": schema, "strict": strict}

    return {
        "name": fallback_name,
        "schema": raw_schema,
        "strict": True,
    }
