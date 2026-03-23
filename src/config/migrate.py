"""Migration logic for importing an existing .env file into the new config system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from src.auth.credentials import set_api_key
from src.config.settings import (
    _SECRET_FIELD_TO_PROVIDER,
    SETTINGS_ENV_VARS,
    _parse_env_field,
)
from src.config.settings_file import flat_to_nested, write_settings_file


@dataclass
class MigrationResult:
    """Summary of a migration run."""

    settings_written: list[str] = field(default_factory=list)
    secrets_stored: list[str] = field(default_factory=list)
    secrets_skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False


def migrate_from_dotenv(
    dotenv_path: Path = Path(".env"),
    settings_out: Path | None = None,
    dry_run: bool = False,
) -> MigrationResult:
    """Read *dotenv_path* and migrate settings and secrets to the new system.

    Settings (non-secret fields) are written to *settings_out* (defaults to
    ``~/.haindy/settings.json``). Secrets (API keys) are stored via
    ``set_api_key``.

    The original .env file is not modified or deleted.
    """
    result = MigrationResult(dry_run=dry_run)

    if settings_out is None:
        settings_out = Path("~/.haindy/settings.json").expanduser()

    if not dotenv_path.exists():
        result.warnings.append(f"File not found: {dotenv_path}")
        return result

    raw = dotenv_values(dotenv_path)

    # Reverse-lookup: env var name -> field name
    _env_to_field = {v: k for k, v in SETTINGS_ENV_VARS.items()}

    flat_settings: dict[str, Any] = {}

    for env_name, raw_value in raw.items():
        field_name = _env_to_field.get(env_name)
        if field_name is None:
            if env_name.startswith("HAINDY_"):
                result.warnings.append(f"Unrecognized env var: {env_name}")
            continue

        if field_name in _SECRET_FIELD_TO_PROVIDER:
            provider = _SECRET_FIELD_TO_PROVIDER[field_name]
            if raw_value and raw_value.strip():
                result.secrets_stored.append(provider)
                if not dry_run:
                    set_api_key(provider, raw_value.strip())
            else:
                result.secrets_skipped.append(provider)
            continue

        # Non-secret field: parse and accumulate
        if raw_value is None or raw_value == "":
            continue
        try:
            parsed = _parse_env_field(field_name, raw_value)
        except (ValueError, Exception) as exc:
            result.warnings.append(f"Could not parse {env_name}={raw_value!r}: {exc}")
            continue

        flat_settings[field_name] = parsed
        result.settings_written.append(field_name)

    if flat_settings and not dry_run:
        nested = flat_to_nested(flat_settings)
        write_settings_file(settings_out, nested)

    return result
