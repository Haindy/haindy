"""Session-variable interpolation and redaction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VAR_TOKEN_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


@dataclass(frozen=True)
class SessionVariable:
    """One stored session variable."""

    value: str
    secret: bool = False


class SessionVariableStore:
    """In-memory store for session variables and secret redaction."""

    def __init__(self) -> None:
        self._values: dict[str, SessionVariable] = {}

    @staticmethod
    def validate_name(name: str) -> str:
        normalized = str(name or "").strip()
        if not _VAR_NAME_PATTERN.match(normalized):
            raise ValueError("Variable names must match [A-Za-z_][A-Za-z0-9_]*.")
        return normalized

    def set(self, name: str, value: str, *, secret: bool = False) -> None:
        normalized = self.validate_name(name)
        self._values[normalized] = SessionVariable(str(value), bool(secret))

    def unset(self, name: str) -> bool:
        normalized = self.validate_name(name)
        return self._values.pop(normalized, None) is not None

    def as_public_map(self) -> dict[str, str]:
        rendered: dict[str, str] = {}
        for name, variable in sorted(self._values.items()):
            rendered[name] = "[secret]" if variable.secret else variable.value
        return rendered

    def secret_values(self) -> list[str]:
        """Return all non-empty secret values for redaction."""

        return sorted(
            {
                variable.value
                for variable in self._values.values()
                if variable.secret and variable.value
            },
            key=len,
            reverse=True,
        )

    def interpolate(self, text: str) -> str:
        """Replace {{NAME}} tokens with stored variable values.

        Unknown tokens are left unchanged. No escaping is needed because
        {{NAME}} has no special meaning to the shell.
        """

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            variable = self._values.get(name)
            return variable.value if variable is not None else match.group(0)

        return _VAR_TOKEN_PATTERN.sub(_replace, str(text or ""))

    def redact(self, text: str | None) -> str | None:
        """Replace any secret value echoes with `[redacted]`."""

        if text is None:
            return None
        redacted = str(text)
        for secret in self.secret_values():
            redacted = redacted.replace(secret, "[redacted]")
        return redacted
