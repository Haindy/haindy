"""Session-variable interpolation and redaction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
        """Apply exact `$NAME` interpolation with `$$` escaping."""

        raw = str(text or "")
        result: list[str] = []
        idx = 0
        length = len(raw)

        while idx < length:
            current = raw[idx]
            if current != "$":
                result.append(current)
                idx += 1
                continue

            if idx + 1 < length and raw[idx + 1] == "$":
                result.append("$")
                idx += 2
                continue

            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", raw[idx + 1 :])
            if not match:
                result.append("$")
                idx += 1
                continue

            name = match.group(0)
            variable = self._values.get(name)
            if variable is None:
                result.append(f"${name}")
            else:
                result.append(variable.value)
            idx += 1 + len(name)

        return "".join(result)

    def redact(self, text: str | None) -> str | None:
        """Replace any secret value echoes with `[redacted]`."""

        if text is None:
            return None
        redacted = str(text)
        for secret in self.secret_values():
            redacted = redacted.replace(secret, "[redacted]")
        return redacted
