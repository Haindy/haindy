"""Shared model-client exception types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCallError(RuntimeError):
    """Structured model-call failure raised after a provider response exists."""

    message: str
    failure_kind: str
    response_payload: Any | None = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)
