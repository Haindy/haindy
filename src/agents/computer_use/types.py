"""Shared types for Computer Use orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.enhanced_types import ComputerToolTurn, SafetyEvent


class ComputerUseExecutionError(RuntimeError):
    """Raised when the Computer Use orchestration fails irrecoverably."""


@dataclass(frozen=True)
class InteractionConstraints:
    """Coarse-grained interaction constraints derived from step/action text."""

    disallow_scroll: bool = False

    def has_any(self) -> bool:
        return self.disallow_scroll

    def to_prompt(self) -> str:
        lines: list[str] = []
        if self.disallow_scroll:
            lines.append("- Do NOT scroll (no scroll actions; no mouse wheel).")
        return "\n".join(lines)

    def apply_overrides(
        self, metadata: dict[str, Any] | None
    ) -> InteractionConstraints:
        """Apply optional constraint overrides supplied via runtime metadata."""
        if not metadata:
            return self

        disallow_scroll_override = metadata.get("disallow_scroll")
        if isinstance(disallow_scroll_override, bool):
            return InteractionConstraints(disallow_scroll=disallow_scroll_override)

        policy = str(metadata.get("scroll_policy") or "").strip().lower()
        if policy in {"auto", ""}:
            return self
        if policy in {"allow", "allow_scroll"}:
            return InteractionConstraints(disallow_scroll=False)
        if policy in {"disallow", "disallow_scroll"}:
            return InteractionConstraints(disallow_scroll=True)
        return self

    @staticmethod
    def from_text(text: str) -> InteractionConstraints:
        lowered = (text or "").lower()
        strict_no_scroll = any(
            phrase in lowered
            for phrase in (
                "without scrolling",
                "no scrolling",
            )
        )
        if strict_no_scroll:
            return InteractionConstraints(disallow_scroll=True)

        soft_no_scroll = any(
            phrase in lowered
            for phrase in (
                "do not scroll",
                "don't scroll",
                "avoid scrolling",
            )
        )
        if not soft_no_scroll:
            return InteractionConstraints(disallow_scroll=False)

        allows_scroll = any(
            phrase in lowered
            for phrase in (
                "scroll down",
                "scroll up",
                "scroll left",
                "scroll right",
                "scroll to",
                "scroll until",
                "scroll just",
                "scroll by",
                "scroll a bit",
                "scroll a little",
                "scroll slightly",
                "scroll enough",
                "scroll more",
                "scroll further down",
                "scroll further up",
                "scroll the page",
            )
        )
        return InteractionConstraints(disallow_scroll=not allows_scroll)


def _strip_bytes(obj: Any) -> Any:
    """Recursively replace bytes values with a placeholder string for logging."""
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, dict):
        return {k: _strip_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_bytes(v) for v in obj]
    return obj


@dataclass
class ComputerUseSessionResult:
    """Result of executing a Computer Use session."""

    actions: list[ComputerToolTurn] = field(default_factory=list)
    safety_events: list[SafetyEvent] = field(default_factory=list)
    final_output: str | None = None
    response_ids: list[str] = field(default_factory=list)
    last_response: dict[str, Any] | None = None
    terminal_status: Literal["success", "failed"] = "success"
    terminal_failure_reason: str | None = None
    terminal_failure_code: str | None = None


@dataclass(frozen=True)
class GoogleFunctionCallEnvelope:
    """Function call plus deterministic ordering metadata for a single turn."""

    function_call: Any
    sequence: int
    candidate_index: int
    part_index: int


__all__ = [
    "ComputerUseExecutionError",
    "ComputerUseSessionResult",
    "GoogleFunctionCallEnvelope",
    "InteractionConstraints",
    "_strip_bytes",
]
