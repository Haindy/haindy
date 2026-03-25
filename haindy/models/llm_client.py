"""Protocol definition for LLM clients used by non-CU agents."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from haindy.models.openai_client import ResponseStreamObserver


@runtime_checkable
class LLMClient(Protocol):
    async def call(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        system_prompt: str | None,
        response_format: dict[str, Any] | None,
        reasoning_level: str | None,
        modalities: set[str] | None,
        stream: bool,
        stream_observer: ResponseStreamObserver | None,
    ) -> dict[str, Any]: ...


def dispatch_observer(observer: Any, method_name: str, *args: Any) -> None:
    """Invoke an observer method by name, silently ignoring missing methods and errors."""
    method = getattr(observer, method_name, None)
    if method is not None:
        try:
            method(*args)
        except Exception:
            pass
