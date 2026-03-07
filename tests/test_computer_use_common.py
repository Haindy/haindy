"""Computer Use shared helper tests."""

from __future__ import annotations

from src.agents.computer_use.common import normalize_response


def test_normalize_response_suppresses_model_dump_warnings() -> None:
    class DummyResponse:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def model_dump(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(kwargs)
            return {"id": "resp_test"}

    response = DummyResponse()

    assert normalize_response(response) == {"id": "resp_test"}
    assert response.calls == [{"warnings": "none"}]
