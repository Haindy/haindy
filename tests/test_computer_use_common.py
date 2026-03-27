"""Computer Use shared helper tests."""

from __future__ import annotations

from types import SimpleNamespace

from haindy.agents.computer_use.common import (
    extract_assistant_text,
    extract_google_function_call_envelopes,
    normalize_response,
)


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


def test_normalize_response_preserves_missing_google_candidate_attributes() -> None:
    function_call = SimpleNamespace(name="click_at", args={"x": 12, "y": 34}, id="c1")
    text_part = SimpleNamespace(text="Done.")
    response_candidates = [
        SimpleNamespace(
            content=SimpleNamespace(
                parts=[
                    SimpleNamespace(function_call=function_call),
                    text_part,
                ]
            )
        )
    ]

    class DummyGoogleResponse:
        def model_dump(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"id": "resp_google", "candidates": []}

        @property
        def candidates(self) -> list[object]:
            return response_candidates

    normalized = normalize_response(DummyGoogleResponse())

    envelopes = extract_google_function_call_envelopes(normalized)
    assert normalized["candidates"] == response_candidates
    assert len(envelopes) == 1
    assert envelopes[0].function_call.name == "click_at"
    assert envelopes[0].function_call.args == {"x": 12, "y": 34}
    assert envelopes[0].function_call.id == "c1"
    assert extract_assistant_text(normalized) == "Done."


def test_extract_assistant_text_prefers_last_meaningful_text_block() -> None:
    response = {
        "outputs": [
            {"type": "text", "text": "60"},
            {"type": "text", "text": "The Schedule screen is now visible."},
        ]
    }

    assert extract_assistant_text(response) == "The Schedule screen is now visible."


def test_extract_assistant_text_preserves_json_payloads() -> None:
    response = {
        "outputs": [
            {
                "type": "text",
                "text": '{"verdict":"PASS","reasoning":"Step passed","confidence":0.9}',
            }
        ]
    }

    assert (
        extract_assistant_text(response)
        == '{"verdict":"PASS","reasoning":"Step passed","confidence":0.9}'
    )
