"""Tests for tool-call session variable handling."""

from src.tool_call_mode.variables import SessionVariableStore


def test_variable_store_interpolates_known_variables_and_preserves_unknowns() -> None:
    store = SessionVariableStore()
    store.set("USERNAME", "alice@example.com")
    store.set("PASSWORD", "hunter2", secret=True)

    rendered = store.interpolate(
        "sign in with {{USERNAME}} and {{PASSWORD}} and keep {{UNKNOWN}} unchanged"
    )

    assert rendered == (
        "sign in with alice@example.com and hunter2 and keep {{UNKNOWN}} unchanged"
    )


def test_variable_store_unknown_token_is_left_unchanged() -> None:
    store = SessionVariableStore()

    assert store.interpolate("type {{EMAIL}} into the field") == (
        "type {{EMAIL}} into the field"
    )


def test_variable_store_redacts_only_secret_values() -> None:
    store = SessionVariableStore()
    store.set("USERNAME", "alice@example.com")
    store.set("PASSWORD", "hunter2", secret=True)

    text = "Used alice@example.com with hunter2 during sign-in."

    assert (
        store.redact(text) == "Used alice@example.com with [redacted] during sign-in."
    )


def test_variable_store_public_map_masks_secret_values() -> None:
    store = SessionVariableStore()
    store.set("USERNAME", "alice@example.com")
    store.set("PASSWORD", "hunter2", secret=True)

    assert store.as_public_map() == {
        "PASSWORD": "[secret]",
        "USERNAME": "alice@example.com",
    }


def test_variable_store_double_brace_token_not_matched_without_closing() -> None:
    store = SessionVariableStore()
    store.set("NAME", "alice")

    assert store.interpolate("hello {{NAME} world") == "hello {{NAME} world"
