"""Tests for tool-call session variable handling."""

from src.tool_call_mode.variables import SessionVariableStore


def test_variable_store_interpolates_known_variables_and_preserves_unknowns() -> None:
    store = SessionVariableStore()
    store.set("USERNAME", "alice@example.com")
    store.set("PASSWORD", "hunter2", secret=True)

    rendered = store.interpolate(
        "sign in with $USERNAME and $PASSWORD and keep $UNKNOWN unchanged"
    )

    assert rendered == (
        "sign in with alice@example.com and hunter2 and keep $UNKNOWN unchanged"
    )


def test_variable_store_treats_double_dollar_as_literal_dollar() -> None:
    store = SessionVariableStore()
    store.set("PRICE", "19.99")

    assert store.interpolate("The price is $$$PRICE") == "The price is $19.99"


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
