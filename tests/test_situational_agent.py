"""Unit tests for SituationalAgent."""

from hashlib import sha256
from unittest.mock import AsyncMock, MagicMock

import pytest

from haindy.agents.situational_agent import SituationalAgent
from haindy.core.types import ActionType
from haindy.runtime.situational_cache import (
    SituationalCache,
    build_situational_cache_key_payload,
    hash_situational_cache_key,
)


def test_heuristic_web_context_is_sufficient() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment(
        "Target type: web\nOpen https://example.com and maximize window"
    )
    assert assessment.sufficient is True
    assert assessment.target_type == "web"
    assert assessment.setup.web_url == "https://example.com"
    assert assessment.setup.maximize is True


def test_heuristic_desktop_context_uses_visual_entry_action_without_os_identifiers() -> (
    None
):
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment("Target type: desktop app for KeenBench")
    assert assessment.sufficient is True
    assert assessment.target_type == "desktop_app"
    assert not assessment.missing_items
    assert assessment.entry_actions
    assert assessment.entry_actions[0].action_type == ActionType.CLICK


def test_heuristic_desktop_context_with_app_name() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment("app_name: Calculator")
    assert assessment.target_type == "desktop_app"
    assert assessment.sufficient is True
    assert assessment.setup.app_name == "Calculator"
    assert assessment.entry_actions
    assert "Calculator" in assessment.entry_actions[0].description


def test_heuristic_mobile_context_requires_setup_path() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment(
        "target_type: mobile_adb\nadb_serial: emulator-5554\napp_package: com.example.app"
    )
    assert assessment.target_type == "mobile_adb"
    assert assessment.sufficient is True
    assert assessment.setup.adb_serial == "emulator-5554"
    assert assessment.setup.app_package == "com.example.app"


def test_parse_assessment_mobile_blocks_without_structured_or_commands() -> None:
    agent = SituationalAgent()
    payload = {
        "target_type": "mobile_adb",
        "sufficient": False,
        "missing_items": [],
        "setup": {"adb_serial": "", "app_package": "", "adb_commands": []},
        "entry_actions": [],
        "notes": [],
    }

    assessment = agent._parse_assessment(
        payload,
        "Test mobile login",
        "Target is android mobile app",
    )

    assert assessment.sufficient is False
    assert assessment.missing_items


def test_parse_assessment_mobile_allows_command_path_without_structured_fields() -> (
    None
):
    agent = SituationalAgent()
    payload = {
        "target_type": "mobile_adb",
        "sufficient": True,
        "missing_items": [],
        "setup": {
            "adb_serial": "",
            "app_package": "",
            "adb_commands": ["adb devices", "adb shell monkey -p com.example.app 1"],
        },
        "entry_actions": [],
        "notes": [],
    }

    assessment = agent._parse_assessment(
        payload,
        "Test mobile login",
        "Target is android mobile app",
    )

    assert assessment.sufficient is True
    assert not assessment.missing_items
    assert assessment.setup.adb_commands


def test_parse_assessment_mobile_demotes_non_launch_context_questions() -> None:
    agent = SituationalAgent()
    payload = {
        "target_type": "mobile_adb",
        "sufficient": False,
        "missing_items": [
            (
                "Branch deep link URL containing a team code for Path 9 (TC-09). "
                "The provided quick link appears to be a shareable invitation link "
                "for Path 7, not the Branch deep link."
            ),
            (
                "Clarification/approval on execution order: Path 2 (TC-02) requires "
                "phone/SMS verification but phone-verification tests are marked "
                "OUT OF SCOPE; confirm that TC-02 should be skipped/omitted so the "
                "remaining paths can still be executed sequentially."
            ),
            (
                "Confirmation that 'Universal team access' is enabled in the TEST "
                "environment for Path 8 (TC-08), since the test case precondition "
                "requires it."
            ),
        ],
        "setup": {
            "adb_serial": "emulator-5554",
            "app_package": "com.playerup.mobile",
            "app_activity": ".MainActivity",
            "adb_commands": [],
        },
        "entry_actions": [],
        "notes": [],
    }

    context_text = """
target_type: mobile_adb
adb_serial: emulator-5554
app_package: com.playerup.mobile
app_activity: .MainActivity
- Assume the user data provided is valid and use it literally
- Leave tests that require phone verification OUT OF SCOPE and don't include them
- Quick link to join team: https://links.playerup.co/7w5Qo0vzd1b
"""

    assessment = agent._parse_assessment(
        payload,
        "PlayerUp onboarding test plan",
        context_text,
    )

    assert assessment.sufficient is True
    assert not assessment.missing_items
    assert any("Non-blocking context gap:" in note for note in assessment.notes)


def test_parse_assessment_filters_deterministic_identifier_blockers() -> None:
    agent = SituationalAgent()
    payload = {
        "target_type": "desktop_app",
        "sufficient": False,
        "missing_items": [
            "Exact Linux window/app name for KeenBench (window title/task switcher)",
            "WM_CLASS or process name",
        ],
        "setup": {"app_name": "KeenBench"},
        "entry_actions": [],
        "notes": [],
    }

    assessment = agent._parse_assessment(
        payload,
        "Test KeenBench workflows",
        "Open KeenBench and verify workbench files",
    )

    assert assessment.sufficient is True
    assert not assessment.missing_items
    assert assessment.entry_actions


def test_parse_assessment_treats_desktop_missing_items_as_non_blocking() -> None:
    agent = SituationalAgent()
    payload = {
        "target_type": "desktop_app",
        "sufficient": False,
        "missing_items": [
            (
                "Confirmation that KeenBench is on the Home screen with a visible "
                "New Workbench button"
            )
        ],
        "setup": {"app_name": "KeenBench"},
        "entry_actions": [],
        "notes": [],
    }

    assessment = agent._parse_assessment(
        payload,
        "Test KeenBench workflows",
        "KeenBench is already open on the desktop",
    )

    assert assessment.sufficient is True
    assert not assessment.missing_items
    assert assessment.entry_actions
    assert any("Non-blocking context gap" in note for note in assessment.notes)


@pytest.mark.asyncio
async def test_assess_context_persists_model_and_debug_logs(
    monkeypatch, tmp_path
) -> None:
    debug_logger = MagicMock()
    monkeypatch.setattr(
        "haindy.agents.situational_agent.get_debug_logger", lambda: debug_logger
    )

    cache = SituationalCache(tmp_path / "situational_cache.json")
    agent = SituationalAgent(situational_cache=cache)
    agent.call_model = AsyncMock(
        return_value={
            "content": {
                "target_type": "desktop_app",
                "sufficient": True,
                "missing_items": [],
                "setup": {"app_name": "KeenBench"},
                "entry_actions": [],
                "notes": [],
            }
        }
    )

    assessment = await agent.assess_context(
        requirements="Validate KeenBench desktop workflow.",
        context_text="KeenBench is open in GNOME.",
    )

    assert assessment.sufficient is True
    call_kwargs = agent.call_model.await_args.kwargs
    assert call_kwargs["log_agent"] == "situational.assessment"
    assert call_kwargs["log_metadata"] == {"phase": "situational_assessment"}
    user_prompt = call_kwargs["messages"][1]["content"]
    assert "REQUIREMENTS:" in user_prompt
    assert "EXECUTION CONTEXT:" in user_prompt

    debug_logger.log_ai_interaction.assert_called_once()
    debug_kwargs = debug_logger.log_ai_interaction.call_args.kwargs
    assert debug_kwargs["action_type"] == "situational_assessment"
    assert "REQUIREMENTS:" in debug_kwargs["prompt"]


@pytest.mark.asyncio
async def test_assess_context_logs_fallback_when_model_call_fails(
    monkeypatch, tmp_path
) -> None:
    debug_logger = MagicMock()
    monkeypatch.setattr(
        "haindy.agents.situational_agent.get_debug_logger", lambda: debug_logger
    )

    cache = SituationalCache(tmp_path / "situational_cache.json")
    agent = SituationalAgent(situational_cache=cache)
    agent.call_model = AsyncMock(side_effect=RuntimeError("simulated API failure"))

    assessment = await agent.assess_context(
        requirements="Target desktop app flow.",
        context_text="app_name: Calculator",
    )

    assert assessment.target_type == "desktop_app"
    assert assessment.sufficient is True

    call_kwargs = agent.call_model.await_args.kwargs
    assert call_kwargs["log_agent"] == "situational.assessment"
    assert call_kwargs["log_metadata"] == {"phase": "situational_assessment"}

    debug_logger.log_ai_interaction.assert_called_once()
    debug_response = debug_logger.log_ai_interaction.call_args.kwargs["response"]
    assert "Fallback note" in debug_response


@pytest.mark.asyncio
async def test_assess_context_cache_hit_skips_model_call(tmp_path) -> None:
    cache = SituationalCache(tmp_path / "situational_cache.json")
    agent = SituationalAgent(situational_cache=cache)
    agent.call_model = AsyncMock(
        return_value={
            "content": {
                "target_type": "desktop_app",
                "sufficient": True,
                "missing_items": [],
                "setup": {"app_name": "KeenBench"},
                "entry_actions": [],
                "notes": [],
            }
        }
    )

    requirements = "Validate KeenBench desktop workflow."
    context_text = "KeenBench is open in GNOME."

    first = await agent.assess_context(
        requirements=requirements, context_text=context_text
    )
    second = await agent.assess_context(
        requirements=requirements,
        context_text=context_text,
    )

    assert first.sufficient is True
    assert second.sufficient is True
    assert agent.call_model.await_count == 1


@pytest.mark.asyncio
async def test_assess_context_cache_replays_insufficient_assessment(tmp_path) -> None:
    cache = SituationalCache(tmp_path / "situational_cache.json")
    agent = SituationalAgent(situational_cache=cache)
    agent.call_model = AsyncMock(
        return_value={
            "content": {
                "target_type": "web",
                "sufficient": False,
                "missing_items": ["web_url"],
                "setup": {"web_url": ""},
                "entry_actions": [],
                "notes": ["Need a URL"],
            }
        }
    )

    requirements = "Validate the web login flow."
    context_text = "Target type: web"

    first = await agent.assess_context(
        requirements=requirements, context_text=context_text
    )
    second = await agent.assess_context(
        requirements=requirements,
        context_text=context_text,
    )

    assert first.sufficient is False
    assert second.sufficient is False
    assert second.missing_items == ["web_url"]
    assert agent.call_model.await_count == 1


@pytest.mark.asyncio
async def test_assess_context_does_not_cache_fallback_only_results(tmp_path) -> None:
    cache = SituationalCache(tmp_path / "situational_cache.json")
    agent = SituationalAgent(situational_cache=cache)
    agent.call_model = AsyncMock(side_effect=RuntimeError("simulated API failure"))

    requirements = "Target desktop app flow."
    context_text = "app_name: Calculator"

    first = await agent.assess_context(
        requirements=requirements, context_text=context_text
    )
    second = await agent.assess_context(
        requirements=requirements,
        context_text=context_text,
    )

    assert first.target_type == "desktop_app"
    assert second.target_type == "desktop_app"
    assert agent.call_model.await_count == 2


@pytest.mark.asyncio
async def test_assess_context_invalid_cached_payload_falls_back_to_model(
    tmp_path,
) -> None:
    cache = SituationalCache(tmp_path / "situational_cache.json")
    agent = SituationalAgent(situational_cache=cache)
    agent.call_model = AsyncMock(
        return_value={
            "content": {
                "target_type": "desktop_app",
                "sufficient": True,
                "missing_items": [],
                "setup": {"app_name": "Calculator"},
                "entry_actions": [],
                "notes": [],
            }
        }
    )

    requirements = "Target desktop app flow."
    context_text = "app_name: Calculator"

    cache_key_payload = {
        "requirements": requirements,
        "context_text": context_text,
        "situational_signature": {
            "name": agent.name,
            "model": agent.model,
            "temperature": agent.temperature,
            "reasoning_level": agent.reasoning_level,
            "modalities": sorted(str(item) for item in agent.modalities),
            "system_prompt_sha256": sha256(
                (agent.system_prompt or "").encode("utf-8")
            ).hexdigest(),
        },
    }

    key_hash = hash_situational_cache_key(
        build_situational_cache_key_payload(**cache_key_payload)
    )
    cache.store(
        key_hash=key_hash,
        assessment_payload={"target_type": "invalid"},
    )

    assessment = await agent.assess_context(
        requirements=requirements,
        context_text=context_text,
    )

    assert assessment.sufficient is True
    assert agent.call_model.await_count == 1


@pytest.mark.asyncio
async def test_prepare_entrypoint_mobile_uses_mobile_hooks() -> None:
    agent = SituationalAgent()

    class DriverStub:
        def __init__(self) -> None:
            self.start = AsyncMock()
            self.configure_target = AsyncMock()
            self.run_adb_commands = AsyncMock()
            self.launch_app = AsyncMock()
            self.screenshot = AsyncMock(return_value=b"png")

    driver = DriverStub()
    setup = type(
        "Setup",
        (),
        {
            "web_url": "",
            "app_name": "",
            "launch_command": "",
            "maximize": True,
            "adb_serial": "emulator-5554",
            "app_package": "com.example.app",
            "app_activity": "com.example.app.MainActivity",
            "adb_commands": ["adb devices"],
        },
    )()
    assessment = type(
        "Assessment",
        (),
        {"target_type": "mobile_adb", "setup": setup, "entry_actions": []},
    )()

    await agent.prepare_entrypoint(driver, assessment, action_agent=AsyncMock())

    driver.configure_target.assert_awaited_once()
    driver.start.assert_awaited_once()
    driver.run_adb_commands.assert_awaited_once()
    driver.screenshot.assert_awaited_once()
