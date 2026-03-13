"""Visual-state planner and Google patch-coordinate tests."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw

from src.agents.computer_use.common import denormalize_coordinates
from src.agents.computer_use.visual_pipeline import VisualStatePlanner
from src.agents.computer_use.visual_state import (
    CartographyMap,
    CartographyTarget,
    VisualBounds,
    VisualFrame,
    build_keyframe,
    compute_diff_bounds,
    encode_png,
    expand_bounds,
)
from src.core.enhanced_types import ComputerToolTurn
from tests.computer_use_session_support import make_session

pytest_plugins = ("tests.computer_use_session_support",)


def _png_with_rect(
    *,
    width: int,
    height: int,
    rect: tuple[int, int, int, int] | None = None,
) -> bytes:
    image = Image.new("RGB", (width, height), color="black")
    if rect is not None:
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = rect
        draw.rectangle((left, top, right - 1, bottom - 1), fill="white")
    return encode_png(image)


def _patch_frame() -> VisualFrame:
    return VisualFrame(
        frame_id="patch_1",
        kind="patch",
        image_bytes=b"patch_png",
        screen_size=(1000, 800),
        bounds=VisualBounds(x=100, y=200, width=300, height=200),
        parent_keyframe_id="keyframe_1",
    )


def test_compute_diff_bounds_detects_changed_region() -> None:
    previous = _png_with_rect(width=100, height=100)
    current = _png_with_rect(width=100, height=100, rect=(20, 30, 40, 50))

    assert compute_diff_bounds(previous, current) == VisualBounds(
        x=20,
        y=30,
        width=20,
        height=20,
    )


def test_expand_bounds_clamps_to_screen() -> None:
    expanded = expand_bounds(
        VisualBounds(x=5, y=5, width=10, height=10),
        screen_size=(50, 50),
        margin_ratio=0.5,
    )

    assert expanded == VisualBounds(x=0, y=0, width=39, height=39)


@pytest.mark.asyncio
async def test_visual_state_planner_uses_target_aware_patch_when_cartography_matches():
    screenshot = _png_with_rect(width=200, height=120)
    cartography = CartographyMap(
        frame_id="vk_prev",
        targets=(
            CartographyTarget(
                target_id="target_1",
                label="submit button",
                bounds=VisualBounds(x=40, y=30, width=30, height=20),
                interaction_point=(55, 40),
                confidence=0.98,
            ),
        ),
        model="test-model",
        provider="google",
    )
    previous_keyframe = build_keyframe(
        screenshot,
        source="test",
        cartography=cartography,
    )
    planner = VisualStatePlanner(
        visual_mode="keyframe_patch",
        keyframe_max_turns=3,
        patch_max_area_ratio=0.35,
        patch_margin_ratio=0.12,
    )
    generate_cartography = AsyncMock(return_value=None)

    visual_frame, current_keyframe = await planner.build_follow_up_frame(
        screenshot_bytes=screenshot,
        metadata={"target": "submit"},
        action_types=["click_at"],
        previous_keyframe=previous_keyframe,
        turns_since_keyframe=1,
        generate_cartography=generate_cartography,
    )

    assert visual_frame.kind == "patch"
    assert visual_frame.target_bounds == VisualBounds(x=40, y=30, width=30, height=20)
    assert current_keyframe.kind == "keyframe"
    generate_cartography.assert_not_awaited()


@pytest.mark.asyncio
async def test_visual_state_planner_logs_selected_patch_context(caplog) -> None:
    screenshot = _png_with_rect(width=200, height=120)
    cartography = CartographyMap(
        frame_id="vk_prev",
        targets=(
            CartographyTarget(
                target_id="target_1",
                label="email",
                bounds=VisualBounds(x=20, y=10, width=80, height=30),
                interaction_point=(60, 25),
                confidence=0.95,
            ),
        ),
        model="test-model",
        provider="google",
    )
    previous_keyframe = build_keyframe(
        screenshot,
        source="test",
        cartography=cartography,
    )
    planner = VisualStatePlanner(
        visual_mode="keyframe_patch",
        keyframe_max_turns=3,
        patch_max_area_ratio=0.35,
        patch_margin_ratio=0.12,
    )
    generate_cartography = AsyncMock(return_value=None)

    with caplog.at_level(logging.INFO, logger="src.agents.computer_use.session"):
        await planner.build_follow_up_frame(
            screenshot_bytes=screenshot,
            metadata={"target": "email"},
            action_types=["click_at"],
            previous_keyframe=previous_keyframe,
            turns_since_keyframe=1,
            generate_cartography=generate_cartography,
        )

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Computer Use visual frame selected"
    )
    assert record.visual_decision == "patch_selected"
    assert record.visual_frame_kind == "patch"
    assert record.matched_target_label == "email"
    assert record.cartography_labels == ["email"]


@pytest.mark.asyncio
async def test_visual_state_planner_forces_keyframe_when_requested_by_metadata():
    screenshot = _png_with_rect(width=200, height=120)
    previous_keyframe = build_keyframe(screenshot, source="test")
    cartography = CartographyMap(
        frame_id="vk_current",
        targets=(
            CartographyTarget(
                target_id="target_1",
                label="submit button",
                bounds=VisualBounds(x=40, y=30, width=30, height=20),
                interaction_point=(55, 40),
                confidence=0.98,
            ),
        ),
        model="test-model",
        provider="openai",
    )
    planner = VisualStatePlanner(
        visual_mode="keyframe_patch",
        keyframe_max_turns=3,
        patch_max_area_ratio=0.35,
        patch_margin_ratio=0.12,
    )
    generate_cartography = AsyncMock(return_value=cartography)

    visual_frame, current_keyframe = await planner.build_follow_up_frame(
        screenshot_bytes=screenshot,
        metadata={
            "target": "submit",
            "_force_keyframe_reason": "mobile_keyboard_or_focus_reflow",
        },
        action_types=["click"],
        previous_keyframe=previous_keyframe,
        turns_since_keyframe=1,
        generate_cartography=generate_cartography,
    )

    assert visual_frame.kind == "keyframe"
    assert current_keyframe.kind == "keyframe"
    assert current_keyframe.cartography == cartography
    generate_cartography.assert_awaited_once()


@pytest.mark.asyncio
async def test_initial_screenshot_seeding_enables_patch_on_first_follow_up(
    mock_client, mock_browser, session_settings
) -> None:
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    initial_screenshot = _png_with_rect(width=200, height=120)
    follow_up_screenshot = _png_with_rect(
        width=200,
        height=120,
        rect=(40, 30, 70, 50),
    )
    mock_browser.screenshot.return_value = follow_up_screenshot
    turn = ComputerToolTurn(
        call_id="call_seeded_patch",
        action_type="click",
        parameters={"type": "click"},
    )

    session._maybe_seed_initial_keyframe(initial_screenshot)
    batch = await session._build_follow_up_batch(call_groups=[[turn]], metadata={})

    assert batch.visual_frame is not None
    assert batch.visual_frame.kind == "patch"
    assert session._current_keyframe is not None
    assert session._current_keyframe.kind == "keyframe"


def test_initial_screenshot_seeding_does_not_override_existing_keyframe(
    mock_client, mock_browser, session_settings
) -> None:
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    existing = build_keyframe(_png_with_rect(width=100, height=100), source="existing")
    session._current_keyframe = existing

    session._maybe_seed_initial_keyframe(_png_with_rect(width=100, height=100))

    assert session._current_keyframe is existing


@pytest.mark.asyncio
async def test_google_patch_click_coordinates_remap_to_full_screen(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    mock_browser.get_viewport_size.return_value = (1000, 800)
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    session._last_visual_frame = _patch_frame()
    turn = ComputerToolTurn(
        call_id="call_patch_click",
        action_type="click_at",
        parameters={"x": 500, "y": 500},
    )

    await session._execute_tool_action(
        turn=turn,
        metadata={},
        turn_index=1,
        normalized_coords=True,
        environment="desktop",
    )

    mock_browser.click.assert_awaited_once_with(
        250,
        300,
        button="left",
        click_count=1,
    )
    assert turn.status == "executed"
    assert turn.metadata["coordinate_frame_kind"] == "patch"
    assert turn.metadata["patch_bounds"] == (100, 200, 300, 200)
    assert turn.metadata["patch_coordinate"] == (150, 100)
    assert turn.metadata["full_screen_coordinate"] == (250, 300)


@pytest.mark.asyncio
async def test_google_patch_drag_coordinates_remap_both_endpoints(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    mock_browser.get_viewport_size.return_value = (1000, 800)
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    session._last_visual_frame = _patch_frame()
    turn = ComputerToolTurn(
        call_id="call_patch_drag",
        action_type="drag",
        parameters={"start_x": 0, "start_y": 0, "end_x": 999, "end_y": 999},
    )

    await session._execute_tool_action(
        turn=turn,
        metadata={},
        turn_index=1,
        normalized_coords=True,
        environment="desktop",
    )

    mock_browser.drag_mouse.assert_awaited_once_with(100, 200, 399, 399, steps=1)
    assert turn.status == "executed"
    assert turn.metadata["start_coordinate_frame_kind"] == "patch"
    assert turn.metadata["start_patch_coordinate"] == (0, 0)
    assert turn.metadata["start_full_screen_coordinate"] == (100, 200)
    assert turn.metadata["end_coordinate_frame_kind"] == "patch"
    assert turn.metadata["end_patch_coordinate"] == (299, 199)
    assert turn.metadata["end_full_screen_coordinate"] == (399, 399)


def test_keyframe_coordinates_still_use_full_viewport_normalization(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    session._last_visual_frame = VisualFrame(
        frame_id="keyframe_1",
        kind="keyframe",
        image_bytes=b"full_png",
        screen_size=(1000, 800),
        bounds=VisualBounds(x=0, y=0, width=1000, height=800),
    )
    turn = ComputerToolTurn(call_id="call_keyframe", action_type="click_at")

    resolved = session._denormalize_coordinates_for_active_frame(
        500,
        500,
        1000,
        800,
        turn=turn,
    )

    assert resolved == denormalize_coordinates(500, 500, 1000, 800)
    assert turn.metadata["coordinate_frame_kind"] == "keyframe"
