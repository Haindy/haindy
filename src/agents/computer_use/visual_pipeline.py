"""Visual-state selection and cartography orchestration for Computer Use."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .visual_state import (
    CartographyMap,
    VisualBounds,
    VisualFrame,
    build_keyframe,
    build_patch,
    compute_diff_bounds,
    expand_bounds,
    union_bounds,
)

CartographyGenerator = Callable[
    [VisualFrame, dict[str, Any]], Awaitable[CartographyMap | None]
]

_KEYFRAME_ACTIONS = {
    "scroll",
    "scroll_at",
    "scroll_document",
    "navigate",
    "search",
    "go_back",
    "go_forward",
}


class VisualStatePlanner:
    """Select keyframes versus patches and invoke provider-owned cartography."""

    def __init__(
        self,
        *,
        visual_mode: str,
        keyframe_max_turns: int,
        patch_max_area_ratio: float,
        patch_margin_ratio: float,
    ) -> None:
        self.visual_mode = str(visual_mode or "keyframe_patch").strip().lower()
        self.keyframe_max_turns = max(int(keyframe_max_turns), 1)
        self.patch_max_area_ratio = max(float(patch_max_area_ratio), 0.01)
        self.patch_margin_ratio = max(float(patch_margin_ratio), 0.0)

    async def build_follow_up_frame(
        self,
        *,
        screenshot_bytes: bytes,
        metadata: dict[str, Any],
        action_types: Sequence[str],
        previous_keyframe: VisualFrame | None,
        turns_since_keyframe: int,
        generate_cartography: CartographyGenerator,
    ) -> tuple[VisualFrame, VisualFrame]:
        """Return the frame to send plus the current full keyframe state."""
        current_full = build_keyframe(screenshot_bytes, source="follow_up_capture")

        if self.visual_mode != "keyframe_patch":
            cartography = await generate_cartography(current_full, metadata)
            if cartography is not None:
                current_full = build_keyframe(
                    screenshot_bytes,
                    source="follow_up_capture",
                    cartography=cartography,
                )
            return current_full, current_full

        force_keyframe = self._should_force_keyframe(
            metadata=metadata,
            action_types=action_types,
            previous_keyframe=previous_keyframe,
            turns_since_keyframe=turns_since_keyframe,
        )
        if force_keyframe or previous_keyframe is None:
            cartography = await generate_cartography(current_full, metadata)
            if cartography is not None:
                current_full = build_keyframe(
                    screenshot_bytes,
                    source="follow_up_capture",
                    cartography=cartography,
                )
            return current_full, current_full

        diff_bounds = compute_diff_bounds(
            previous_keyframe.image_bytes, screenshot_bytes
        )
        target_bounds = self._match_target_bounds(
            previous_keyframe.cartography, metadata
        )
        selected_bounds = union_bounds(
            [bound for bound in (diff_bounds, target_bounds) if bound is not None]
        )
        if selected_bounds is None:
            cartography = await generate_cartography(current_full, metadata)
            if cartography is not None:
                current_full = build_keyframe(
                    screenshot_bytes,
                    source="follow_up_capture",
                    cartography=cartography,
                )
            return current_full, current_full

        expanded = expand_bounds(
            selected_bounds,
            screen_size=current_full.screen_size,
            margin_ratio=self.patch_margin_ratio,
        )
        if (
            expanded.area / float(max(current_full.bounds.area, 1))
            > self.patch_max_area_ratio
        ):
            cartography = await generate_cartography(current_full, metadata)
            if cartography is not None:
                current_full = build_keyframe(
                    screenshot_bytes,
                    source="follow_up_capture",
                    cartography=cartography,
                )
            return current_full, current_full

        patch = build_patch(
            screenshot_bytes,
            source_frame=previous_keyframe,
            bounds=expanded,
            diff_bounds=diff_bounds,
            target_bounds=target_bounds,
            source="follow_up_patch",
        )
        return patch, current_full

    def _should_force_keyframe(
        self,
        *,
        metadata: dict[str, Any],
        action_types: Sequence[str],
        previous_keyframe: VisualFrame | None,
        turns_since_keyframe: int,
    ) -> bool:
        if previous_keyframe is None:
            return True
        if turns_since_keyframe >= self.keyframe_max_turns:
            return True
        if (
            str(metadata.get("interaction_mode") or "").strip().lower()
            == "observe_only"
        ):
            return True
        if any(
            str(action_type or "").strip().lower() in _KEYFRAME_ACTIONS
            for action_type in action_types
        ):
            return True
        return False

    @staticmethod
    def _match_target_bounds(
        cartography: CartographyMap | None,
        metadata: dict[str, Any],
    ) -> VisualBounds | None:
        if cartography is None or not cartography.targets:
            return None
        target_text = str(metadata.get("target") or "").strip().lower()
        if not target_text:
            return None
        for target in cartography.targets:
            if not target.label:
                continue
            if target_text in target.label.lower():
                return target.bounds
        return None


__all__ = ["CartographyGenerator", "VisualStatePlanner"]
