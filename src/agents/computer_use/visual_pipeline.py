"""Visual-state selection using session-owned cartography state."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .visual_state import (
    CartographyMap,
    CartographyTarget,
    VisualBounds,
    VisualFrame,
    build_keyframe,
    build_patch,
    compute_diff_bounds,
    expand_bounds,
    union_bounds,
)

logger = logging.getLogger("src.agents.computer_use.session")

_KEYFRAME_ACTIONS = {
    "scroll",
    "scroll_at",
    "scroll_document",
    "navigate",
    "search",
    "go_back",
    "go_forward",
}


@dataclass(frozen=True)
class VisualPlanResult:
    """Planned visual follow-up state for one provider turn."""

    visual_frame: VisualFrame
    current_keyframe: VisualFrame
    request_localization: bool = False
    localization_reason: str | None = None


class VisualStatePlanner:
    """Select keyframes versus patches using session-local cartography."""

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
        turns_since_cartography_refresh: int,
        cartography: CartographyMap | None,
    ) -> VisualPlanResult:
        """Return the frame to send plus the current full keyframe state."""
        target_text = str(metadata.get("target") or "").strip()
        interaction_mode = str(metadata.get("interaction_mode") or "").strip().lower()
        matched_target = self._match_target(cartography, metadata)
        localization_reason = self._resolve_localization_requirement(
            metadata=metadata,
            action_types=action_types,
            cartography=cartography,
            matched_target=matched_target,
            turns_since_cartography_refresh=turns_since_cartography_refresh,
        )
        carried_cartography = None if localization_reason else cartography
        current_full = build_keyframe(
            screenshot_bytes,
            source="follow_up_capture",
            cartography=carried_cartography,
        )

        if self.visual_mode != "keyframe_patch":
            self._log_visual_decision(
                decision="visual_mode_disabled",
                metadata=metadata,
                action_types=action_types,
                interaction_mode=interaction_mode,
                target_text=target_text,
                previous_keyframe=previous_keyframe,
                returned_frame=current_full,
                current_keyframe=current_full,
                turns_since_keyframe=turns_since_keyframe,
                cartography=carried_cartography,
                matched_target=matched_target,
                request_localization=localization_reason is not None,
                localization_reason=localization_reason,
            )
            return VisualPlanResult(
                visual_frame=current_full,
                current_keyframe=current_full,
                request_localization=localization_reason is not None,
                localization_reason=localization_reason,
            )

        force_keyframe_reason = self._resolve_keyframe_requirement(
            metadata=metadata,
            action_types=action_types,
            previous_keyframe=previous_keyframe,
            turns_since_keyframe=turns_since_keyframe,
            cartography=carried_cartography,
        )
        if force_keyframe_reason is not None or localization_reason is not None:
            self._log_visual_decision(
                decision=force_keyframe_reason or "cartography_refresh_requested",
                metadata=metadata,
                action_types=action_types,
                interaction_mode=interaction_mode,
                target_text=target_text,
                previous_keyframe=previous_keyframe,
                returned_frame=current_full,
                current_keyframe=current_full,
                turns_since_keyframe=turns_since_keyframe,
                cartography=carried_cartography,
                matched_target=matched_target,
                request_localization=localization_reason is not None,
                localization_reason=localization_reason,
            )
            return VisualPlanResult(
                visual_frame=current_full,
                current_keyframe=current_full,
                request_localization=localization_reason is not None,
                localization_reason=localization_reason,
            )

        assert previous_keyframe is not None

        diff_bounds = compute_diff_bounds(
            previous_keyframe.image_bytes,
            screenshot_bytes,
        )
        target_bounds = matched_target.bounds if matched_target is not None else None
        selected_bounds = union_bounds(
            [bound for bound in (diff_bounds, target_bounds) if bound is not None]
        )
        if selected_bounds is None:
            self._log_visual_decision(
                decision="missing_diff_and_target_bounds",
                metadata=metadata,
                action_types=action_types,
                interaction_mode=interaction_mode,
                target_text=target_text,
                previous_keyframe=previous_keyframe,
                returned_frame=current_full,
                current_keyframe=current_full,
                turns_since_keyframe=turns_since_keyframe,
                diff_bounds=diff_bounds,
                matched_target=matched_target,
                cartography=carried_cartography,
            )
            return VisualPlanResult(
                visual_frame=current_full,
                current_keyframe=current_full,
            )

        expanded = expand_bounds(
            selected_bounds,
            screen_size=current_full.screen_size,
            margin_ratio=self.patch_margin_ratio,
        )
        patch_area_ratio = expanded.area / float(max(current_full.bounds.area, 1))
        if patch_area_ratio > self.patch_max_area_ratio:
            self._log_visual_decision(
                decision="patch_area_ratio_exceeded",
                metadata=metadata,
                action_types=action_types,
                interaction_mode=interaction_mode,
                target_text=target_text,
                previous_keyframe=previous_keyframe,
                returned_frame=current_full,
                current_keyframe=current_full,
                turns_since_keyframe=turns_since_keyframe,
                diff_bounds=diff_bounds,
                matched_target=matched_target,
                selected_bounds=selected_bounds,
                expanded_bounds=expanded,
                patch_area_ratio=patch_area_ratio,
                cartography=carried_cartography,
            )
            return VisualPlanResult(
                visual_frame=current_full,
                current_keyframe=current_full,
            )

        patch = build_patch(
            screenshot_bytes,
            source_frame=previous_keyframe,
            bounds=expanded,
            diff_bounds=diff_bounds,
            target_bounds=target_bounds,
            source="follow_up_patch",
        )
        self._log_visual_decision(
            decision="patch_selected",
            metadata=metadata,
            action_types=action_types,
            interaction_mode=interaction_mode,
            target_text=target_text,
            previous_keyframe=previous_keyframe,
            returned_frame=patch,
            current_keyframe=current_full,
            turns_since_keyframe=turns_since_keyframe,
            diff_bounds=diff_bounds,
            matched_target=matched_target,
            selected_bounds=selected_bounds,
            expanded_bounds=expanded,
            patch_area_ratio=patch_area_ratio,
            cartography=carried_cartography,
        )
        return VisualPlanResult(
            visual_frame=patch,
            current_keyframe=current_full,
        )

    def _resolve_keyframe_requirement(
        self,
        *,
        metadata: dict[str, Any],
        action_types: Sequence[str],
        previous_keyframe: VisualFrame | None,
        turns_since_keyframe: int,
        cartography: CartographyMap | None,
    ) -> str | None:
        if previous_keyframe is None:
            return "missing_previous_keyframe"
        if turns_since_keyframe >= self.keyframe_max_turns:
            return "keyframe_refresh_interval"
        if cartography is None:
            return "missing_session_cartography"
        if (
            str(metadata.get("interaction_mode") or "").strip().lower()
            == "observe_only"
        ):
            return "observe_only_mode"
        if any(
            str(action_type or "").strip().lower() in _KEYFRAME_ACTIONS
            for action_type in action_types
        ):
            return "action_requires_keyframe"
        return None

    def _resolve_localization_requirement(
        self,
        *,
        metadata: dict[str, Any],
        action_types: Sequence[str],
        cartography: CartographyMap | None,
        matched_target: CartographyTarget | None,
        turns_since_cartography_refresh: int,
    ) -> str | None:
        target_text = str(metadata.get("target") or "").strip()
        if not target_text:
            return None
        if cartography is None:
            return "missing_session_cartography"
        if turns_since_cartography_refresh >= self.keyframe_max_turns:
            return "cartography_refresh_interval"
        if any(
            str(action_type or "").strip().lower() in _KEYFRAME_ACTIONS
            for action_type in action_types
        ):
            return "navigation_or_major_transition"
        if matched_target is None:
            return "target_map_untrusted"
        return None

    @staticmethod
    def _match_target(
        cartography: CartographyMap | None,
        metadata: dict[str, Any],
    ) -> CartographyTarget | None:
        if cartography is None or not cartography.targets:
            return None
        target_text = str(metadata.get("target") or "").strip().lower()
        if not target_text:
            return None
        for target in cartography.targets:
            if not target.label:
                continue
            if target_text in target.label.lower():
                return target
        return None

    @staticmethod
    def _bounds_tuple(bounds: VisualBounds | None) -> tuple[int, int, int, int] | None:
        if bounds is None:
            return None
        return bounds.as_tuple()

    def _log_visual_decision(
        self,
        *,
        decision: str,
        metadata: dict[str, Any],
        action_types: Sequence[str],
        interaction_mode: str,
        target_text: str,
        previous_keyframe: VisualFrame | None,
        returned_frame: VisualFrame,
        current_keyframe: VisualFrame,
        turns_since_keyframe: int,
        diff_bounds: VisualBounds | None = None,
        matched_target: CartographyTarget | None = None,
        selected_bounds: VisualBounds | None = None,
        expanded_bounds: VisualBounds | None = None,
        patch_area_ratio: float | None = None,
        cartography: CartographyMap | None = None,
        request_localization: bool = False,
        localization_reason: str | None = None,
    ) -> None:
        previous_cartography = (
            previous_keyframe.cartography if previous_keyframe is not None else None
        )
        effective_cartography = cartography or previous_cartography
        labels = (
            [target.label for target in effective_cartography.targets if target.label]
            if effective_cartography is not None
            else []
        )
        logger.info(
            "Computer Use visual frame selected",
            extra={
                "visual_decision": decision,
                "visual_mode": self.visual_mode,
                "visual_frame_kind": returned_frame.kind,
                "visual_frame_id": returned_frame.frame_id,
                "current_keyframe_id": current_keyframe.frame_id,
                "previous_keyframe_id": (
                    previous_keyframe.frame_id
                    if previous_keyframe is not None
                    else None
                ),
                "turns_since_keyframe": turns_since_keyframe,
                "interaction_mode": interaction_mode,
                "action_types": list(action_types),
                "target": target_text or None,
                "previous_has_cartography": bool(
                    previous_cartography is not None and previous_cartography.targets
                ),
                "has_session_cartography": bool(
                    effective_cartography is not None and effective_cartography.targets
                ),
                "cartography_target_count": len(labels),
                "cartography_labels": labels or None,
                "cartography_refresh_requested": request_localization,
                "cartography_refresh_reason": localization_reason,
                "matched_target_label": (
                    matched_target.label if matched_target is not None else None
                ),
                "matched_target_bounds": (
                    self._bounds_tuple(matched_target.bounds)
                    if matched_target is not None
                    else None
                ),
                "diff_bounds": self._bounds_tuple(diff_bounds),
                "selected_bounds": self._bounds_tuple(selected_bounds),
                "expanded_bounds": self._bounds_tuple(expanded_bounds),
                "returned_frame_bounds": returned_frame.bounds.as_tuple(),
                "patch_area_ratio": (
                    round(patch_area_ratio, 6) if patch_area_ratio is not None else None
                ),
                "step_number": metadata.get("step_number"),
            },
        )


__all__ = ["VisualPlanResult", "VisualStatePlanner"]
