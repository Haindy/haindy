"""
Pattern matching for action reuse and caching.
"""

import math
import re
from typing import Any

from haindy.journal.models import ActionRecord, PatternMatch
from haindy.monitoring.logger import get_logger

logger = get_logger(__name__)


class PatternMatcher:
    """
    Matches test actions against recorded patterns for reuse.

    Uses similarity scoring to find the best matching pattern
    from the pattern library.
    """

    def __init__(self) -> None:
        """Initialize the pattern matcher."""
        # Weights for different matching criteria
        self.weights = {
            "action_type": 0.3,
            "text_similarity": 0.25,
            "url_similarity": 0.2,
            "element_type": 0.15,
            "visual_signature": 0.1,
        }

    def match_pattern(
        self, pattern: ActionRecord, features: dict[str, Any]
    ) -> PatternMatch | None:
        """
        Match a pattern against given features.

        Args:
            pattern: The pattern to match against
            features: Features to match

        Returns:
            PatternMatch if confidence is above threshold
        """
        confidence = 0.0
        adjustments: dict[str, Any] = {}

        # Action type match (exact match required)
        if pattern.visual_signature.get("action_type") == features.get("action_type"):
            confidence += self.weights["action_type"]
        else:
            return None  # Different action types don't match

        # Text similarity
        pattern_desc = pattern.visual_signature.get("description", "").lower()
        feature_desc = features.get("description", "").lower()
        text_sim = self._text_similarity(pattern_desc, feature_desc)
        confidence += text_sim * self.weights["text_similarity"]

        # Element text match
        if pattern.element_text and features.get("element_text"):
            elem_sim = self._text_similarity(
                pattern.element_text.lower(), features["element_text"].lower()
            )
            confidence += elem_sim * self.weights["text_similarity"]

        # URL pattern match
        if pattern.url_pattern and features.get("url_pattern"):
            url_sim = self._url_similarity(pattern.url_pattern, features["url_pattern"])
            confidence += url_sim * self.weights["url_similarity"]

        # Element type match
        if pattern.element_type and features.get("element_type"):
            if pattern.element_type == features["element_type"]:
                confidence += self.weights["element_type"]

        # Visual signature similarity (coordinate metadata)
        if pattern.visual_signature.get("coordinate_metadata") and features.get(
            "coordinate_metadata"
        ):
            visual_sim = self._visual_similarity(
                pattern.visual_signature["coordinate_metadata"],
                features["coordinate_metadata"],
            )
            confidence += visual_sim * self.weights["visual_signature"]

        # Ensure confidence doesn't exceed 1.0
        confidence = min(confidence, 1.0)

        # Determine match type
        if confidence >= 0.95:
            match_type = "exact"
        elif confidence >= 0.8:
            match_type = "similar"
        elif confidence >= 0.7:
            match_type = "partial"
        else:
            return None

        return PatternMatch(
            pattern_id=pattern.record_id,
            confidence=confidence,
            match_type=match_type,
            adjustments=adjustments,
        )

    def rank_patterns(
        self, patterns: list[ActionRecord], features: dict[str, Any]
    ) -> list[tuple[ActionRecord, PatternMatch]]:
        """
        Rank patterns by match confidence.

        Args:
            patterns: List of patterns to rank
            features: Features to match against

        Returns:
            List of (pattern, match) tuples sorted by confidence
        """
        matches = []

        for pattern in patterns:
            match = self.match_pattern(pattern, features)
            if match:
                matches.append((pattern, match))

        # Sort by confidence descending
        matches.sort(key=lambda x: x[1].confidence, reverse=True)

        return matches

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings.

        Uses token-based similarity with normalization.
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize and normalize
        tokens1 = set(re.findall(r"\w+", text1.lower()))
        tokens2 = set(re.findall(r"\w+", text2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _url_similarity(self, pattern_url: str, target_url: str) -> float:
        """
        Calculate URL similarity.

        Handles both exact matches and pattern-based matches.
        """
        if pattern_url == target_url:
            return 1.0

        # Check if pattern is a regex
        if pattern_url.startswith("^") or pattern_url.endswith("$"):
            try:
                if re.match(pattern_url, target_url):
                    return 0.9
            except re.error:
                pass

        # Check domain match
        pattern_domain = self._extract_domain(pattern_url)
        target_domain = self._extract_domain(target_url)

        if pattern_domain == target_domain:
            # Same domain, check path similarity
            pattern_path = self._extract_path(pattern_url)
            target_path = self._extract_path(target_url)

            path_sim = self._text_similarity(pattern_path, target_path)
            return 0.5 + (0.5 * path_sim)  # Adjusted to allow up to 1.0

        return 0.0

    def _visual_similarity(
        self, coords1: dict[str, Any], coords2: dict[str, Any]
    ) -> float:
        """
        Calculate similarity for provider-neutral coordinate metadata.
        """
        ref1 = (coords1.get("target_reference") or "").strip().lower()
        ref2 = (coords2.get("target_reference") or "").strip().lower()

        if ref1 and ref2 and ref1 == ref2:
            return 1.0

        ref_similarity = self._text_similarity(ref1, ref2) if ref1 and ref2 else 0.0

        pixels1 = coords1.get("pixel_coordinates")
        pixels2 = coords2.get("pixel_coordinates")
        pixel_similarity = 0.0
        if (
            isinstance(pixels1, (list, tuple))
            and isinstance(pixels2, (list, tuple))
            and len(pixels1) == 2
            and len(pixels2) == 2
        ):
            dx = float(pixels1[0]) - float(pixels2[0])
            dy = float(pixels1[1]) - float(pixels2[1])
            distance = math.hypot(dx, dy)
            if distance <= 10:
                pixel_similarity = 1.0
            elif distance <= 40:
                pixel_similarity = 0.8
            elif distance <= 100:
                pixel_similarity = 0.5
            else:
                pixel_similarity = 0.0

        if ref_similarity and pixel_similarity:
            return 0.6 * ref_similarity + 0.4 * pixel_similarity
        return max(ref_similarity, pixel_similarity)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        import urllib.parse

        try:
            # Handle URLs without scheme
            if not url.startswith(("http://", "https://")):
                url = "http://" + url
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc or ""
        except Exception:
            return ""

    def _extract_path(self, url: str) -> str:
        """Extract path from URL."""
        import urllib.parse

        try:
            # Handle URLs without scheme
            if not url.startswith(("http://", "https://")):
                url = "http://" + url
            parsed = urllib.parse.urlparse(url)
            return parsed.path or ""
        except Exception:
            return ""

    def update_pattern_performance(
        self, pattern: ActionRecord, success: bool, execution_time_ms: int
    ) -> None:
        """
        Update pattern performance metrics.

        Args:
            pattern: Pattern to update
            success: Whether execution was successful
            execution_time_ms: Execution time
        """
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1

        # Update average execution time
        total_uses = pattern.success_count + pattern.failure_count
        if total_uses > 0:
            # Weighted average
            pattern.avg_execution_time_ms = (
                pattern.avg_execution_time_ms * (total_uses - 1) + execution_time_ms
            ) / total_uses

        from datetime import datetime, timezone

        pattern.last_used = datetime.now(timezone.utc)
