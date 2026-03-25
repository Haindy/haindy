"""Pattern matcher tests with provider-neutral coordinate metadata."""

from haindy.core.types import ActionType
from haindy.journal.models import ActionRecord, PatternType
from haindy.journal.pattern_matcher import PatternMatcher


def _pattern() -> ActionRecord:
    return ActionRecord(
        pattern_type=PatternType.CLICK,
        visual_signature={
            "action_type": ActionType.CLICK,
            "description": "click search button",
            "coordinate_metadata": {
                "target_reference": "search_button",
                "pixel_coordinates": (600, 300),
            },
        },
        automation_command="click('#search')",
        element_type="button",
        element_text="Search",
        url_pattern="https://example.com/search",
    )


def test_match_pattern_returns_match_for_similar_features() -> None:
    matcher = PatternMatcher()
    pattern = _pattern()

    match = matcher.match_pattern(
        pattern,
        {
            "action_type": ActionType.CLICK,
            "description": "click the search button",
            "element_type": "button",
            "element_text": "Search",
            "url_pattern": "https://example.com/search?q=test",
            "coordinate_metadata": {
                "target_reference": "search_button",
                "pixel_coordinates": (605, 302),
            },
        },
    )

    assert match is not None
    assert match.confidence >= 0.7


def test_rank_patterns_orders_by_confidence() -> None:
    matcher = PatternMatcher()
    strong = _pattern()
    weak = _pattern()
    weak.visual_signature["description"] = "open profile menu"

    ranked = matcher.rank_patterns(
        [weak, strong],
        {
            "action_type": ActionType.CLICK,
            "description": "click search button",
            "element_type": "button",
            "element_text": "Search",
            "url_pattern": "https://example.com/search",
            "coordinate_metadata": {
                "target_reference": "search_button",
                "pixel_coordinates": (600, 300),
            },
        },
    )

    assert ranked
    assert ranked[0][1].confidence >= ranked[-1][1].confidence


def test_update_pattern_performance_updates_counters() -> None:
    matcher = PatternMatcher()
    pattern = _pattern()

    matcher.update_pattern_performance(pattern, success=True, execution_time_ms=100)
    matcher.update_pattern_performance(pattern, success=False, execution_time_ms=300)

    assert pattern.success_count == 1
    assert pattern.failure_count == 1
    assert pattern.avg_execution_time_ms == 200.0
    assert pattern.last_used is not None
