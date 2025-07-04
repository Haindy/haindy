"""
Tests for the pattern matcher.
"""

from uuid import uuid4

import pytest

from src.journal.models import ActionRecord, PatternType
from src.journal.pattern_matcher import PatternMatcher


@pytest.fixture
def pattern_matcher():
    """Create a PatternMatcher instance for testing."""
    return PatternMatcher()


@pytest.fixture
def sample_pattern():
    """Create a sample action pattern."""
    return ActionRecord(
        pattern_type=PatternType.CLICK,
        visual_signature={
            "action_type": ActionType.CLICK,
            "description": "click the login button",
            "grid_coordinates": {
                "initial_selection": "M23",
                "initial_confidence": 0.7,
                "refined_coordinates": "M23+offset(0.7,0.4)",
                "final_confidence": 0.95
            }
        },
        playwright_command="await page.click('#login-btn')",
        selectors={"primary": "#login-btn"},
        element_text="Login",
        element_type="button",
        url_pattern="https://example.com/login",
        success_count=10,
        failure_count=1,
        avg_execution_time_ms=250.0
    )


class TestPatternMatcher:
    """Test cases for PatternMatcher."""
    
    def test_exact_match(self, pattern_matcher, sample_pattern):
        """Test exact pattern matching."""
        features = {
            "action_type": ActionType.CLICK,
            "description": "click the login button",
            "element_text": "Login",
            "element_type": "button",
            "url_pattern": "https://example.com/login"
        }
        
        match = pattern_matcher.match_pattern(sample_pattern, features)
        
        assert match is not None
        assert match.confidence >= 0.95
        assert match.match_type == "exact"
        assert match.pattern_id == sample_pattern.record_id
    
    def test_similar_match(self, pattern_matcher, sample_pattern):
        """Test similar pattern matching."""
        features = {
            "action_type": ActionType.CLICK,
            "description": "click login",  # Slightly different
            "element_text": "Login",
            "element_type": "button",
            "url_pattern": "https://example.com/login"
        }
        
        match = pattern_matcher.match_pattern(sample_pattern, features)
        
        assert match is not None
        assert 0.8 <= match.confidence < 0.95
        assert match.match_type == "similar"
    
    def test_partial_match(self, pattern_matcher, sample_pattern):
        """Test partial pattern matching."""
        features = {
            "action_type": ActionType.CLICK,
            "description": "press button",  # Different description
            "element_type": "button",
            # Missing element_text and url_pattern
        }
        
        match = pattern_matcher.match_pattern(sample_pattern, features)
        
        assert match is not None
        assert 0.7 <= match.confidence < 0.8
        assert match.match_type == "partial"
    
    def test_no_match_different_action_type(self, pattern_matcher, sample_pattern):
        """Test no match for different action types."""
        features = {
            "action_type": ActionType.TYPE,  # Different action
            "description": "click the login button",
            "element_text": "Login"
        }
        
        match = pattern_matcher.match_pattern(sample_pattern, features)
        assert match is None
    
    def test_no_match_low_confidence(self, pattern_matcher, sample_pattern):
        """Test no match for low confidence."""
        features = {
            "action_type": ActionType.CLICK,
            "description": "do something completely different",
            # No other matching features
        }
        
        match = pattern_matcher.match_pattern(sample_pattern, features)
        assert match is None
    
    def test_text_similarity(self, pattern_matcher):
        """Test text similarity calculation."""
        # Exact match
        assert pattern_matcher._text_similarity("hello world", "hello world") == 1.0
        
        # Partial match
        sim = pattern_matcher._text_similarity("click the button", "click button")
        assert 0.5 < sim < 1.0
        
        # No match
        assert pattern_matcher._text_similarity("hello", "goodbye") == 0.0
        
        # Empty strings
        assert pattern_matcher._text_similarity("", "text") == 0.0
        assert pattern_matcher._text_similarity("text", "") == 0.0
    
    def test_url_similarity(self, pattern_matcher):
        """Test URL similarity calculation."""
        # Exact match
        assert pattern_matcher._url_similarity(
            "https://example.com/login",
            "https://example.com/login"
        ) == 1.0
        
        # Same domain, different path
        sim = pattern_matcher._url_similarity(
            "https://example.com/login",
            "https://example.com/signin"
        )
        assert 0.5 < sim < 1.0
        
        # Different domain
        assert pattern_matcher._url_similarity(
            "https://example.com/login",
            "https://other.com/login"
        ) == 0.0
        
        # Regex pattern
        assert pattern_matcher._url_similarity(
            "^https://example.com/.*",
            "https://example.com/anything"
        ) == 0.9
    
    def test_visual_similarity(self, pattern_matcher):
        """Test visual similarity based on grid coordinates."""
        # Same cell
        coords1 = {"initial_selection": "M23"}
        coords2 = {"initial_selection": "M23"}
        assert pattern_matcher._visual_similarity(coords1, coords2) == 1.0
        
        # Adjacent cells
        coords1 = {"initial_selection": "M23"}
        coords2 = {"initial_selection": "N23"}
        assert pattern_matcher._visual_similarity(coords1, coords2) == 0.8
        
        # Nearby cells
        coords1 = {"initial_selection": "M23"}
        coords2 = {"initial_selection": "O25"}
        assert pattern_matcher._visual_similarity(coords1, coords2) == 0.5
        
        # Far cells
        coords1 = {"initial_selection": "A1"}
        coords2 = {"initial_selection": "Z50"}
        assert pattern_matcher._visual_similarity(coords1, coords2) == 0.0
    
    def test_rank_patterns(self, pattern_matcher):
        """Test ranking multiple patterns."""
        patterns = [
            ActionRecord(
                pattern_type=PatternType.CLICK,
                visual_signature={"action_type": ActionType.CLICK, "description": "click submit"},
                playwright_command="cmd1"
            ),
            ActionRecord(
                pattern_type=PatternType.CLICK,
                visual_signature={"action_type": ActionType.CLICK, "description": "click the submit button"},
                playwright_command="cmd2"
            ),
            ActionRecord(
                pattern_type=PatternType.CLICK,
                visual_signature={"action_type": ActionType.CLICK, "description": "press enter"},
                playwright_command="cmd3"
            )
        ]
        
        features = {
            "action_type": ActionType.CLICK,
            "description": "click submit button"
        }
        
        ranked = pattern_matcher.rank_patterns(patterns, features)
        
        assert len(ranked) >= 2  # At least 2 should match
        assert ranked[0][1].confidence > ranked[1][1].confidence  # Descending order
        
        # Best match should be the one with most similar description
        best_pattern, best_match = ranked[0]
        assert "submit" in best_pattern.visual_signature["description"]
    
    def test_update_pattern_performance(self, pattern_matcher, sample_pattern):
        """Test updating pattern performance metrics."""
        initial_success = sample_pattern.success_count
        initial_avg_time = sample_pattern.avg_execution_time_ms
        
        # Update with success
        pattern_matcher.update_pattern_performance(sample_pattern, True, 300)
        
        assert sample_pattern.success_count == initial_success + 1
        assert sample_pattern.failure_count == 1  # Unchanged
        assert sample_pattern.last_used is not None
        
        # Average time should be updated
        expected_avg = (initial_avg_time * (initial_success + 1) + 300) / (initial_success + 2)
        assert abs(sample_pattern.avg_execution_time_ms - expected_avg) < 0.01
        
        # Update with failure
        pattern_matcher.update_pattern_performance(sample_pattern, False, 500)
        
        assert sample_pattern.success_count == initial_success + 1
        assert sample_pattern.failure_count == 2
    
    def test_extract_domain_and_path(self, pattern_matcher):
        """Test URL parsing helpers."""
        url = "https://example.com/path/to/page?query=1"
        
        assert pattern_matcher._extract_domain(url) == "example.com"
        assert pattern_matcher._extract_path(url) == "/path/to/page"
        
        # Invalid URL
        assert pattern_matcher._extract_domain("not-a-url") == ""
        assert pattern_matcher._extract_path("not-a-url") == ""


# Import ActionType for the tests
from src.core.types import ActionType