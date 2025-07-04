"""
Tests for the script recorder.
"""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.core.types import ActionType
from src.journal.models import ActionRecord, JournalActionResult, PatternType, ScriptedCommand
from src.journal.script_recorder import ScriptRecorder


@pytest.fixture
def script_recorder():
    """Create a ScriptRecorder instance for testing."""
    return ScriptRecorder()


@pytest.fixture
def click_action_result():
    """Create a sample click action result."""
    return JournalActionResult(
        success=True,
        action=ActionType.CLICK,
        confidence=0.95,
        coordinates=(500, 300),
        playwright_command="await page.click('#submit-btn')",
        actual_outcome="Button clicked"
    )


@pytest.fixture
def type_action_result():
    """Create a sample type action result."""
    return JournalActionResult(
        success=True,
        action=ActionType.TYPE,
        confidence=0.9,
        input_text="test@example.com",
        playwright_command="await page.fill('#email', 'test@example.com')",
        actual_outcome="Text entered"
    )


class TestScriptRecorder:
    """Test cases for ScriptRecorder."""
    
    def test_record_click_action(self, script_recorder, click_action_result):
        """Test recording a click action."""
        element_info = {
            "tag": "button",
            "id": "submit-btn",
            "class": "btn btn-primary",
            "text": "Submit",
            "role": "button",
            "data-testid": "submit-button"
        }
        
        command = script_recorder.record_action(
            ActionType.CLICK,
            click_action_result,
            element_info
        )
        
        assert command is not None
        assert command.command_type == "click"
        assert command.command == click_action_result.playwright_command
        assert len(command.selectors) > 0
        assert '[data-testid="submit-button"]' in command.selectors
        assert '#submit-btn' in command.selectors
    
    def test_record_type_action(self, script_recorder, type_action_result):
        """Test recording a type action."""
        element_info = {
            "tag": "input",
            "id": "email",
            "type": "email",
            "aria-label": "Email address"
        }
        
        command = script_recorder.record_action(
            ActionType.TYPE,
            type_action_result,
            element_info
        )
        
        assert command is not None
        assert command.command_type == "type"
        assert "test@example.com" in command.command
        assert command.parameters["text"] == "test@example.com"
    
    def test_record_navigate_action(self, script_recorder):
        """Test recording a navigate action."""
        result = JournalActionResult(
            success=True,
            action=ActionType.NAVIGATE,
            confidence=1.0
        )
        
        element_info = {"url": "https://example.com/login"}
        
        command = script_recorder.record_action(
            ActionType.NAVIGATE,
            result,
            element_info
        )
        
        assert command is not None
        assert command.command_type == "navigate"
        assert "https://example.com/login" in command.command
        assert command.parameters["url"] == "https://example.com/login"
    
    def test_record_failed_action(self, script_recorder):
        """Test that failed actions are not recorded."""
        result = JournalActionResult(
            success=False,
            action=ActionType.CLICK,
            error_message="Element not found"
        )
        
        command = script_recorder.record_action(
            ActionType.CLICK,
            result,
            {}
        )
        
        assert command is None
    
    def test_generate_selectors_priority(self, script_recorder):
        """Test selector generation priority."""
        element_info = {
            "data-testid": "login-btn",
            "id": "loginButton",
            "tag": "button",
            "text": "Log In",
            "class": "btn primary",
            "role": "button",
            "aria-label": "Login to your account"
        }
        
        selectors = script_recorder.generate_selectors(element_info)
        
        # data-testid should be first priority
        assert selectors[0] == '[data-testid="login-btn"]'
        # ID should be second
        assert '#loginButton' in selectors[:2]
        # Other selectors should be present
        assert any('button:has-text("Log In")' in s for s in selectors)
        assert any('role=button' in s for s in selectors)
    
    def test_generate_selectors_minimal_info(self, script_recorder):
        """Test selector generation with minimal element info."""
        element_info = {
            "tag": "div",
            "text": "Click me"
        }
        
        selectors = script_recorder.generate_selectors(element_info)
        
        assert len(selectors) > 0
        assert 'div:has-text("Click me")' in selectors
    
    def test_create_fallback_scripts(self, script_recorder):
        """Test creating fallback scripts."""
        pattern = ActionRecord(
            pattern_type=PatternType.CLICK,
            visual_signature={},
            playwright_command="await page.click('#btn')",
            selectors={
                "primary": "#btn",
                "secondary": "button.submit",
                "tertiary": "button[type=submit]"
            }
        )
        
        fallbacks = script_recorder.create_fallback_script(pattern)
        
        assert len(fallbacks) > 0
        # Should include wait strategy
        assert any("wait_for_load_state" in fb for fb in fallbacks)
        # Should include force click
        assert any("force=True" in fb for fb in fallbacks)
    
    def test_record_scroll_action(self, script_recorder):
        """Test recording a scroll action."""
        result = JournalActionResult(
            success=True,
            action=ActionType.SCROLL,
            confidence=1.0
        )
        
        element_info = {
            "direction": "down",
            "amount": 500
        }
        
        command = script_recorder.record_action(
            ActionType.SCROLL,
            result,
            element_info
        )
        
        assert command is not None
        assert command.command_type == "scroll"
        assert "mouse.wheel" in command.command
        assert "500" in command.command
    
    def test_record_wait_action(self, script_recorder):
        """Test recording a wait action."""
        result = JournalActionResult(
            success=True,
            action=ActionType.WAIT,
            confidence=1.0
        )
        
        element_info = {"duration": 2000}
        
        command = script_recorder.record_action(
            ActionType.WAIT,
            result,
            element_info
        )
        
        assert command is not None
        assert command.command_type == "wait"
        assert "wait_for_timeout(2000)" in command.command
    
    def test_text_selector_escaping(self, script_recorder):
        """Test that text in selectors is properly escaped."""
        element_info = {
            "tag": "button",
            "text": 'Click "here" to continue'
        }
        
        selectors = script_recorder.generate_selectors(element_info)
        text_selector = next((s for s in selectors if ":has-text" in s), None)
        
        assert text_selector is not None
        # Should escape quotes properly
        assert '\\"' in text_selector
    
    def test_xpath_selector_generation(self, script_recorder):
        """Test XPath selector generation as fallback."""
        element_info = {
            "tag": "span",
            "text": "Special Text"
        }
        
        selectors = script_recorder.generate_selectors(element_info)
        xpath_selector = next((s for s in selectors if s.startswith("//")), None)
        
        assert xpath_selector is not None
        assert "//span[text()='Special Text']" == xpath_selector
    
    def test_replace_selector(self, script_recorder):
        """Test selector replacement in commands."""
        original = "await page.click('#old-selector')"
        new_selector = "button.new-class"
        
        replaced = script_recorder._replace_selector(original, new_selector)
        
        assert replaced == "await page.click('button.new-class')"
        assert "#old-selector" not in replaced
    
    def test_unknown_action_type(self, script_recorder):
        """Test handling unknown action types."""
        result = JournalActionResult(
            success=True,
            action=None,  # No action type
            confidence=1.0
        )
        
        # Use a mock action type that's not in the mapping
        from unittest.mock import Mock
        unknown_action = Mock()
        unknown_action.value = "unknown_action"
        
        command = script_recorder.record_action(
            unknown_action,
            result,
            {}
        )
        
        assert command is None