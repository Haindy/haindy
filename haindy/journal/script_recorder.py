"""
Script recorder for converting AI actions to Automation commands.
"""

import re
from typing import Any

from haindy.core.enhanced_types import EnhancedActionResult
from haindy.core.types import ActionType
from haindy.journal.adapters import enhanced_to_journal_action_result
from haindy.journal.models import ActionRecord, JournalActionResult, ScriptedCommand
from haindy.monitoring.logger import get_logger

logger = get_logger(__name__)


class ScriptRecorder:
    """
    Records and converts AI-driven actions into scripted Automation commands.

    Enables dual-mode execution by creating replayable scripts from
    successful visual interactions.
    """

    def __init__(self) -> None:
        """Initialize the script recorder."""
        self.selector_strategies = [
            self._generate_data_testid_selector,
            self._generate_id_selector,
            self._generate_text_selector,
            self._generate_role_selector,
            self._generate_css_selector,
            self._generate_xpath_selector,
        ]

    def record_action(
        self,
        action_type: ActionType,
        action_result: JournalActionResult | EnhancedActionResult,
        element_info: dict[str, Any] | None = None,
    ) -> ScriptedCommand | None:
        """
        Record an action as a scripted command.

        Args:
            action_type: Type of action performed
            action_result: Result of the action (JournalActionResult or EnhancedActionResult)
            element_info: Additional element information

        Returns:
            ScriptedCommand if recording successful
        """
        # Convert EnhancedActionResult to JournalActionResult if needed
        if isinstance(action_result, EnhancedActionResult):
            journal_result = enhanced_to_journal_action_result(
                action_result, action_type
            )
            logger.debug(
                "Converted EnhancedActionResult to JournalActionResult for script recording",
                extra={
                    "enhanced_success": action_result.overall_success,
                    "journal_success": journal_result.success,
                },
            )
        else:
            journal_result = action_result

        if not journal_result.success:
            return None

        # Generate command based on action type
        command_func = {
            ActionType.CLICK: self._record_click,
            ActionType.TYPE: self._record_type,
            ActionType.NAVIGATE: self._record_navigate,
            ActionType.SCROLL: self._record_scroll,
            ActionType.WAIT: self._record_wait,
            ActionType.SCREENSHOT: self._record_screenshot,
        }.get(action_type)

        if not command_func:
            logger.warning(f"Unknown action type: {action_type}")
            return None

        return command_func(journal_result, element_info)

    def generate_selectors(self, element_info: dict[str, Any]) -> list[str]:
        """
        Generate multiple selector strategies for an element.

        Args:
            element_info: Element information

        Returns:
            List of selectors ordered by preference
        """
        selectors = []

        for strategy in self.selector_strategies:
            selector = strategy(element_info)
            if selector:
                selectors.append(selector)

        return selectors

    def create_fallback_script(self, pattern: ActionRecord) -> list[str]:
        """
        Create fallback scripts for a pattern.

        Args:
            pattern: Action pattern

        Returns:
            List of alternative commands
        """
        fallbacks = []

        # Add variations with different wait strategies
        base_command = pattern.automation_command

        # Add explicit wait before action
        fallbacks.append(f"wait ms=500; {base_command}")

        # Add retry with different selector
        if pattern.selectors:
            for selector in list(pattern.selectors.values())[
                1:3
            ]:  # Use next 2 selectors
                alt_command = self._replace_selector(base_command, selector)
                if alt_command != base_command:
                    fallbacks.append(alt_command)

        # Add force click option for clicks
        if base_command.startswith("click"):
            force_click = base_command + " force=true"
            fallbacks.append(force_click)

        return fallbacks

    def _record_click(
        self, action_result: JournalActionResult, element_info: dict[str, Any] | None
    ) -> ScriptedCommand | None:
        """Record a click action."""
        if not element_info:
            return None

        selectors = self.generate_selectors(element_info)
        if not selectors:
            return None

        primary_selector = selectors[0]
        command = f"click selector='{primary_selector}'"

        # Store the actual command that was generated
        if action_result.automation_command:
            command = action_result.automation_command

        return ScriptedCommand(
            command_type="click",
            command=command,
            selectors=selectors,
            parameters={"timeout_seconds": 30.0, "force": False},
        )

    def _record_type(
        self, action_result: JournalActionResult, element_info: dict[str, Any] | None
    ) -> ScriptedCommand | None:
        """Record a type action."""
        if not element_info or not action_result.input_text:
            return None

        selectors = self.generate_selectors(element_info)
        if not selectors:
            return None

        primary_selector = selectors[0]
        text = action_result.input_text.replace("'", "\\'")
        command = f"type selector='{primary_selector}' text='{text}'"

        if action_result.automation_command:
            command = action_result.automation_command

        return ScriptedCommand(
            command_type="type",
            command=command,
            selectors=selectors,
            parameters={"text": action_result.input_text, "timeout_seconds": 30.0},
        )

    def _record_navigate(
        self, action_result: JournalActionResult, element_info: dict[str, Any] | None
    ) -> ScriptedCommand | None:
        """Record a navigation action."""
        url = element_info.get("url") if element_info else None
        if not url:
            return None

        command = f"navigate url='{url}'"

        if action_result.automation_command:
            command = action_result.automation_command

        return ScriptedCommand(
            command_type="navigate",
            command=command,
            selectors=[],
            parameters={
                "url": url,
                "wait_until": "networkidle",
                "timeout_seconds": 60.0,
            },
        )

    def _record_scroll(
        self, action_result: JournalActionResult, element_info: dict[str, Any] | None
    ) -> ScriptedCommand | None:
        """Record a scroll action."""
        scroll_direction = (
            element_info.get("direction", "down") if element_info else "down"
        )
        scroll_amount = element_info.get("amount", 300) if element_info else 300

        if scroll_direction == "down":
            command = f"scroll direction='down' amount={scroll_amount}"
        else:
            command = f"scroll direction='up' amount={scroll_amount}"

        if action_result.automation_command:
            command = action_result.automation_command

        return ScriptedCommand(
            command_type="scroll",
            command=command,
            selectors=[],
            parameters={"direction": scroll_direction, "amount": scroll_amount},
        )

    def _record_wait(
        self, action_result: JournalActionResult, element_info: dict[str, Any] | None
    ) -> ScriptedCommand | None:
        """Record a wait action."""
        wait_time = element_info.get("duration", 1000) if element_info else 1000
        command = f"wait ms={wait_time}"

        if action_result.automation_command:
            command = action_result.automation_command

        return ScriptedCommand(
            command_type="wait",
            command=command,
            selectors=[],
            parameters={"duration": wait_time},
        )

    def _record_screenshot(
        self, action_result: JournalActionResult, element_info: dict[str, Any] | None
    ) -> ScriptedCommand | None:
        """Record a screenshot action."""
        path = (
            element_info.get("path", "screenshot.png")
            if element_info
            else "screenshot.png"
        )
        command = f"screenshot path='{path}'"

        if action_result.automation_command:
            command = action_result.automation_command

        return ScriptedCommand(
            command_type="screenshot",
            command=command,
            selectors=[],
            parameters={"path": path, "full_page": True},
        )

    def _generate_data_testid_selector(
        self, element_info: dict[str, Any]
    ) -> str | None:
        """Generate data-testid selector."""
        testid = element_info.get("data-testid")
        if testid:
            return f'[data-testid="{testid}"]'
        return None

    def _generate_id_selector(self, element_info: dict[str, Any]) -> str | None:
        """Generate ID selector."""
        element_id = element_info.get("id")
        if element_id:
            return f"#{element_id}"
        return None

    def _generate_text_selector(self, element_info: dict[str, Any]) -> str | None:
        """Generate text-based selector."""
        text = element_info.get("text")
        tag = element_info.get("tag", "").lower()

        if text and tag in ["button", "a", "span", "div", "p"]:
            # Escape quotes in text
            text = text.replace('"', '\\"')
            return f'{tag}:has-text("{text}")'
        return None

    def _generate_role_selector(self, element_info: dict[str, Any]) -> str | None:
        """Generate ARIA role selector."""
        role = element_info.get("role")
        name = element_info.get("aria-label") or element_info.get("text")

        if role and name:
            name = name.replace('"', '\\"')
            return f'role={role}[name="{name}"]'
        elif role:
            return f"role={role}"
        return None

    def _generate_css_selector(self, element_info: dict[str, Any]) -> str | None:
        """Generate CSS selector."""
        raw_tag = element_info.get("tag", "")
        tag = raw_tag.lower() if isinstance(raw_tag, str) else ""
        raw_class_names = element_info.get("class", "")
        classes = raw_class_names.split() if isinstance(raw_class_names, str) else []

        if tag and classes:
            class_selector = ".".join(classes[:2])  # Use first 2 classes
            return f"{tag}.{class_selector}"
        elif tag:
            return tag
        return None

    def _generate_xpath_selector(self, element_info: dict[str, Any]) -> str | None:
        """Generate XPath selector as last resort."""
        raw_tag = element_info.get("tag", "")
        tag = raw_tag.lower() if isinstance(raw_tag, str) else ""
        text = element_info.get("text")

        if tag and isinstance(text, str):
            # Simple XPath for exact text match
            text = text.replace("'", "\\'")
            return f"//{tag}[text()='{text}']"
        return None

    def _replace_selector(self, command: str, new_selector: str) -> str:
        """Replace selector in a command."""
        # Replace only automation-command selector args, e.g. selector='...'
        pattern = r"(selector=)(['\"])(.*?)(\2)"
        match = re.search(pattern, command)
        if match:
            start = match.start(3)
            end = match.end(3)
            return f"{command[:start]}{new_selector}{command[end:]}"
        return command
