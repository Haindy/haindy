"""
Debug logging utilities for HAINDY.

Provides enhanced logging for AI interactions and screenshot management.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class DebugLogger:
    """Enhanced debug logger for AI interactions and screenshots."""
    
    def __init__(self, test_run_id: Optional[str] = None):
        """
        Initialize debug logger.
        
        Args:
            test_run_id: Unique identifier for the test run
        """
        self.test_run_id = test_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = Path("debug_screenshots") / self.test_run_id
        self.reports_dir = Path("reports") / self.test_run_id
        
        # Create directories
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AI interaction log
        self.ai_log_path = self.debug_dir / "ai_interactions.jsonl"
        self.screenshot_counter = 0
        
        logger.info(f"Debug logger initialized for test run: {self.test_run_id}")
        logger.info(f"Debug directory: {self.debug_dir}")
        logger.info(f"Reports directory: {self.reports_dir}")
    
    def log_ai_interaction(
        self,
        agent_name: str,
        action_type: str,
        prompt: str,
        response: str,
        screenshot_path: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """
        Log an AI interaction with prompt and response.
        
        Args:
            agent_name: Name of the agent making the call
            action_type: Type of action being performed
            prompt: The prompt sent to the AI
            response: The AI's response
            screenshot_path: Path to associated screenshot if any
            additional_context: Additional context information
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "test_run_id": self.test_run_id,
            "agent": agent_name,
            "action_type": action_type,
            "prompt": prompt,
            "response": response,
            "screenshot": screenshot_path,
            "context": additional_context or {}
        }
        
        # Write to JSONL file
        with open(self.ai_log_path, 'a') as f:
            f.write(json.dumps(interaction) + '\n')
        
        # Also log to console with formatting
        base_extra: Dict[str, Any] = {
            "agent_name": agent_name,
            "action_type": action_type,
        }
        if additional_context:
            base_extra.update(additional_context)

        response_ids = base_extra.pop("response_ids", None)

        truncated_prompt = (
            f"{prompt[:200]}..." if len(prompt) > 200 else prompt
        )
        truncated_response = (
            f"{response[:200]}..." if len(response) > 200 else response
        )

        logger.info("Action: %s", action_type, extra=dict(base_extra))
        logger.info("Prompt: %s", truncated_prompt, extra=dict(base_extra))
        logger.info("Response: %s", truncated_response, extra=dict(base_extra))
        if screenshot_path:
            logger.debug(
                "Screenshot saved: %s",
                screenshot_path,
                extra={**base_extra, "screenshot_path": screenshot_path},
            )
        if response_ids:
            logger.debug(
                "Response IDs: %s",
                response_ids,
                extra={**base_extra, "response_ids": response_ids},
            )
    
    def save_screenshot(
        self,
        screenshot_bytes: bytes,
        name: str,
        step_number: Optional[int] = None,
        with_grid: bool = False
    ) -> str:
        """
        Save a screenshot to the debug directory.
        
        Args:
            screenshot_bytes: Screenshot data
            name: Descriptive name for the screenshot
            step_number: Test step number if applicable
            with_grid: Whether this screenshot includes grid overlay
            
        Returns:
            Path to saved screenshot
        """
        self.screenshot_counter += 1
        
        # Build filename
        timestamp = datetime.now().strftime("%H%M%S_%f")[:12]  # Include microseconds
        grid_suffix = "_grid" if with_grid else ""
        step_prefix = f"step_{step_number}_" if step_number is not None else ""
        
        filename = f"{step_prefix}{name}{grid_suffix}_{timestamp}.png"
        filepath = self.debug_dir / filename
        
        # Save screenshot
        with open(filepath, 'wb') as f:
            f.write(screenshot_bytes)
        
        logger.debug(
            "Screenshot saved: %s",
            filepath,
            extra={
                "step_number": step_number,
                "with_grid": with_grid,
                "screenshot_path": str(filepath),
            },
        )
        return str(filepath)
    
    def save_grid_overlay(
        self,
        screenshot_bytes: bytes,
        grid_cell: str,
        coordinates: tuple,
        step_number: Optional[int] = None,
        action_type: str = "click"
    ) -> str:
        """
        Save a screenshot with grid overlay highlighting the selected cell.
        
        Args:
            screenshot_bytes: Screenshot with grid overlay
            grid_cell: The grid cell selected (e.g., "B7")
            coordinates: The (x, y) coordinates
            step_number: Test step number
            action_type: Type of action performed
            
        Returns:
            Path to saved screenshot
        """
        name = f"{action_type}_cell_{grid_cell}_at_{coordinates[0]}x{coordinates[1]}"
        return self.save_screenshot(
            screenshot_bytes,
            name=name,
            step_number=step_number,
            with_grid=True
        )
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get summary of debug information collected."""
        ai_interaction_count = 0
        if self.ai_log_path.exists():
            with open(self.ai_log_path, 'r') as f:
                ai_interaction_count = sum(1 for _ in f)
        
        screenshots = list(self.debug_dir.glob("*.png"))
        
        return {
            "test_run_id": self.test_run_id,
            "debug_directory": str(self.debug_dir),
            "reports_directory": str(self.reports_dir),
            "ai_interactions": ai_interaction_count,
            "screenshots_saved": len(screenshots),
            "screenshot_files": [s.name for s in screenshots]
        }
    
    def get_ai_conversations_html(self) -> str:
        """Get AI conversations formatted as HTML for reports."""
        if not self.ai_log_path.exists():
            return "<p>No AI interactions logged.</p>"
        
        html = "<div class='ai-conversations'>\n"
        html += "<h3>🤖 AI Conversations</h3>\n"
        
        with open(self.ai_log_path, 'r') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    
                    html += f"<div class='ai-interaction' style='border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;'>\n"
                    html += f"<h4>📋 Interaction #{i}: {data['action_type']} (Step {data.get('context', {}).get('step_number', 'N/A')})</h4>\n"
                    
                    # Prompt
                    html += "<div style='margin: 10px 0;'>\n"
                    html += "<strong>🤖 PROMPT:</strong>\n"
                    html += f"<pre style='background: #f5f5f5; padding: 10px; border-radius: 3px; white-space: pre-wrap;'>{data['prompt']}</pre>\n"
                    html += "</div>\n"
                    
                    # Response
                    html += "<div style='margin: 10px 0;'>\n"
                    html += "<strong>💬 AI RESPONSE:</strong>\n"
                    html += f"<pre style='background: #e8f4f8; padding: 10px; border-radius: 3px; white-space: pre-wrap;'>{data['response']}</pre>\n"
                    html += "</div>\n"
                    
                    # Screenshot
                    if data.get('screenshot'):
                        html += f"<p><strong>📸 SCREENSHOT:</strong> {data['screenshot']}</p>\n"
                    
                    html += "</div>\n"
                    
                except json.JSONDecodeError:
                    continue
        
        html += "</div>\n"
        return html


# Global instance for easy access
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> Optional[DebugLogger]:
    """Get the global debug logger instance."""
    return _debug_logger


def set_debug_logger(logger: DebugLogger):
    """Set the global debug logger instance."""
    global _debug_logger
    _debug_logger = logger


def initialize_debug_logger(test_run_id: Optional[str] = None) -> DebugLogger:
    """Initialize and return a new debug logger."""
    logger = DebugLogger(test_run_id)
    set_debug_logger(logger)
    return logger
