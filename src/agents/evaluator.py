"""
Evaluator Agent implementation.

Analyzes screenshots to assess test results and determine success/failure.
"""

import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from PIL import Image

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import EVALUATOR_AGENT_SYSTEM_PROMPT
from src.core.types import EvaluationResult
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class EvaluatorAgent(BaseAgent):
    """
    AI agent that evaluates test results by analyzing screenshots.
    
    This agent compares actual UI state with expected outcomes to determine
    if test steps succeeded, failed, or partially succeeded. It provides
    confidence scores and detailed analysis of any deviations.
    """
    
    def __init__(self, name: str = "EvaluatorAgent", **kwargs):
        """Initialize the Evaluator Agent."""
        super().__init__(name=name, **kwargs)
        self.system_prompt = EVALUATOR_AGENT_SYSTEM_PROMPT
    
    async def evaluate_result(
        self, screenshot: bytes, expected_outcome: str, step_id: Optional[UUID] = None
    ) -> EvaluationResult:
        """
        Evaluate if the actual result matches expected outcome.
        
        Args:
            screenshot: Screenshot of current state
            expected_outcome: Expected outcome description
            step_id: Optional test step ID
            
        Returns:
            Evaluation result with confidence score
        """
        logger.info("Evaluating test result", extra={
            "expected_outcome": expected_outcome[:100] + "..." if len(expected_outcome) > 100 else expected_outcome,
            "step_id": str(step_id) if step_id else None
        })
        
        # Analyze the screenshot
        analysis = await self._analyze_screenshot(screenshot, expected_outcome)
        
        # Create evaluation result
        result = EvaluationResult(
            step_id=step_id or uuid4(),
            success=analysis["success"],
            confidence=analysis["confidence"],
            expected_outcome=expected_outcome,
            actual_outcome=analysis["actual_outcome"],
            deviations=analysis.get("deviations", []),
            suggestions=analysis.get("suggestions", []),
            screenshot_analysis=analysis.get("detailed_analysis")
        )
        
        logger.info("Evaluation complete", extra={
            "success": result.success,
            "confidence": result.confidence,
            "deviations_count": len(result.deviations),
            "has_suggestions": bool(result.suggestions),
            "expected": expected_outcome[:100] + "..." if len(expected_outcome) > 100 else expected_outcome,
            "actual": result.actual_outcome[:100] + "..." if len(result.actual_outcome) > 100 else result.actual_outcome
        })
        
        return result
    
    async def _analyze_screenshot(
        self, screenshot: bytes, expected_outcome: str
    ) -> Dict[str, Any]:
        """Analyze screenshot to determine if expected outcome was achieved."""
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(expected_outcome)
        
        # Convert screenshot to base64
        base64_image = base64.b64encode(screenshot).decode('utf-8')
        
        # Build messages for AI
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        # Call AI for analysis
        response = await self.call_openai(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for consistent evaluation
        )
        
        # Parse and validate response
        return self._parse_analysis_response(response)
    
    def _build_analysis_prompt(self, expected_outcome: str) -> str:
        """Build the prompt for screenshot analysis."""
        prompt = f"""Analyze this screenshot to determine if the expected outcome has been achieved.

Expected Outcome: {expected_outcome}

Please evaluate the screenshot and provide a detailed analysis in the following JSON format:
{{
    "success": true/false,  // Whether the expected outcome was achieved
    "confidence": 0.95,     // Confidence score (0.0-1.0)
    "actual_outcome": "Description of what is actually shown in the screenshot",
    "deviations": [         // List of deviations from expected (if any)
        "Deviation 1",
        "Deviation 2"
    ],
    "suggestions": [        // Suggestions for next actions (if failed)
        "Suggestion 1",
        "Suggestion 2"
    ],
    "detailed_analysis": {{  // Optional detailed analysis
        "elements_found": ["list of UI elements visible"],
        "text_content": ["key text visible on screen"],
        "ui_state": "description of overall UI state",
        "error_indicators": ["any error messages or indicators"],
        "success_indicators": ["any success messages or indicators"]
    }}
}}

Guidelines for evaluation:
1. Look for specific UI elements mentioned in the expected outcome
2. Check for success indicators (green checkmarks, success messages, etc.)
3. Check for error indicators (red text, error icons, warning messages)
4. Verify text content matches expectations
5. Consider partial success scenarios
6. Be specific about what you see vs. what was expected
7. Provide actionable suggestions if the test failed"""
        
        return prompt
    
    def _parse_analysis_response(self, response: Dict) -> Dict[str, Any]:
        """Parse AI response into structured analysis data."""
        try:
            content = response.get("content", {})
            
            # Handle string content (JSON)
            if isinstance(content, str):
                content = json.loads(content)
            
            # Extract and validate required fields
            analysis = {
                "success": bool(content.get("success", False)),
                "confidence": float(content.get("confidence", 0.5)),
                "actual_outcome": content.get("actual_outcome", "Unable to determine outcome"),
                "deviations": content.get("deviations", []),
                "suggestions": content.get("suggestions", []),
                "detailed_analysis": content.get("detailed_analysis")
            }
            
            # Ensure confidence is in valid range
            analysis["confidence"] = max(0.0, min(1.0, analysis["confidence"]))
            
            # Ensure lists are actually lists
            if not isinstance(analysis["deviations"], list):
                analysis["deviations"] = []
            if not isinstance(analysis["suggestions"], list):
                analysis["suggestions"] = []
            
            return analysis
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse analysis response", extra={
                "error": str(e),
                "response": response
            })
            # Return default analysis on error
            return {
                "success": False,
                "confidence": 0.1,
                "actual_outcome": "Error analyzing screenshot",
                "deviations": ["Failed to parse AI response"],
                "suggestions": ["Retry the evaluation"],
                "detailed_analysis": None
            }
    
    async def compare_screenshots(
        self, before: bytes, after: bytes, expected_changes: str
    ) -> Dict[str, Any]:
        """
        Compare two screenshots to detect changes.
        
        Args:
            before: Screenshot before action
            after: Screenshot after action
            expected_changes: Description of expected changes
            
        Returns:
            Analysis of changes detected
        """
        logger.info("Comparing screenshots for changes", extra={
            "expected_changes": expected_changes[:100] + "..." if len(expected_changes) > 100 else expected_changes
        })
        
        # Build comparison prompt
        prompt = f"""Compare these two screenshots to identify what changed.

Expected Changes: {expected_changes}

Analyze both screenshots and provide:
1. What specific changes occurred
2. Whether the expected changes were observed
3. Any unexpected changes
4. Overall assessment of the transition

Provide response in JSON format:
{{
    "changes_detected": ["list of specific changes"],
    "expected_changes_found": true/false,
    "unexpected_changes": ["list of unexpected changes"],
    "confidence": 0.9,
    "assessment": "overall assessment of the transition"
}}"""
        
        # Convert screenshots to base64
        before_b64 = base64.b64encode(before).decode('utf-8')
        after_b64 = base64.b64encode(after).decode('utf-8')
        
        # Build messages with both images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{before_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{after_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        # Get AI analysis
        response = await self.call_openai(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse response
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
            return content
        except Exception as e:
            logger.error("Failed to parse comparison response", extra={"error": str(e)})
            return {
                "changes_detected": [],
                "expected_changes_found": False,
                "unexpected_changes": [],
                "confidence": 0.0,
                "assessment": "Failed to analyze changes"
            }
    
    async def check_for_errors(self, screenshot: bytes) -> Dict[str, Any]:
        """
        Specialized check for error indicators in screenshot.
        
        Args:
            screenshot: Screenshot to analyze
            
        Returns:
            Analysis of any errors found
        """
        prompt = """Analyze this screenshot specifically for error indicators.

Look for:
1. Error messages or alerts
2. Red text or error colors
3. Warning icons or symbols
4. Modal dialogs with errors
5. Form validation errors
6. System error pages (404, 500, etc.)
7. Loading/timeout indicators stuck

Provide response in JSON format:
{
    "has_errors": true/false,
    "error_count": 0,
    "errors": [
        {
            "type": "error type (validation/system/network/etc)",
            "message": "error message if visible",
            "location": "where on screen",
            "severity": "high/medium/low"
        }
    ],
    "confidence": 0.9
}"""
        
        base64_image = base64.b64encode(screenshot).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        response = await self.call_openai(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2  # Very low temperature for error detection
        )
        
        try:
            content = response.get("content", {})
            if isinstance(content, str):
                content = json.loads(content)
            return content
        except Exception as e:
            logger.error("Failed to parse error check response", extra={"error": str(e)})
            return {
                "has_errors": False,
                "error_count": 0,
                "errors": [],
                "confidence": 0.0
            }