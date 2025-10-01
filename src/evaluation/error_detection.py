"""
Error detection utilities extracted from Evaluator Agent.

Provides specialized error detection capabilities for screenshots
and UI states.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class ErrorType(str, Enum):
    """Types of errors that can be detected in UI."""
    VALIDATION = "validation"
    SYSTEM = "system"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    UI = "ui"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DetectedError:
    """Represents a detected error in the UI."""
    error_type: ErrorType
    message: str
    location: str
    severity: ErrorSeverity
    confidence: float = 0.0
    details: Optional[Dict[str, Any]] = None


class ErrorDetector:
    """
    Utilities for detecting errors in UI states.
    
    Extracted from Evaluator Agent to provide reusable error detection
    capabilities for any agent that needs them.
    """
    
    # Common error indicators to look for
    ERROR_KEYWORDS = [
        "error", "fail", "invalid", "unauthorized", "forbidden",
        "not found", "404", "500", "503", "timeout", "expired",
        "denied", "rejected", "unable", "cannot", "exception"
    ]
    
    # Common error UI patterns
    ERROR_PATTERNS = {
        "red_text": ["color: red", "text-red", "error-text", "danger"],
        "error_icons": ["❌", "⚠️", "⛔", "!", "X", "✗"],
        "error_classes": ["error", "alert", "danger", "warning", "fail"],
    }
    
    @classmethod
    def build_error_detection_prompt(cls) -> str:
        """
        Build a prompt for AI to detect errors in screenshots.
        
        Returns:
            Prompt string for error detection
        """
        return f"""Analyze this screenshot specifically for error indicators.

Look for:
1. Error messages or alerts
2. Red text or error colors  
3. Warning icons or symbols ({', '.join(cls.ERROR_PATTERNS['error_icons'])})
4. Modal dialogs with errors
5. Form validation errors
6. System error pages (404, 500, etc.)
7. Loading/timeout indicators stuck
8. Keywords like: {', '.join(cls.ERROR_KEYWORDS[:10])}

Provide response in JSON format:
{{
    "has_errors": true/false,
    "error_count": 0,
    "errors": [
        {{
            "type": "{'/'.join([e.value for e in ErrorType])}",
            "message": "error message if visible",
            "location": "where on screen", 
            "severity": "{'/'.join([s.value for s in ErrorSeverity])}"
        }}
    ],
    "confidence": 0.0-1.0
}}"""
    
    @staticmethod
    def parse_error_detection_response(response: Dict[str, Any]) -> List[DetectedError]:
        """
        Parse AI response into structured error objects.
        
        Args:
            response: AI response dictionary
            
        Returns:
            List of detected errors
        """
        errors = []
        
        if not response.get("has_errors", False):
            return errors
            
        for error_data in response.get("errors", []):
            try:
                error = DetectedError(
                    error_type=ErrorType(error_data.get("type", "unknown")),
                    message=error_data.get("message", "Unknown error"),
                    location=error_data.get("location", "Unknown location"),
                    severity=ErrorSeverity(error_data.get("severity", "medium")),
                    confidence=response.get("confidence", 0.5)
                )
                errors.append(error)
            except (ValueError, KeyError):
                # Skip malformed error entries
                continue
                
        return errors
    
    @staticmethod
    def categorize_error_by_message(message: str) -> ErrorType:
        """
        Attempt to categorize an error based on its message.
        
        Args:
            message: Error message text
            
        Returns:
            Best guess at error type
        """
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["401", "403", "unauthorized", "forbidden", "denied"]):
            return ErrorType.AUTHENTICATION
        elif any(word in message_lower for word in ["404", "not found", "missing"]):
            return ErrorType.NOT_FOUND
        elif any(word in message_lower for word in ["500", "502", "503", "server error", "internal"]):
            return ErrorType.SYSTEM
        elif any(word in message_lower for word in ["timeout", "timed out", "network", "connection"]):
            return ErrorType.NETWORK
        elif any(word in message_lower for word in ["invalid", "validation", "required", "must"]):
            return ErrorType.VALIDATION
        elif any(word in message_lower for word in ["permission", "access", "privilege"]):
            return ErrorType.PERMISSION
        else:
            return ErrorType.UNKNOWN
    
    @staticmethod
    def assess_error_impact(errors: List[DetectedError]) -> Dict[str, Any]:
        """
        Assess the overall impact of detected errors.
        
        Args:
            errors: List of detected errors
            
        Returns:
            Impact assessment dictionary
        """
        if not errors:
            return {
                "has_blocking_errors": False,
                "highest_severity": None,
                "error_count": 0,
                "can_continue": True,
                "recommendations": []
            }
        
        severities = [error.severity for error in errors]
        highest_severity = min(severities, key=lambda s: list(ErrorSeverity).index(s))
        
        has_blocking = highest_severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
        
        recommendations = []
        if has_blocking:
            recommendations.append("Address critical errors before continuing")
        
        auth_errors = [e for e in errors if e.error_type == ErrorType.AUTHENTICATION]
        if auth_errors:
            recommendations.append("Check authentication credentials or session")
            
        network_errors = [e for e in errors if e.error_type == ErrorType.NETWORK]
        if network_errors:
            recommendations.append("Verify network connectivity and retry")
            
        return {
            "has_blocking_errors": has_blocking,
            "highest_severity": highest_severity,
            "error_count": len(errors),
            "can_continue": not has_blocking,
            "recommendations": recommendations,
            "error_types": list(set(e.error_type for e in errors))
        }