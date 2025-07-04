"""
Data sanitization for sensitive information protection.

Provides patterns and methods to detect and redact sensitive data
in logs, screenshots, and test outputs.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Pattern, Set, Union, Callable
from copy import deepcopy

logger = logging.getLogger(__name__)


class RedactionMethod(Enum):
    """Methods for redacting sensitive data."""
    MASK = auto()          # Replace with asterisks
    HASH = auto()          # Replace with hash
    REMOVE = auto()        # Remove entirely
    PARTIAL = auto()       # Show partial (first/last few chars)
    PLACEHOLDER = auto()   # Replace with placeholder text


@dataclass
class SensitiveDataPattern:
    """Pattern for identifying sensitive data."""
    
    name: str
    pattern: Pattern[str]
    redaction_method: RedactionMethod = RedactionMethod.MASK
    placeholder: str = "[REDACTED]"
    partial_chars: int = 4  # For PARTIAL method
    description: str = ""
    enabled: bool = True
    
    def matches(self, text: str) -> List[re.Match]:
        """Find all matches in text."""
        if not self.enabled:
            return []
        return list(self.pattern.finditer(text))


@dataclass
class SanitizationRule:
    """Rule for sanitizing specific data types."""
    
    name: str
    patterns: List[SensitiveDataPattern]
    apply_to_keys: List[str] = field(default_factory=list)
    apply_to_values: bool = True
    case_sensitive: bool = False
    enabled: bool = True


class DataSanitizer:
    """Main sanitizer for protecting sensitive data."""
    
    def __init__(self):
        """Initialize with default patterns."""
        self.patterns: List[SensitiveDataPattern] = []
        self.rules: List[SanitizationRule] = []
        self.custom_sanitizers: Dict[str, Callable] = {}
        self._setup_default_patterns()
        self._setup_default_rules()
    
    def _setup_default_patterns(self) -> None:
        """Set up default sensitive data patterns."""
        # API Keys
        self.patterns.extend([
            SensitiveDataPattern(
                name="api_key_sk_format",
                pattern=re.compile(r'\bsk_[a-zA-Z0-9_]{8,}\b'),
                description="Stripe-style sk_ API keys"
            ),
            SensitiveDataPattern(
                name="api_key_generic",
                pattern=re.compile(r'\b[A-Za-z0-9_]{20,}\b'),
                description="Generic API key pattern"
            ),
            SensitiveDataPattern(
                name="api_key_prefix",
                pattern=re.compile(r'(api[_-]?key|apikey|api_secret|access[_-]?token)\s*[:=]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE),
                description="API key with common prefixes"
            ),
            SensitiveDataPattern(
                name="bearer_token",
                pattern=re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', re.IGNORECASE),
                description="Bearer authentication tokens"
            ),
        ])
        
        # Credit Cards
        self.patterns.extend([
            SensitiveDataPattern(
                name="credit_card_visa",
                pattern=re.compile(r'\b4[0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b'),
                redaction_method=RedactionMethod.PARTIAL,
                description="Visa card numbers"
            ),
            SensitiveDataPattern(
                name="credit_card_mastercard",
                pattern=re.compile(r'\b5[1-5][0-9]{2}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b'),
                redaction_method=RedactionMethod.PARTIAL,
                description="Mastercard numbers"
            ),
            SensitiveDataPattern(
                name="credit_card_amex",
                pattern=re.compile(r'\b3[47][0-9]{2}[\s-]?[0-9]{6}[\s-]?[0-9]{5}\b'),
                redaction_method=RedactionMethod.PARTIAL,
                description="American Express numbers"
            ),
        ])
        
        # Personal Information
        self.patterns.extend([
            SensitiveDataPattern(
                name="ssn",
                pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                redaction_method=RedactionMethod.MASK,
                description="Social Security Numbers"
            ),
            SensitiveDataPattern(
                name="email",
                pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                redaction_method=RedactionMethod.PARTIAL,
                partial_chars=3,
                description="Email addresses"
            ),
            SensitiveDataPattern(
                name="phone_us",
                pattern=re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
                redaction_method=RedactionMethod.PARTIAL,
                description="US phone numbers"
            ),
            SensitiveDataPattern(
                name="ip_address",
                pattern=re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
                redaction_method=RedactionMethod.PARTIAL,
                partial_chars=6,
                description="IPv4 addresses"
            ),
        ])
        
        # Authentication
        self.patterns.extend([
            SensitiveDataPattern(
                name="password_field",
                pattern=re.compile(r'(password|passwd|pwd)\s*[:=]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE),
                placeholder="[PASSWORD]",
                description="Password fields"
            ),
            SensitiveDataPattern(
                name="password_value",
                pattern=re.compile(r'\b\w*[sS]ecret\w*\b'),
                description="Common password patterns"
            ),
            SensitiveDataPattern(
                name="jwt_token",
                pattern=re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'),
                redaction_method=RedactionMethod.HASH,
                description="JWT tokens"
            ),
        ])
        
        # Financial
        self.patterns.extend([
            SensitiveDataPattern(
                name="bank_account",
                pattern=re.compile(r'\b[0-9]{8,17}\b'),
                enabled=False,  # Disabled by default due to false positives
                description="Bank account numbers"
            ),
            SensitiveDataPattern(
                name="routing_number",
                pattern=re.compile(r'\b[0-9]{9}\b'),
                enabled=False,  # Disabled by default due to false positives
                description="Bank routing numbers"
            ),
        ])
    
    def _setup_default_rules(self) -> None:
        """Set up default sanitization rules."""
        # Rule for JSON/Dict keys
        self.rules.append(
            SanitizationRule(
                name="auth_keys",
                patterns=[p for p in self.patterns if p.name in [
                    "api_key_sk_format", "api_key_generic", "api_key_prefix", 
                    "password_field", "password_value", "bearer_token", "jwt_token"
                ]],
                apply_to_keys=["password", "api_key", "apikey", "token", "secret", 
                              "auth", "authorization", "x-api-key"],
                apply_to_values=True
            )
        )
        
        # Rule for personal information
        self.rules.append(
            SanitizationRule(
                name="personal_info",
                patterns=[p for p in self.patterns if p.name in [
                    "email", "phone_us", "ssn", "credit_card_visa", 
                    "credit_card_mastercard", "credit_card_amex", "credit_card_generic"
                ]],
                apply_to_values=True
            )
        )
    
    def add_pattern(self, pattern: SensitiveDataPattern) -> None:
        """Add a custom pattern."""
        self.patterns.append(pattern)
    
    def add_rule(self, rule: SanitizationRule) -> None:
        """Add a custom rule."""
        self.rules.append(rule)
    
    def add_custom_sanitizer(self, name: str, func: Callable[[str], str]) -> None:
        """Add a custom sanitization function."""
        self.custom_sanitizers[name] = func
    
    def sanitize_string(
        self, 
        text: str, 
        patterns: Optional[List[SensitiveDataPattern]] = None
    ) -> str:
        """
        Sanitize a string using specified patterns.
        
        Args:
            text: Text to sanitize
            patterns: Patterns to use (defaults to all enabled patterns)
            
        Returns:
            Sanitized text
        """
        if not text:
            return text
        
        patterns = patterns or [p for p in self.patterns if p.enabled]
        result = text
        
        # Sort patterns by match position to handle overlaps
        all_matches = []
        for pattern in patterns:
            for match in pattern.matches(result):
                all_matches.append((match, pattern))
        
        # Sort by start position (reverse to process from end)
        all_matches.sort(key=lambda x: x[0].start(), reverse=True)
        
        # Apply redactions
        for match, pattern in all_matches:
            result = self._apply_redaction(result, match, pattern)
        
        return result
    
    def _apply_redaction(
        self, 
        text: str, 
        match: re.Match, 
        pattern: SensitiveDataPattern
    ) -> str:
        """Apply redaction based on method."""
        start, end = match.span()
        matched_text = match.group()
        
        if pattern.redaction_method == RedactionMethod.MASK:
            # Replace with asterisks
            replacement = "*" * len(matched_text)
            
        elif pattern.redaction_method == RedactionMethod.HASH:
            # Replace with hash
            import hashlib
            hash_val = hashlib.sha256(matched_text.encode()).hexdigest()[:8]
            replacement = f"[HASH:{hash_val}]"
            
        elif pattern.redaction_method == RedactionMethod.REMOVE:
            # Remove entirely
            replacement = ""
            
        elif pattern.redaction_method == RedactionMethod.PARTIAL:
            # Show partial
            if len(matched_text) > pattern.partial_chars * 2:
                replacement = (
                    matched_text[:pattern.partial_chars] + 
                    "*" * (len(matched_text) - pattern.partial_chars * 2) +
                    matched_text[-pattern.partial_chars:]
                )
            else:
                replacement = "*" * len(matched_text)
                
        else:  # PLACEHOLDER
            replacement = pattern.placeholder
        
        return text[:start] + replacement + text[end:]
    
    def sanitize_dict(
        self, 
        data: Dict[str, Any], 
        rules: Optional[List[SanitizationRule]] = None,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Sanitize a dictionary recursively.
        
        Args:
            data: Dictionary to sanitize
            rules: Rules to apply (defaults to all enabled rules)
            max_depth: Maximum recursion depth
            
        Returns:
            Sanitized dictionary (copy)
        """
        if max_depth <= 0:
            logger.warning("Max recursion depth reached in sanitize_dict")
            return data
        
        rules = rules or [r for r in self.rules if r.enabled]
        result = deepcopy(data)
        
        def _sanitize_value(value: Any, key: Optional[str] = None) -> Any:
            """Sanitize a single value."""
            if isinstance(value, str):
                # Check if key matches any rule
                applicable_patterns = []
                for rule in rules:
                    key_matches = False
                    if key:
                        if not rule.case_sensitive:
                            key_lower = key.lower()
                            key_matches = any(k.lower() in key_lower for k in rule.apply_to_keys)
                        else:
                            key_matches = any(k in key for k in rule.apply_to_keys)
                    
                    if key_matches or rule.apply_to_values:
                        applicable_patterns.extend(rule.patterns)
                
                # Apply custom sanitizers first
                for name, func in self.custom_sanitizers.items():
                    if key and name in key.lower():
                        value = func(value)
                
                # Apply pattern sanitization
                if applicable_patterns:
                    value = self.sanitize_string(value, applicable_patterns)
                    
            elif isinstance(value, dict):
                value = self.sanitize_dict(value, rules, max_depth - 1)
            elif isinstance(value, list):
                value = [_sanitize_value(item) for item in value]
                
            return value
        
        # Sanitize all values
        for key, value in result.items():
            result[key] = _sanitize_value(value, key)
        
        return result
    
    def sanitize_log_record(self, record: logging.LogRecord) -> logging.LogRecord:
        """
        Sanitize a log record.
        
        Args:
            record: Log record to sanitize
            
        Returns:
            Sanitized log record
        """
        # Sanitize message
        if hasattr(record, 'msg'):
            record.msg = self.sanitize_string(str(record.msg))
        
        # Sanitize args
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = self.sanitize_dict(record.args)
            else:
                record.args = tuple(
                    self.sanitize_string(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        
        return record
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        return {
            "patterns": {
                "total": len(self.patterns),
                "enabled": len([p for p in self.patterns if p.enabled]),
                "by_type": {
                    method.name: len([p for p in self.patterns if p.redaction_method == method])
                    for method in RedactionMethod
                }
            },
            "rules": {
                "total": len(self.rules),
                "enabled": len([r for r in self.rules if r.enabled])
            },
            "custom_sanitizers": len(self.custom_sanitizers)
        }


# Convenience functions
_default_sanitizer = DataSanitizer()

def sanitize_string(text: str) -> str:
    """Sanitize a string using default patterns."""
    return _default_sanitizer.sanitize_string(text)

def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize a dictionary using default rules."""
    return _default_sanitizer.sanitize_dict(data)

def mask_sensitive_data(
    text: str, 
    start_chars: int = 4, 
    end_chars: int = 4
) -> str:
    """
    Mask sensitive data showing only start/end characters.
    
    Args:
        text: Text to mask
        start_chars: Number of characters to show at start
        end_chars: Number of characters to show at end
        
    Returns:
        Masked text
    """
    if len(text) <= start_chars + end_chars:
        return "*" * len(text)
    
    return (
        text[:start_chars] + 
        "*" * (len(text) - start_chars - end_chars) +
        text[-end_chars:]
    )