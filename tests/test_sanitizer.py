"""
Unit tests for data sanitization.
"""

import pytest
import re
import logging
from unittest.mock import Mock

from src.security.sanitizer import (
    DataSanitizer,
    SensitiveDataPattern,
    SanitizationRule,
    RedactionMethod,
    sanitize_string,
    sanitize_dict,
    mask_sensitive_data
)


class TestSensitiveDataPattern:
    """Test sensitive data pattern matching."""
    
    def test_pattern_creation(self):
        """Test pattern creation and configuration."""
        pattern = SensitiveDataPattern(
            name="test_pattern",
            pattern=re.compile(r'\d{3}-\d{2}-\d{4}'),
            redaction_method=RedactionMethod.MASK,
            placeholder="[REDACTED]",
            description="Test pattern"
        )
        
        assert pattern.name == "test_pattern"
        assert pattern.redaction_method == RedactionMethod.MASK
        assert pattern.enabled is True
    
    def test_pattern_matching(self):
        """Test pattern matching functionality."""
        pattern = SensitiveDataPattern(
            name="ssn",
            pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        )
        
        text = "My SSN is 123-45-6789 and my friend's is 987-65-4321."
        matches = pattern.matches(text)
        
        assert len(matches) == 2
        assert matches[0].group() == "123-45-6789"
        assert matches[1].group() == "987-65-4321"
    
    def test_disabled_pattern(self):
        """Test disabled patterns don't match."""
        pattern = SensitiveDataPattern(
            name="test",
            pattern=re.compile(r'test'),
            enabled=False
        )
        
        matches = pattern.matches("This is a test")
        assert len(matches) == 0


class TestDataSanitizer:
    """Test data sanitizer functionality."""
    
    def test_default_patterns(self):
        """Test default patterns are loaded."""
        sanitizer = DataSanitizer()
        
        # Should have various pattern categories
        pattern_names = [p.name for p in sanitizer.patterns]
        assert any("api_key" in name for name in pattern_names)
        assert any("credit_card" in name for name in pattern_names)
        assert any("email" in name for name in pattern_names)
        assert any("ssn" in name for name in pattern_names)
    
    def test_sanitize_api_keys(self):
        """Test API key sanitization."""
        sanitizer = DataSanitizer()
        
        # Test generic API key
        text = "My API key is sk_test_abcdef123456789012345678901234567890"
        result = sanitizer.sanitize_string(text)
        assert "sk_test_abcdef123456789012345678901234567890" not in result
        assert "****" in result or "[REDACTED]" in result
        
        # Test API key with prefix
        text2 = "api_key: 'super_secret_key_12345'"
        result2 = sanitizer.sanitize_string(text2)
        assert "super_secret_key_12345" not in result2
    
    def test_sanitize_credit_cards(self):
        """Test credit card sanitization."""
        sanitizer = DataSanitizer()
        
        # Visa
        text = "Card: 4111-1111-1111-1111"
        result = sanitizer.sanitize_string(text)
        assert "4111-1111-1111-1111" not in result
        assert "4111" in result  # Should show first 4 (partial redaction)
        assert "1111" in result  # Should show last 4
        
        # Mastercard
        text2 = "Payment: 5500 0000 0000 0004"
        result2 = sanitizer.sanitize_string(text2)
        assert "5500 0000 0000 0004" not in result2
    
    def test_sanitize_personal_info(self):
        """Test personal information sanitization."""
        sanitizer = DataSanitizer()
        
        # Email (partial redaction)
        text = "Contact: john.doe@example.com"
        result = sanitizer.sanitize_string(text)
        assert "john.doe@example.com" not in result
        assert "joh" in result  # First 3 chars
        assert "com" in result  # Last 3 chars
        
        # SSN
        text2 = "SSN: 123-45-6789"
        result2 = sanitizer.sanitize_string(text2)
        assert "123-45-6789" not in result2
        assert "***********" in result2  # Should be masked with asterisks
    
    def test_redaction_methods(self):
        """Test different redaction methods."""
        sanitizer = DataSanitizer()
        
        # MASK method
        pattern_mask = SensitiveDataPattern(
            name="test_mask",
            pattern=re.compile(r'SECRET'),
            redaction_method=RedactionMethod.MASK
        )
        sanitizer.add_pattern(pattern_mask)
        
        result = sanitizer.sanitize_string("The SECRET is here")
        assert "******" in result
        
        # PLACEHOLDER method
        pattern_placeholder = SensitiveDataPattern(
            name="test_placeholder",
            pattern=re.compile(r'CONFIDENTIAL'),
            redaction_method=RedactionMethod.PLACEHOLDER,
            placeholder="[REMOVED]"
        )
        sanitizer.add_pattern(pattern_placeholder)
        
        result2 = sanitizer.sanitize_string("This is CONFIDENTIAL data")
        assert "[REMOVED]" in result2
        
        # REMOVE method
        pattern_remove = SensitiveDataPattern(
            name="test_remove",
            pattern=re.compile(r'DELETE_ME'),
            redaction_method=RedactionMethod.REMOVE
        )
        sanitizer.add_pattern(pattern_remove)
        
        result3 = sanitizer.sanitize_string("Please DELETE_ME now")
        assert "DELETE_ME" not in result3
        assert "Please  now" in result3
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        sanitizer = DataSanitizer()
        
        data = {
            "api_key": "sk_test_123456789",
            "user": {
                "email": "user@example.com",
                "password": "supersecret123",
                "name": "John Doe"
            },
            "payment": {
                "card_number": "4111111111111111",
                "cvv": "123"
            }
        }
        
        result = sanitizer.sanitize_dict(data)
        
        # Check sanitization
        assert "sk_test_123456789" not in str(result)
        assert "supersecret123" not in str(result)
        assert "4111111111111111" not in str(result)
        assert result["user"]["name"] == "John Doe"  # Should not be sanitized
    
    def test_nested_dict_sanitization(self):
        """Test deeply nested dictionary sanitization."""
        sanitizer = DataSanitizer()
        
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "secret": "api_key: secret123"
                    }
                }
            }
        }
        
        result = sanitizer.sanitize_dict(data)
        assert "secret123" not in str(result)
    
    def test_list_sanitization(self):
        """Test list sanitization in dictionaries."""
        sanitizer = DataSanitizer()
        
        data = {
            "emails": ["user1@example.com", "user2@example.com"],
            "tokens": ["Bearer abc123", "Bearer xyz789"]
        }
        
        result = sanitizer.sanitize_dict(data)
        
        # Emails should be partially redacted
        assert "user1@example.com" not in str(result)
        assert "user2@example.com" not in str(result)
        
        # Bearer tokens should be redacted
        assert "abc123" not in str(result)
        assert "xyz789" not in str(result)
    
    def test_custom_sanitizer(self):
        """Test custom sanitization function."""
        sanitizer = DataSanitizer()
        
        # Add custom sanitizer
        def custom_redact(value: str) -> str:
            return "CUSTOM_REDACTED"
        
        sanitizer.add_custom_sanitizer("custom_field", custom_redact)
        
        data = {
            "custom_field": "sensitive data",
            "normal_field": "normal data"
        }
        
        result = sanitizer.sanitize_dict(data)
        assert result["custom_field"] == "CUSTOM_REDACTED"
        assert result["normal_field"] == "normal data"
    
    def test_sanitize_log_record(self):
        """Test log record sanitization."""
        sanitizer = DataSanitizer()
        
        # Create mock log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User email is %s and token is %s",
            args=("user@example.com", "Bearer secret123"),
            exc_info=None
        )
        
        sanitized = sanitizer.sanitize_log_record(record)
        
        # Check message is sanitized
        assert "user@example.com" not in sanitized.getMessage()
        assert "secret123" not in sanitized.getMessage()
    
    def test_statistics(self):
        """Test sanitizer statistics."""
        sanitizer = DataSanitizer()
        
        stats = sanitizer.get_statistics()
        assert stats["patterns"]["total"] > 0
        assert stats["patterns"]["enabled"] > 0
        assert stats["rules"]["total"] > 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_sanitize_string_function(self):
        """Test module-level sanitize_string."""
        text = "My password is secret123"
        result = sanitize_string(text)
        assert "secret123" not in result
    
    def test_sanitize_dict_function(self):
        """Test module-level sanitize_dict."""
        data = {"password": "secret", "user": "john"}
        result = sanitize_dict(data)
        assert result["password"] != "secret"
        assert result["user"] == "john"
    
    def test_mask_sensitive_data(self):
        """Test mask_sensitive_data function."""
        # Test default masking
        result1 = mask_sensitive_data("1234567890")
        assert result1 == "1234**7890"
        
        # Test custom masking
        result2 = mask_sensitive_data("abcdefghij", start_chars=2, end_chars=2)
        assert result2 == "ab******ij"
        
        # Test short strings
        result3 = mask_sensitive_data("abc", start_chars=2, end_chars=2)
        assert result3 == "***"