"""
Common validation helpers for test evaluation.

Provides reusable validation patterns and utilities.
"""

from typing import Dict, List, Optional, Any, Tuple
import re


class ValidationHelpers:
    """
    Common validation utilities for test evaluation.
    
    These helpers can be used by any agent to validate
    test outcomes and UI states.
    """
    
    @staticmethod
    def extract_url_from_content(content: str) -> Optional[str]:
        """
        Extract URL from AI response content.
        
        Args:
            content: AI response content
            
        Returns:
            Extracted URL or None
        """
        # Look for URL patterns
        url_patterns = [
            r'URL:\s*([^\s\n]+)',
            r'NEW_URL:\s*([^\s\n]+)',
            r'url["\']?\s*:\s*["\']([^"\']+)["\']',
            r'https?://[^\s\n]+',
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return None
    
    @staticmethod
    def parse_boolean_from_content(content: str, key: str) -> Optional[bool]:
        """
        Parse boolean value from AI response content.
        
        Args:
            content: AI response content  
            key: Key to look for (e.g., "SUCCESS", "CHANGED")
            
        Returns:
            Boolean value or None if not found
        """
        patterns = [
            f"{key}:\\s*(true|false)",
            f"{key}\\s*=\\s*(true|false)",
            f'"{key}"\\s*:\\s*(true|false)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).lower() == "true"
                
        return None
    
    @staticmethod
    def extract_list_from_content(content: str, key: str) -> List[str]:
        """
        Extract list items from AI response content.
        
        Args:
            content: AI response content
            key: Key to look for (e.g., "UI_CHANGES")
            
        Returns:
            List of extracted items
        """
        # Look for the key and extract content after it
        pattern = f"{key}:\\s*([^\\n]+)"
        match = re.search(pattern, content, re.IGNORECASE)
        
        if not match:
            return []
            
        list_content = match.group(1).strip()
        
        # Handle different list formats
        if list_content.startswith("[") and list_content.endswith("]"):
            # JSON array format
            list_content = list_content[1:-1]
            
        # Split by common delimiters
        items = re.split(r'[,;|]', list_content)
        
        # Clean and filter items
        cleaned_items = []
        for item in items:
            item = item.strip().strip('"\'')
            if item and item.lower() not in ["none", "n/a", "null", ""]:
                cleaned_items.append(item)
                
        return cleaned_items
    
    @staticmethod
    def validate_expected_text_present(
        text_list: List[str],
        expected_texts: List[str]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that expected text appears in found text.
        
        Args:
            text_list: List of text found on screen
            expected_texts: List of expected text
            
        Returns:
            Tuple of (all_found, found_texts, missing_texts)
        """
        found_texts = []
        missing_texts = []
        
        # Normalize for comparison
        normalized_found = [text.lower().strip() for text in text_list]
        
        for expected in expected_texts:
            expected_normalized = expected.lower().strip()
            
            # Check for exact match or substring
            found = False
            for text in normalized_found:
                if expected_normalized in text or text in expected_normalized:
                    found = True
                    found_texts.append(expected)
                    break
                    
            if not found:
                missing_texts.append(expected)
                
        all_found = len(missing_texts) == 0
        return all_found, found_texts, missing_texts
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
            
        # Check substring relationship
        if text1 in text2 or text2 in text1:
            shorter = min(len(text1), len(text2))
            longer = max(len(text1), len(text2))
            return shorter / longer
            
        # Simple character overlap
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 or not set2:
            return 0.0
            
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def detect_common_ui_states(
        text_content: List[str],
        element_descriptions: List[str]
    ) -> Dict[str, bool]:
        """
        Detect common UI states from content.
        
        Args:
            text_content: Text found on screen
            element_descriptions: Descriptions of UI elements
            
        Returns:
            Dictionary of detected states
        """
        all_content = " ".join(text_content + element_descriptions).lower()
        
        states = {
            "loading": any(indicator in all_content for indicator in 
                          ["loading", "spinner", "progress", "please wait"]),
            "error": any(indicator in all_content for indicator in
                        ["error", "fail", "invalid", "exception"]),
            "success": any(indicator in all_content for indicator in
                          ["success", "complete", "done", "✓", "✔"]),
            "form": any(indicator in all_content for indicator in
                       ["input", "field", "form", "textbox", "submit"]),
            "modal": any(indicator in all_content for indicator in
                        ["modal", "dialog", "popup", "overlay"]),
            "empty": len(text_content) < 3 and "empty" in all_content,
        }
        
        return states
    
    @staticmethod
    def validate_ui_transition(
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        expected_changes: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that a UI transition occurred as expected.
        
        Args:
            before_state: UI state before action
            after_state: UI state after action  
            expected_changes: List of expected changes
            
        Returns:
            Validation result dictionary
        """
        actual_changes = []
        unexpected_changes = []
        missing_changes = []
        
        # Compare states
        for key in set(before_state.keys()).union(after_state.keys()):
            before_val = before_state.get(key)
            after_val = after_state.get(key)
            
            if before_val != after_val:
                change_desc = f"{key}: {before_val} → {after_val}"
                actual_changes.append(change_desc)
                
                # Check if this was expected
                expected = False
                for expected_change in expected_changes:
                    if key.lower() in expected_change.lower():
                        expected = True
                        break
                        
                if not expected:
                    unexpected_changes.append(change_desc)
        
        # Check for missing expected changes
        for expected_change in expected_changes:
            found = False
            for actual_change in actual_changes:
                if expected_change.lower() in actual_change.lower():
                    found = True
                    break
            if not found:
                missing_changes.append(expected_change)
        
        return {
            "transition_occurred": len(actual_changes) > 0,
            "all_expected_changes": len(missing_changes) == 0,
            "actual_changes": actual_changes,
            "unexpected_changes": unexpected_changes,
            "missing_changes": missing_changes,
            "change_count": len(actual_changes)
        }