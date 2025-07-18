"""
Google Gemini API client wrapper for the HAINDY framework.
"""

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from PIL import Image

from src.config.settings import get_settings


class GeminiClient:
    """Wrapper for Google Gemini API interactions."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        reasoning_effort: str = "medium",
    ) -> None:
        """
        Initialize Gemini client.

        Args:
            model: Model to use for completions
            api_key: Optional API key (defaults to env/config)
            max_retries: Maximum number of retry attempts
            reasoning_effort: Reasoning effort level (mapped to thinking budget)
        """
        self.model = model
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger("gemini_client")

        settings = get_settings()
        self.api_key = api_key or getattr(settings, 'gemini_api_key', None)
        
        # Check environment variable if not in settings
        if not self.api_key:
            import os
            self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(api_key=self.api_key)
        
        # Map reasoning effort to thinking budget
        self.thinking_budgets = {
            "low": 512,
            "medium": 1024,
            "high": 2048,
        }

    async def call(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call Gemini API with messages.
        
        This method maps OpenAI-style messages to Gemini's format.
        
        Args:
            messages: List of message dictionaries with OpenAI format
            temperature: Temperature for generation
            response_format: Response format (currently unused for compatibility)
            
        Returns:
            Dictionary with response content and usage info
        """
        try:
            # Convert OpenAI-style messages to Gemini format
            contents = self._convert_messages_to_contents(messages)
            
            # Configure generation with thinking budget based on reasoning effort
            config = types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budgets.get(self.reasoning_effort, 1024)
                )
            )
            
            # Make the API call
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            # Extract text from response
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Handle multi-candidate responses
                response_text = response.candidates[0].content.parts[0].text
            
            # Return in OpenAI-compatible format
            return {
                "content": response_text,
                "usage": {
                    "total_tokens": getattr(response, 'usage', {}).get('total_token_count', 0),
                    "prompt_tokens": getattr(response, 'usage', {}).get('prompt_token_count', 0),
                    "completion_tokens": getattr(response, 'usage', {}).get('candidates_token_count', 0),
                }
            }
            
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {str(e)}")
            raise

    def _convert_messages_to_contents(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert OpenAI-style messages to Gemini contents format.
        
        OpenAI format:
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]}
        ]
        
        Gemini format needs a flat list of content parts.
        """
        contents = []
        
        # Combine system message with first user message if present
        system_prompt = None
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
        
        # Process messages
        for msg in messages:
            if msg["role"] == "system":
                continue  # Already handled
                
            if msg["role"] in ["user", "assistant"]:
                # Handle content that might be string or list
                content = msg["content"]
                
                if isinstance(content, str):
                    # Simple text message
                    text = content
                    if system_prompt and msg["role"] == "user" and not contents:
                        # Prepend system prompt to first user message
                        text = f"{system_prompt}\n\n{text}"
                    contents.append(text)
                    
                elif isinstance(content, list):
                    # Multi-part message (text + images)
                    for part in content:
                        if part["type"] == "text":
                            text = part["text"]
                            if system_prompt and msg["role"] == "user" and not contents:
                                # Prepend system prompt to first user message
                                text = f"{system_prompt}\n\n{text}"
                            contents.append(text)
                            
                        elif part["type"] == "image_url":
                            # Extract base64 image data
                            image_url = part["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                # Parse data URL
                                header, base64_data = image_url.split(",", 1)
                                mime_type = header.split(";")[0].split(":")[1]
                                
                                # Create image part for Gemini
                                image_bytes = base64.b64decode(base64_data)
                                image_part = types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=mime_type
                                )
                                contents.append(image_part)
        
        return contents

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """
        Complete a conversation (compatibility method).
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
            max_tokens: Max tokens (currently unused)
            response_format: Response format (currently unused)
            
        Returns:
            Generated text response
        """
        response = await self.call(messages, temperature, response_format)
        return response["content"]