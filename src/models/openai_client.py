"""
OpenAI API client wrapper for the HAINDY framework.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI

from src.config.settings import get_settings


class OpenAIClient:
    """Wrapper for OpenAI API interactions."""

    def __init__(
        self,
        model: str = "o4-mini-2025-04-16",
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            model: Model to use for completions
            api_key: Optional API key (defaults to env/config)
            max_retries: Maximum number of retry attempts
        """
        self.model = model
        self.max_retries = max_retries
        self.logger = logging.getLogger("openai_client")

        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            max_retries=self.max_retries,
        )

    async def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a call to OpenAI API.

        Args:
            messages: List of message dictionaries
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            system_prompt: System prompt to prepend
            response_format: Optional response format (for JSON mode)

        Returns:
            API response as dictionary
        """
        # Prepare messages with system prompt
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        # Log the request
        self.logger.debug(
            f"OpenAI API call: model={self.model}, "
            f"messages={len(final_messages)}, temperature={temperature}"
        )

        try:
            # Prepare kwargs
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": final_messages,
            }
            
            # o4-mini uses different parameters
            if "o4-mini" in self.model:
                # Use max_completion_tokens instead of max_tokens
                if max_tokens:
                    kwargs["max_completion_tokens"] = max_tokens
                # Add reasoning_effort for better accuracy
                kwargs["reasoning_effort"] = "medium"
            else:
                # Traditional models use temperature and max_tokens
                kwargs["temperature"] = temperature
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

            if response_format:
                kwargs["response_format"] = response_format

            # Make the API call
            response = await self.client.chat.completions.create(**kwargs)

            # Extract and log response
            content = response.choices[0].message.content
            self.logger.debug(f"OpenAI API response: {len(content)} characters")

            # Parse JSON if response format was specified
            if response_format and response_format.get("type") == "json_object":
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response: {e}")
                    content = {"error": "Invalid JSON response", "raw": content}

            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error calling OpenAI: {e}")
            raise

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        temperature: float = 0.7,
        detail: str = "high",
    ) -> Dict[str, Any]:
        """
        Analyze an image using vision capabilities.

        Args:
            image_data: Image data as bytes
            prompt: Analysis prompt
            temperature: Temperature for response
            detail: Image detail level ('low', 'high', 'auto')

        Returns:
            Analysis response
        """
        import base64

        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": detail,
                        },
                    },
                ],
            }
        ]

        return await self.call(messages=messages, temperature=temperature)

    async def create_structured_output(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        temperature: float = 0.7,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Create structured output with schema validation.

        Args:
            prompt: User prompt
            response_schema: JSON schema for response
            temperature: Temperature for response
            examples: Optional examples to include

        Returns:
            Structured response matching schema
        """
        # Build the system prompt with schema
        system_prompt = (
            "You are a precise AI that always returns valid JSON matching the provided schema. "
            f"Schema: {json.dumps(response_schema, indent=2)}"
        )

        messages = []

        # Add examples if provided
        if examples:
            for example in examples:
                messages.append({"role": "user", "content": example["input"]})
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(example["output"]),
                    }
                )

        # Add the actual prompt
        messages.append({"role": "user", "content": prompt})

        response = await self.call(
            messages=messages,
            temperature=temperature,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
        )

        return response["content"]

    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """
        Estimate cost based on token usage.

        Args:
            usage: Token usage dictionary

        Returns:
            Estimated cost in USD
        """
        # Pricing as of GPT-4o mini (adjust as needed)
        # These are example prices - update with actual pricing
        pricing = {
            "gpt-4o-mini": {
                "prompt": 0.00015 / 1000,  # per token
                "completion": 0.0006 / 1000,  # per token
            },
            "gpt-4o": {
                "prompt": 0.005 / 1000,
                "completion": 0.015 / 1000,
            },
        }

        model_pricing = pricing.get(
            self.model, pricing["gpt-4o-mini"]
        )  # Default to mini pricing

        prompt_cost = usage.get("prompt_tokens", 0) * model_pricing["prompt"]
        completion_cost = (
            usage.get("completion_tokens", 0) * model_pricing["completion"]
        )

        return prompt_cost + completion_cost