"""Base provider for OpenAI-compatible APIs.

This module provides a base class for all providers that use the OpenAI SDK
to communicate with their APIs. This includes OpenRouter, Anthropic (via OpenAI
compatibility layer), Ollama, and potentially future OpenAI-compatible providers.

All the common logic for chat completions, streaming, metadata extraction,
rate limiting, and error handling is centralized here to eliminate duplication
across provider implementations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Unpack, cast

from httpx import HTTPStatusError
from openai import Stream
from openai.types.chat import ChatCompletionChunk

from .base_model_provider import (
    BaseModelProvider,
    ChatMessage,
    GenerationMetadata,
    ModelOverrides,
)
from .exceptions import ProviderRateLimitError

if TYPE_CHECKING:
    from dialectus.engine.config.settings import ModelConfig, SystemConfig
    from dialectus.engine.debate_engine.types import ChunkCallback

logger = logging.getLogger(__name__)


class OpenAICompatibleProviderBase(BaseModelProvider):
    """Base class for providers using OpenAI-compatible APIs.

    This class provides all the common functionality for providers that use
    the OpenAI SDK, including:
    - Chat completion generation
    - Streaming responses
    - Metadata extraction (tokens, cost, timing)
    - Rate limiting
    - Error handling

    Subclasses need only implement:
    - __init__() - Client setup with provider-specific configuration
    - provider_name property - Return provider name string
    - get_enhanced_models() - Model discovery/catalog
    - validate_model_config() - Model validation

    Subclasses may optionally override:
    - _min_request_interval - Rate limit interval (default 0)
    - query_generation_cost() - Provider-specific cost API
    - is_running() - Provider health check
    - _calculate_cost() - Custom cost calculation logic
    """

    # Class-level rate limiting configuration
    # Subclasses can override this class variable to set their rate limit interval
    _min_request_interval: ClassVar[float] = 0.0  # No rate limiting by default
    _last_request_time: ClassVar[float | None] = None
    _request_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, system_config: SystemConfig):
        """Initialize the provider.

        Subclasses should call super().__init__(system_config) and then
        set up their OpenAI client with provider-specific configuration.
        """
        super().__init__(system_config)

    @staticmethod
    def _coerce_content_to_text(content: object) -> str:
        """Extract textual content from OpenAI-style message content payloads.

        OpenAI API responses can return content in various formats:
        - Plain string: "Hello world"
        - List of content blocks: [{"type": "text", "text": "Hello"}]
        - Nested structures with reasoning/tool blocks

        This method normalizes all these formats to a single string, filtering
        out non-user-facing content like reasoning and tool calls.

        Args:
            content: The content value from an OpenAI message

        Returns:
            Normalized text content as a string
        """
        if isinstance(content, str):
            return content

        if isinstance(content, Sequence):
            parts: list[str] = []
            content_seq = cast(Sequence[object], content)
            for element in content_seq:
                if isinstance(element, str):
                    parts.append(element)
                    continue
                if isinstance(element, Mapping):
                    element_typed = cast(Mapping[str, object], element)

                    # Check if this is a reasoning or tool block (should be filtered)
                    part_type: object | None = element_typed.get("type")
                    if isinstance(part_type, str):
                        normalized_type = part_type.lower()
                        if normalized_type in {"reasoning", "tool"}:
                            continue  # Skip reasoning/tool blocks

                    # Extract text content
                    text_value: object | None = element_typed.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                        continue

                    # Check for nested content field
                    nested_value: object | None = element_typed.get("content")
                    if isinstance(nested_value, str):
                        parts.append(nested_value)
                        continue
            return "".join(parts)

        if isinstance(content, Mapping):
            content_typed = cast(Mapping[str, object], content)
            for key in ("text", "content", "message"):
                candidate: object | None = content_typed.get(key)
                if isinstance(candidate, str):
                    return candidate
        return ""

    def _apply_overrides(
        self,
        model_config: ModelConfig,
        overrides: ModelOverrides,
    ) -> tuple[int, float]:
        """Apply parameter overrides to model configuration.

        Args:
            model_config: Base model configuration
            overrides: Optional parameter overrides

        Returns:
            Tuple of (max_tokens, temperature) with overrides applied
        """
        max_tokens = model_config.max_tokens
        max_tokens_override = overrides.get("max_tokens")
        if isinstance(max_tokens_override, int):
            max_tokens = max_tokens_override

        temperature = model_config.temperature
        temperature_override = overrides.get("temperature")
        if isinstance(temperature_override, (int, float)):
            temperature = float(temperature_override)

        return max_tokens, temperature

    async def _rate_limit_request(self) -> None:
        """Ensure minimum time between requests to avoid 429 errors.

        Uses class-level rate limiting to prevent hitting provider rate limits.
        The minimum interval is controlled by _min_request_interval class variable.
        """
        # Skip if no rate limiting configured
        if self._min_request_interval <= 0:
            return

        async with self._request_lock:
            current_time = time.time()

            if self._last_request_time is not None:
                time_since_last = current_time - self._last_request_time
                if time_since_last < self._min_request_interval:
                    sleep_time = self._min_request_interval - time_since_last
                    logger.debug(
                        f"Rate limiting: waiting {sleep_time:.2f}s before next"
                        f" {self.provider_name} request"
                    )
                    await asyncio.sleep(sleep_time)

            # Update the class variable to track last request time
            # Note: This is safe because we're inside the lock
            type(self)._last_request_time = time.time()

    def _handle_rate_limit_error(
        self,
        exc: Exception,
        model_name: str,
        context: str = "request",
    ) -> None:
        """Handle rate limit errors with provider-specific messaging.

        Args:
            exc: The exception to check
            model_name: Name of the model being used
            context: Context string for the error message

        Raises:
            ProviderRateLimitError: If the exception is a 429 rate limit error
        """
        if isinstance(exc, HTTPStatusError) and exc.response.status_code == 429:
            detail = f"{self.provider_name.title()} rate limited the {context}."
            raise ProviderRateLimitError(
                provider=self.provider_name,
                model=model_name,
                status_code=429,
                detail=detail,
            ) from exc

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        **overrides: Unpack[ModelOverrides],
    ) -> str:
        """Generate a response using OpenAI-compatible API.

        Args:
            model_config: Model configuration
            messages: Conversation messages
            **overrides: Optional parameter overrides (max_tokens, temperature)

        Returns:
            Generated response text

        Raises:
            RuntimeError: If client not initialized
            ProviderRateLimitError: If rate limited by provider
        """
        if not self._client:
            raise RuntimeError(
                f"{self.provider_name.title()} client not initialized - check API key"
            )

        max_tokens, temperature = self._apply_overrides(model_config, overrides)

        try:
            await self._rate_limit_request()

            response = self._client.chat.completions.create(
                model=model_config.name,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = ""
            if response.choices:
                first_choice = response.choices[0]
                message = first_choice.message
                if message.content:
                    content = self._coerce_content_to_text(message.content)

            if not content.strip():
                logger.warning(
                    "%s model %s returned empty content",
                    self.provider_name.title(),
                    model_config.name,
                )
            else:
                logger.debug(
                    "Generated %s chars from %s model %s",
                    len(content),
                    self.provider_name,
                    model_config.name,
                )

            return content.strip()

        except Exception as exc:
            self._handle_rate_limit_error(exc, model_config.name, "request")
            logger.error(
                "%s generation failed for %s: %s",
                self.provider_name.title(),
                model_config.name,
                exc,
            )
            raise

    def supports_streaming(self) -> bool:
        """All OpenAI-compatible providers support streaming."""
        return True

    async def generate_response_stream(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        chunk_callback: ChunkCallback,
        **overrides: Unpack[ModelOverrides],
    ) -> str:
        """Generate a streaming response using OpenAI-compatible API.

        Args:
            model_config: Model configuration
            messages: Conversation messages
            chunk_callback: Callback for streaming chunks
            **overrides: Optional parameter overrides

        Returns:
            Complete generated response text

        Raises:
            RuntimeError: If client not initialized
            ProviderRateLimitError: If rate limited by provider
        """
        if not self._client:
            raise RuntimeError(
                f"{self.provider_name.title()} client not initialized - check API key"
            )

        max_tokens, temperature = self._apply_overrides(model_config, overrides)

        try:
            await self._rate_limit_request()

            complete_content = ""

            # Use OpenAI SDK streaming
            # Note: OpenAI SDK streaming types are not fully typed
            with self._client.chat.completions.create(  # type: ignore[call-overload]
                model=model_config.name,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ) as raw_stream:  # type: ignore[attr-defined]
                stream = cast(Stream[ChatCompletionChunk], raw_stream)
                for chunk in stream:
                    if not chunk.choices:
                        continue

                    first_choice = chunk.choices[0]
                    delta = first_choice.delta

                    content = delta.content
                    if content:
                        chunk_text = self._coerce_content_to_text(content)
                        if chunk_text:
                            complete_content += chunk_text
                            await chunk_callback(chunk_text, False)

                    # Check for completion
                    if first_choice.finish_reason:
                        await chunk_callback("", True)
                        break

            logger.debug(
                "%s streaming completed: %s chars from %s",
                self.provider_name.title(),
                len(complete_content),
                model_config.name,
            )
            return complete_content.strip()

        except Exception as exc:
            self._handle_rate_limit_error(exc, model_config.name, "streaming request")
            logger.error(
                "%s streaming failed for %s: %s",
                self.provider_name.title(),
                model_config.name,
                exc,
            )
            raise

    async def generate_response_with_metadata(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        **overrides: Unpack[ModelOverrides],
    ) -> GenerationMetadata:
        """Generate response with full metadata including tokens and cost.

        Args:
            model_config: Model configuration
            messages: Conversation messages
            **overrides: Optional parameter overrides

        Returns:
            GenerationMetadata with content, tokens, cost, and timing

        Raises:
            RuntimeError: If client not initialized
            ProviderRateLimitError: If rate limited by provider
        """
        if not self._client:
            raise RuntimeError(
                f"{self.provider_name.title()} client not initialized - check API key"
            )

        max_tokens, temperature = self._apply_overrides(model_config, overrides)

        try:
            await self._rate_limit_request()

            # Measure provider-level timing (just the API call)
            start_time = time.time()

            response = self._client.chat.completions.create(
                model=model_config.name,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )

            generation_time_ms = int((time.time() - start_time) * 1000)

            content = ""
            if response.choices:
                first_choice = response.choices[0]
                message = first_choice.message
                if message.content:
                    content = self._coerce_content_to_text(message.content)

            # Extract token usage from response
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None
            cost: float | None = None

            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Calculate cost if provider implements cost calculation
                cost = await self._calculate_cost(
                    model_config.name, prompt_tokens, completion_tokens
                )

            logger.debug(
                "Generated %s chars from %s %s, tokens: %s, cost: $%.6f",
                len(content),
                self.provider_name,
                model_config.name,
                total_tokens,
                cost if cost else 0,
            )

            return GenerationMetadata(
                content=content,
                generation_id=response.id if hasattr(response, "id") else None,
                cost=cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time_ms,
                model=model_config.name,
                provider=self.provider_name,
            )

        except Exception as exc:
            self._handle_rate_limit_error(exc, model_config.name, "metadata request")
            logger.error(
                "%s metadata generation failed for %s: %s",
                self.provider_name.title(),
                model_config.name,
                exc,
            )
            raise

    async def generate_response_stream_with_metadata(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        chunk_callback: ChunkCallback,
        **overrides: Unpack[ModelOverrides],
    ) -> GenerationMetadata:
        """Generate streaming response with metadata.

        Note: Token counts and cost are estimated for streaming responses
        since the OpenAI SDK doesn't provide usage data in stream mode.

        Args:
            model_config: Model configuration
            messages: Conversation messages
            chunk_callback: Callback for streaming chunks
            **overrides: Optional parameter overrides

        Returns:
            GenerationMetadata with estimated tokens and cost
        """
        start_time = time.time()

        # Stream the content
        content = await self.generate_response_stream(
            model_config, messages, chunk_callback, **overrides
        )

        generation_time_ms = int((time.time() - start_time) * 1000)

        # Estimate token counts (rough approximation: ~4 chars per token)
        estimated_completion_tokens = len(content) // 4
        estimated_prompt_tokens = sum(len(msg["content"]) for msg in messages) // 4
        estimated_total = estimated_prompt_tokens + estimated_completion_tokens

        # Calculate estimated cost
        cost = await self._calculate_cost(
            model_config.name, estimated_prompt_tokens, estimated_completion_tokens
        )

        logger.debug(
            "%s streaming metadata: %s chars, est. tokens: %s, est. cost: $%.6f",
            self.provider_name.title(),
            len(content),
            estimated_total,
            cost if cost else 0,
        )

        return GenerationMetadata(
            content=content,
            generation_id=None,  # Not available for streaming
            cost=cost,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_total,
            generation_time_ms=generation_time_ms,
            model=model_config.name,
            provider=self.provider_name,
        )

    async def _calculate_cost(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """Calculate cost based on token usage and model pricing.

        Default implementation returns None. Subclasses should override
        this method if they have pricing information available.

        Args:
            model_name: Name of the model
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD, or None if pricing not available
        """
        return None

    async def is_running(self) -> bool:
        """Check if provider is running/available.

        Default implementation returns True (assume provider is available).
        Subclasses can override for custom health checks (e.g., Ollama).

        Returns:
            True if provider is available, False otherwise
        """
        return True

    async def query_generation_cost(self, generation_id: str) -> float | None:
        """Query the cost for a specific generation ID.

        Default implementation returns None. Subclasses can override
        if they provide a post-generation cost query API (e.g., OpenRouter).

        Args:
            generation_id: The generation ID to query

        Returns:
            Cost in USD, or None if not available
        """
        return None

    # Abstract methods that subclasses must implement
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider (e.g., 'ollama', 'anthropic')."""
