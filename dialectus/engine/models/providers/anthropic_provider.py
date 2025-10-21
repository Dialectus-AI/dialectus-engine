"""Anthropic provider implementation using OpenAI SDK."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Unpack, cast

from httpx import HTTPStatusError
from openai import OpenAI

from dialectus.engine.models.base_types import BaseEnhancedModelInfo

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


class AnthropicProvider(BaseModelProvider):
    """Anthropic model provider using OpenAI SDK compatibility."""

    # Class-level rate limiting to prevent 429 errors
    _last_request_time: ClassVar[float | None] = None
    _request_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _min_request_interval: ClassVar[float] = 3.0  # 3 seconds between requests

    def __init__(self, system_config: SystemConfig):
        super().__init__(system_config)

        # Get API key from config or environment
        api_key = self._get_api_key()

        if not api_key:
            logger.warning(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or configure in"
                " system settings."
            )
            self._client = None
        else:
            # Initialize OpenAI client pointing to Anthropic
            self._client = OpenAI(
                base_url=system_config.anthropic.base_url,
                api_key=api_key,
                timeout=system_config.anthropic.timeout,
                max_retries=system_config.anthropic.max_retries,
            )

    def _get_api_key(self) -> str | None:
        """Get Anthropic API key from environment or config.

        Returns:
            API key string if found, None otherwise
        """
        return os.getenv("ANTHROPIC_API_KEY") or self.system_config.anthropic.api_key

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def _rate_limit_request(self) -> None:
        """Ensure minimum time between requests to avoid 429 errors."""
        async with self._request_lock:
            current_time = time.time()

            if self._last_request_time is not None:
                time_since_last = current_time - self._last_request_time
                if time_since_last < self._min_request_interval:
                    sleep_time = self._min_request_interval - time_since_last
                    logger.debug(
                        f"Rate limiting: waiting {sleep_time:.2f}s before next"
                        " Anthropic request"
                    )
                    await asyncio.sleep(sleep_time)

            # Update the class variable to track last request time
            AnthropicProvider._last_request_time = time.time()

    @staticmethod
    def _coerce_content_to_text(content: object) -> str:
        """Return textual content from OpenAI-style message content payloads."""
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
                    text_value: object | None = element_typed.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                        continue
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

    async def get_available_models(self) -> list[str]:
        """Get curated list of Anthropic models.

        Returns model IDs from our hardcoded catalog.
        """
        enhanced_models = await self.get_enhanced_models()
        return [model.id for model in enhanced_models]

    async def get_enhanced_models(self) -> list[BaseEnhancedModelInfo]:
        """Get enhanced model information with filtering and classification."""
        if not self._client:
            logger.warning("Anthropic client not initialized - no API key")
            return []

        try:
            from dialectus.engine.models.cache_manager import cache_manager
            from dialectus.engine.models.anthropic.anthropic_enhanced_model_info import (
                AnthropicEnhancedModelInfo,
            )
            from dialectus.engine.models.anthropic.anthropic_model_filter import (
                AnthropicModelFilter,
            )

            # Check cache first (6 hour default TTL)
            cached_models = cache_manager.get("anthropic", "models")
            enhanced_models: list[AnthropicEnhancedModelInfo] = []
            if isinstance(cached_models, list):
                cached_list = cast(list[object], cached_models)
                for model_candidate in cached_list:
                    if not isinstance(model_candidate, dict):
                        continue
                    model_dict = cast(dict[str, object], model_candidate)
                    try:
                        enhanced_model = AnthropicEnhancedModelInfo(**model_dict)
                        enhanced_models.append(enhanced_model)
                    except Exception as exc:
                        model_id = str(model_dict.get("id", "unknown"))
                        logger.warning(
                            "Failed to reconstruct cached Anthropic model %s: %s",
                            model_id,
                            exc,
                        )
                        continue

                if enhanced_models:
                    logger.info(
                        "Using cached Anthropic models (%s models)",
                        len(enhanced_models),
                    )
                    return cast(list[BaseEnhancedModelInfo], enhanced_models)

                logger.warning(
                    "All cached models failed to reconstruct, using fresh data..."
                )
                cache_manager.invalidate("anthropic", "models")
            elif cached_models is not None:
                logger.warning(
                    "Anthropic model cache contains unexpected type: %s",
                    type(cached_models).__name__,
                )

            # Generate fresh enhanced models from hardcoded catalog
            logger.info("Generating Anthropic models from curated catalog...")
            enhanced_models = AnthropicModelFilter.filter_and_enhance_models()

            # Cache enhanced models (6 hours - models don't change frequently)
            serializable_models = [
                {
                    "id": m.id,
                    "name": m.name,
                    "provider": m.provider,
                    "description": m.description,
                    "weight_class": m.weight_class.value,
                    "tier": m.tier.value,
                    "context_length": m.context_length,
                    "max_completion_tokens": m.max_completion_tokens,
                    "pricing": {
                        "prompt_cost_per_1k": m.pricing.prompt_cost_per_1k,
                        "completion_cost_per_1k": m.pricing.completion_cost_per_1k,
                        "is_free": m.pricing.is_free,
                        "currency": m.pricing.currency,
                    },
                    "value_score": m.value_score,
                    "is_preview": m.is_preview,
                    "is_text_only": m.is_text_only,
                    "estimated_params": m.estimated_params,
                    "source_info": dict(m.source_info) if m.source_info else {},
                }
                for m in enhanced_models
            ]
            cache_manager.set("anthropic", "models", serializable_models, ttl_hours=6)

            logger.info(
                f"Anthropic: Generated and cached {len(enhanced_models)} curated models"
            )
            return cast(list[BaseEnhancedModelInfo], enhanced_models)

        except Exception as e:
            logger.error(f"Failed to get enhanced Anthropic models: {e}")
            raise

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        **overrides: Unpack[ModelOverrides],
    ) -> str:
        """Generate a response using Anthropic via OpenAI SDK."""
        if not self._client:
            raise RuntimeError("Anthropic client not initialized - check API key")

        max_tokens = model_config.max_tokens
        max_tokens_override = overrides.get("max_tokens")
        if isinstance(max_tokens_override, int):
            max_tokens = max_tokens_override

        temperature = model_config.temperature
        temperature_override = overrides.get("temperature")
        if isinstance(temperature_override, (int, float)):
            temperature = float(temperature_override)

        try:
            await self._rate_limit_request()

            # Use OpenAI SDK to call Anthropic
            response = self._client.chat.completions.create(
                model=model_config.name,
                messages=messages,  # type: ignore
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
                    "Anthropic model %s returned empty content", model_config.name
                )
            else:
                logger.debug(
                    "Generated %s chars from Anthropic model %s",
                    len(content),
                    model_config.name,
                )

            return content.strip()

        except Exception as exc:
            # Check for rate limit errors
            if isinstance(exc, HTTPStatusError) and exc.response.status_code == 429:
                detail = "Anthropic rate limited the request."
                raise ProviderRateLimitError(
                    provider="anthropic",
                    model=model_config.name,
                    status_code=429,
                    detail=detail,
                ) from exc

            logger.error(
                "Anthropic generation failed for %s: %s", model_config.name, exc
            )
            raise

    def validate_model_config(self, model_config: ModelConfig) -> bool:
        """Validate Anthropic model configuration."""
        return model_config.provider == "anthropic"

    def supports_streaming(self) -> bool:
        """Anthropic supports streaming responses."""
        return True

    async def generate_response_stream(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        chunk_callback: ChunkCallback,
        **overrides: Unpack[ModelOverrides],
    ) -> str:
        """Generate a streaming response using Anthropic."""
        if not self._client:
            raise RuntimeError("Anthropic client not initialized - check API key")

        max_tokens = model_config.max_tokens
        max_tokens_override = overrides.get("max_tokens")
        if isinstance(max_tokens_override, int):
            max_tokens = max_tokens_override

        temperature = model_config.temperature
        temperature_override = overrides.get("temperature")
        if isinstance(temperature_override, (int, float)):
            temperature = float(temperature_override)

        try:
            await self._rate_limit_request()

            complete_content = ""

            # Use OpenAI SDK streaming
            # Note: OpenAI SDK streaming types are not fully typed, hence the type: ignore comments
            with self._client.chat.completions.create(  # type: ignore[call-overload]
                model=model_config.name,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ) as stream:  # type: ignore[attr-defined]
                for chunk in stream:  # type: ignore[attr-defined]
                    if not chunk.choices:  # type: ignore[attr-defined]
                        continue

                    first_choice = chunk.choices[0]  # type: ignore[attr-defined]
                    delta = first_choice.delta  # type: ignore[attr-defined]

                    if hasattr(delta, "content") and delta.content:  # type: ignore[attr-defined]
                        chunk_text = self._coerce_content_to_text(delta.content)  # type: ignore[attr-defined]
                        if chunk_text:
                            complete_content += chunk_text
                            await chunk_callback(chunk_text, False)

                    # Check for completion
                    if hasattr(first_choice, "finish_reason") and first_choice.finish_reason:  # type: ignore[attr-defined]
                        await chunk_callback("", True)
                        break

            logger.debug(
                "Anthropic streaming completed: %s chars from %s",
                len(complete_content),
                model_config.name,
            )
            return complete_content.strip()

        except Exception as exc:
            if isinstance(exc, HTTPStatusError) and exc.response.status_code == 429:
                detail = "Anthropic rate limited the streaming request."
                raise ProviderRateLimitError(
                    provider="anthropic",
                    model=model_config.name,
                    status_code=429,
                    detail=detail,
                ) from exc

            logger.error(
                "Anthropic streaming failed for %s: %s", model_config.name, exc
            )
            raise

    async def generate_response_with_metadata(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        **overrides: Unpack[ModelOverrides],
    ) -> GenerationMetadata:
        """Generate response with full metadata including token usage and cost."""
        if not self._client:
            raise RuntimeError("Anthropic client not initialized - check API key")

        max_tokens = model_config.max_tokens
        max_tokens_override = overrides.get("max_tokens")
        if isinstance(max_tokens_override, int):
            max_tokens = max_tokens_override

        temperature = model_config.temperature
        temperature_override = overrides.get("temperature")
        if isinstance(temperature_override, (int, float)):
            temperature = float(temperature_override)

        try:
            await self._rate_limit_request()

            start_time = time.time()

            response = self._client.chat.completions.create(
                model=model_config.name,
                messages=messages,  # type: ignore
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

            # Extract token usage
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None
            cost: float | None = None

            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Calculate cost based on model pricing
                cost = await self._calculate_cost(
                    model_config.name, prompt_tokens, completion_tokens
                )

            logger.debug(
                "Generated %s chars from Anthropic %s, tokens: %s, cost: $%.6f",
                len(content),
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
                provider="anthropic",
            )

        except Exception as exc:
            if isinstance(exc, HTTPStatusError) and exc.response.status_code == 429:
                detail = "Anthropic rate limited the metadata request."
                raise ProviderRateLimitError(
                    provider="anthropic",
                    model=model_config.name,
                    status_code=429,
                    detail=detail,
                ) from exc

            logger.error(
                "Anthropic metadata generation failed for %s: %s",
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

        Note: Token counts and cost are estimated for streaming responses.
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
            "Anthropic streaming metadata: %s chars, est. tokens: %s, est. cost: $%.6f",
            len(content),
            estimated_total,
            cost if cost else 0,
        )

        return GenerationMetadata(
            content=content,
            generation_id=None,
            cost=cost,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_total,
            generation_time_ms=generation_time_ms,
            model=model_config.name,
            provider="anthropic",
        )

    async def _calculate_cost(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """Calculate cost based on token usage and model pricing."""
        try:
            from dialectus.engine.models.anthropic.anthropic_model import (
                ANTHROPIC_MODELS,
            )

            # Find the model in our catalog
            model_pricing = None
            for model in ANTHROPIC_MODELS:
                if model.id == model_name:
                    model_pricing = model.pricing
                    break

            if not model_pricing:
                logger.warning(
                    "No pricing found for model %s, cannot calculate cost", model_name
                )
                return None

            # Calculate cost: (tokens / 1000) * cost_per_1k
            prompt_cost = (prompt_tokens / 1000) * model_pricing.prompt_cost_per_1k
            completion_cost = (
                completion_tokens / 1000
            ) * model_pricing.completion_cost_per_1k
            total_cost = prompt_cost + completion_cost

            return total_cost

        except Exception as exc:
            logger.warning("Failed to calculate cost for %s: %s", model_name, exc)
            return None

    async def query_generation_cost(self, generation_id: str) -> float | None:
        """Query cost for a specific generation.

        Anthropic doesn't provide a post-generation cost query API,
        so we return None. Costs should be calculated during generation.
        """
        logger.debug(
            "Anthropic does not support post-generation cost queries for ID %s",
            generation_id,
        )
        return None
