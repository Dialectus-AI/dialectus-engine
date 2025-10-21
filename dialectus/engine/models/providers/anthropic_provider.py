"""Anthropic provider implementation using OpenAI SDK."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar, cast

from openai import OpenAI

from dialectus.engine.models.base_types import BaseEnhancedModelInfo

from .openai_compatible_provider import OpenAICompatibleProviderBase

if TYPE_CHECKING:
    from dialectus.engine.config.settings import ModelConfig, SystemConfig

logger = logging.getLogger(__name__)


class AnthropicProvider(OpenAICompatibleProviderBase):
    """Anthropic model provider using OpenAI SDK compatibility.

    Uses Anthropic's OpenAI-compatible API endpoint for all operations.
    Provides hardcoded model catalog with pricing information for cost tracking.
    """

    # Class-level rate limiting to prevent 429 errors (3 seconds between requests)
    _min_request_interval: ClassVar[float] = 3.0

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

    def validate_model_config(self, model_config: ModelConfig) -> bool:
        """Validate Anthropic model configuration."""
        return model_config.provider == "anthropic"

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
