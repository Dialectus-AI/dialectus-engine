"""OpenAI provider implementation using the OpenAI SDK."""

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


class OpenAIProvider(OpenAICompatibleProviderBase):
    """Direct OpenAI provider using the native OpenAI API."""

    # Gentle rate limiting to avoid accidental 429s on shared keys
    _min_request_interval: ClassVar[float] = 1.0

    def __init__(self, system_config: SystemConfig):
        super().__init__(system_config)

        api_key = self._get_api_key()
        if not api_key:
            logger.warning(
                "No OpenAI API key found. Set OPENAI_API_KEY or configure in system settings."
            )
            self._client = None
        else:
            self._client = OpenAI(
                base_url=system_config.openai.base_url,
                api_key=api_key,
                timeout=system_config.openai.timeout,
                max_retries=system_config.openai.max_retries,
            )

    def _get_api_key(self) -> str | None:
        """Return OpenAI API key from environment or configuration."""
        return os.getenv("OPENAI_API_KEY") or self.system_config.openai.api_key

    @property
    def provider_name(self) -> str:
        return "openai"

    async def get_available_models(self) -> list[str]:
        enhanced_models = await self.get_enhanced_models()
        return [model.id for model in enhanced_models]

    async def get_enhanced_models(self) -> list[BaseEnhancedModelInfo]:
        if not self._client:
            logger.warning("OpenAI client not initialized - no API key")
            return []

        try:
            from dialectus.engine.models.cache_manager import cache_manager
            from dialectus.engine.models.openai.openai_enhanced_model_info import (
                OpenAIEnhancedModelInfo,
            )
            from dialectus.engine.models.openai.openai_model_filter import (
                OpenAIModelFilter,
            )

            cached_models = cache_manager.get("openai", "models")
            enhanced_models: list[OpenAIEnhancedModelInfo] = []

            if isinstance(cached_models, list):
                cached_list = cast(list[object], cached_models)
                for model_candidate in cached_list:
                    if not isinstance(model_candidate, dict):
                        continue
                    model_dict = cast(dict[str, object], model_candidate)
                    try:
                        enhanced_model = OpenAIEnhancedModelInfo(**model_dict)
                        enhanced_models.append(enhanced_model)
                    except Exception as exc:
                        model_id = str(model_dict.get("id", "unknown"))
                        logger.warning(
                            "Failed to reconstruct cached OpenAI model %s: %s",
                            model_id,
                            exc,
                        )
                        continue

                if enhanced_models:
                    logger.info(
                        "Using cached OpenAI models (%s models)", len(enhanced_models)
                    )
                    return cast(list[BaseEnhancedModelInfo], enhanced_models)

                logger.warning(
                    "All cached OpenAI models failed to reconstruct, generating fresh data..."
                )
                cache_manager.invalidate("openai", "models")
            elif cached_models is not None:
                logger.warning(
                    "OpenAI model cache contains unexpected type: %s",
                    type(cached_models).__name__,
                )

            logger.info("Generating OpenAI models from curated catalog...")
            enhanced_models = OpenAIModelFilter.filter_and_enhance_models()

            serializable_models = [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "description": model.description,
                    "weight_class": model.weight_class.value,
                    "tier": model.tier.value,
                    "context_length": model.context_length,
                    "max_completion_tokens": model.max_completion_tokens,
                    "pricing": {
                        "prompt_cost_per_1k": model.pricing.prompt_cost_per_1k,
                        "completion_cost_per_1k": model.pricing.completion_cost_per_1k,
                        "is_free": model.pricing.is_free,
                        "currency": model.pricing.currency,
                    },
                    "value_score": model.value_score,
                    "is_preview": model.is_preview,
                    "is_text_only": model.is_text_only,
                    "estimated_params": model.estimated_params,
                    "source_info": dict(model.source_info) if model.source_info else {},
                }
                for model in enhanced_models
            ]
            cache_manager.set("openai", "models", serializable_models, ttl_hours=6)

            logger.info(
                "OpenAI: Generated and cached %s curated models",
                len(enhanced_models),
            )
            return cast(list[BaseEnhancedModelInfo], enhanced_models)
        except Exception as exc:
            logger.error("Failed to get enhanced OpenAI models: %s", exc)
            raise

    def validate_model_config(self, model_config: ModelConfig) -> bool:
        return model_config.provider == "openai"

    async def _calculate_cost(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        try:
            from dialectus.engine.models.openai.openai_model import OPENAI_MODELS

            pricing = None
            for model in OPENAI_MODELS:
                if model.id == model_name:
                    pricing = model.pricing
                    break

            if pricing is None:
                logger.warning(
                    "No pricing found for OpenAI model %s, cannot calculate cost",
                    model_name,
                )
                return None

            prompt_cost = (prompt_tokens / 1000) * pricing.prompt_cost_per_1k
            completion_cost = (completion_tokens / 1000) * pricing.completion_cost_per_1k
            return prompt_cost + completion_cost
        except Exception as exc:
            logger.warning("Failed to calculate cost for %s: %s", model_name, exc)
            return None
