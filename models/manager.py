"""Model manager with multi-provider support."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Protocol, TypeAlias, cast

from config.settings import ModelConfig, SystemConfig
from models.base_types import (
    BaseEnhancedModelInfo,
    ModelPricing,
    ModelTier,
    ModelWeightClass,
    SourceInfo,
)

from .providers.base_model_provider import BaseModelProvider, GenerationMetadata
from .providers.providers import ProviderFactory

ChunkCallback: TypeAlias = Callable[[str, bool], Awaitable[None]]
MessageDict: TypeAlias = dict[str, str]
MessageList: TypeAlias = list[MessageDict]
EnhancedModelRecord: TypeAlias = dict[str, object]
ModelCatalog: TypeAlias = dict[str, list[str]]

BLACKLISTED_MODELS: frozenset[str] = frozenset(
    {
        "meta-llama/llama-3.2-11b-vision-instruct",
    }
)


class EnhancedModelProvider(Protocol):
    """Protocol for providers that support enhanced model information."""

    async def get_enhanced_models(self) -> list[BaseEnhancedModelInfo]:
        """Return enhanced models for the provider."""
        ...


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model inference via multiple providers."""

    def __init__(self, system_config: SystemConfig):
        self._system_config = system_config
        self._model_configs: dict[str, ModelConfig] = {}
        self._providers: dict[str, BaseModelProvider] = {}

    def _get_provider(self, provider_name: str) -> BaseModelProvider:
        """Return (and cache) the provider instance identified by name."""
        if provider_name not in self._providers:
            self._providers[provider_name] = ProviderFactory.create_provider(
                provider_name, self._system_config
            )
        return self._providers[provider_name]

    def register_model(self, model_id: str, config: ModelConfig) -> None:
        """Register a model configuration for quick lookup."""
        try:
            provider = self._get_provider(config.provider)
            if not provider.validate_model_config(config):
                msg = f"Invalid model config for provider {config.provider}"
                raise ValueError(msg)
        except ValueError as exc:
            logger.error("Failed to register model %s: %s", model_id, exc)
            raise

        self._model_configs[model_id] = config
        logger.info("Registered model %s: %s (%s)", model_id, config.name, config.provider)

    async def generate_response(
        self, model_id: str, messages: MessageList, **overrides: object
    ) -> str:
        """Generate a response from the specified model."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        response = await provider.generate_response(config, messages, **overrides)
        logger.debug(
            "Generated %s chars from %s (%s)", len(response), model_id, config.provider
        )
        return response

    async def generate_response_stream(
        self,
        model_id: str,
        messages: MessageList,
        chunk_callback: ChunkCallback,
        **overrides: object,
    ) -> str:
        """Generate a streaming response from the specified model."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        if provider.supports_streaming():
            response = await provider.generate_response_stream(
                config, messages, chunk_callback, **overrides
            )
            logger.debug(
                "Generated %s chars via streaming from %s (%s)",
                len(response),
                model_id,
                config.provider,
            )
            return response

        response = await provider.generate_response(config, messages, **overrides)
        await chunk_callback(response, True)
        logger.debug(
            "Generated %s chars via fallback non-streaming from %s (%s)",
            len(response),
            model_id,
            config.provider,
        )
        return response

    @asynccontextmanager
    async def model_session(self, model_id: str) -> AsyncIterator["ModelManager"]:
        """Create a session context for model operations."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        logger.debug("Starting session for model %s", model_id)
        try:
            yield self
        finally:
            logger.debug("Ending session for model %s", model_id)

    async def get_available_models(self) -> ModelCatalog:
        """Get list of available models from all providers."""
        all_models: ModelCatalog = {}

        for provider_name in ProviderFactory.get_available_providers():
            try:
                provider = self._get_provider(provider_name)
                models = await provider.get_available_models()
                all_models[provider_name] = models
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to get models from %s: %s", provider_name, exc)
                all_models[provider_name] = []

        return all_models

    async def get_enhanced_models(self) -> list[EnhancedModelRecord]:
        """Get enhanced model information with metadata, filtering, and classification."""
        typed_models = await self.get_enhanced_models_typed()

        enhanced_models: list[EnhancedModelRecord] = []
        for model in typed_models:
            description = (
                f"{model.description[:200]}..."
                if len(model.description) > 200
                else model.description
            )
            enhanced_models.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "description": description,
                    "weight_class": model.weight_class.value,
                    "tier": model.tier.value,
                    "context_length": model.context_length,
                    "max_completion_tokens": model.max_completion_tokens,
                    "pricing": {
                        "prompt_cost_per_1k": model.pricing.prompt_cost_per_1k,
                        "completion_cost_per_1k": model.pricing.completion_cost_per_1k,
                        "avg_cost_per_1k": model.pricing.avg_cost_per_1k,
                        "is_free": model.pricing.is_free,
                    },
                    "value_score": round(model.value_score, 2),
                    "is_preview": model.is_preview,
                    "is_text_only": model.is_text_only,
                    "estimated_params": model.estimated_params,
                    "display_name": model.display_name,
                }
            )

        return enhanced_models

    async def get_enhanced_models_typed(self) -> list[BaseEnhancedModelInfo]:
        """Get enhanced model information as properly typed objects for internal use."""
        enhanced_models: list[BaseEnhancedModelInfo] = []

        for provider_name in ProviderFactory.get_available_providers():
            try:
                provider = self._get_provider(provider_name)

                if hasattr(provider, "get_enhanced_models"):
                    enhanced_provider = cast(EnhancedModelProvider, provider)
                    provider_enhanced = await enhanced_provider.get_enhanced_models()
                    enhanced_models.extend(provider_enhanced)
                    continue

                basic_models = await provider.get_available_models()
                for model_id in basic_models:
                    fallback_model = BaseEnhancedModelInfo(
                        id=model_id,
                        name=model_id,
                        provider=provider_name,
                        description="Standard model",
                        weight_class=ModelWeightClass.MIDDLEWEIGHT,
                        tier=ModelTier.BALANCED,
                        context_length=4096,
                        max_completion_tokens=1024,
                        pricing=ModelPricing(
                            prompt_cost_per_1k=0.0,
                            completion_cost_per_1k=0.0,
                            is_free=True,
                            currency="USD",
                        ),
                        value_score=5.0,
                        is_preview=False,
                        is_text_only=True,
                        estimated_params=None,
                        source_info=cast(SourceInfo, {}),
                    )
                    enhanced_models.append(fallback_model)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to get enhanced models from %s: %s", provider_name, exc
                )

        filtered_models = [
            model for model in enhanced_models if model.id not in BLACKLISTED_MODELS
        ]

        if len(filtered_models) != len(enhanced_models):
            blacklisted_count = len(enhanced_models) - len(filtered_models)
            logger.info(
                "Filtered out %s blacklisted model(s) with problematic safety filters",
                blacklisted_count,
            )

        return filtered_models

    async def generate_response_with_metadata(
        self, model_id: str, messages: MessageList, **overrides: object
    ) -> GenerationMetadata:
        """Generate a response with full metadata for cost tracking."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        metadata = await provider.generate_response_with_metadata(
            config, messages, **overrides
        )
        logger.debug(
            "Generated %s chars with metadata from %s (%s), generation_id: %s",
            len(metadata.content),
            model_id,
            config.provider,
            metadata.generation_id,
        )
        return metadata

    async def generate_response_stream_with_metadata(
        self,
        model_id: str,
        messages: MessageList,
        chunk_callback: ChunkCallback,
        **overrides: object,
    ) -> GenerationMetadata:
        """Generate a streaming response with full metadata for cost tracking."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        metadata = await provider.generate_response_stream_with_metadata(
            config, messages, chunk_callback, **overrides
        )
        logger.debug(
            "Generated %s chars via streaming with metadata from %s (%s), generation_id: %s",
            len(metadata.content),
            model_id,
            config.provider,
            metadata.generation_id,
        )
        return metadata

    async def query_generation_cost(
        self, model_id: str, generation_id: str
    ) -> float | None:
        """Query the cost for a specific generation using the appropriate provider."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        return await provider.query_generation_cost(generation_id)
