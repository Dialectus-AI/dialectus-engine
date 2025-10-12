"""Model manager with multi-provider support."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Protocol, cast

from dialectus.engine.config.settings import ModelConfig, SystemConfig
from dialectus.engine.models.base_types import BaseEnhancedModelInfo

from .providers.base_model_provider import BaseModelProvider, GenerationMetadata
from .providers.providers import ProviderFactory

type ChunkCallback = Callable[[str, bool], Awaitable[None]]
type MessageDict = dict[str, str]
type MessageList = list[MessageDict]
type EnhancedModelRecord = dict[str, object]
type ModelCatalog = dict[str, list[str]]

BLACKLISTED_MODELS: frozenset[str] = frozenset(
    {
        # Meta routes this variant through heavy-handed safety filters
        # that break debate flow.
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

    def _require_model_config(self, model_id: str) -> ModelConfig:
        """Return the registered config or raise a uniform not-registered error."""
        try:
            return self._model_configs[model_id]
        except KeyError as exc:
            raise ValueError(f"Model {model_id} not registered") from exc

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
        logger.info(
            "Registered model %s: %s (%s)", model_id, config.name, config.provider
        )

    async def generate_response(
        self, model_id: str, messages: MessageList, **overrides: object
    ) -> str:
        """Generate a response from the specified model."""
        config = self._require_model_config(model_id)
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
        config = self._require_model_config(model_id)
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

        # Providers that cannot stream still call the callback once so
        # the caller can reuse the same code path.
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
    async def model_session(self, model_id: str) -> AsyncIterator[ModelManager]:
        """Create a session context for model operations."""
        self._require_model_config(model_id)

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
            except Exception as exc:
                logger.error("Failed to get models from %s: %s", provider_name, exc)
                raise

        return all_models

    async def get_enhanced_models(self) -> list[EnhancedModelRecord]:
        """Get enhanced model info with metadata, filtering, classification."""
        typed_models = await self.get_enhanced_models_typed()

        enhanced_models: list[EnhancedModelRecord] = []
        for model in typed_models:
            enhanced_models.append(
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
                # Each provider must surface rich metadata; bail out if
                # a new provider forgets.
                if not hasattr(provider, "get_enhanced_models"):
                    raise NotImplementedError(
                        f"Provider {provider_name} does not expose enhanced metadata"
                    )

                enhanced_provider = cast(EnhancedModelProvider, provider)
                provider_enhanced = await enhanced_provider.get_enhanced_models()
                enhanced_models.extend(provider_enhanced)

            except Exception as exc:
                logger.error(
                    "Failed to get enhanced models from %s: %s", provider_name, exc
                )
                raise

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
        config = self._require_model_config(model_id)
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
        config = self._require_model_config(model_id)
        provider = self._get_provider(config.provider)

        metadata = await provider.generate_response_stream_with_metadata(
            config, messages, chunk_callback, **overrides
        )
        logger.debug(
            "Generated %s chars via streaming with metadata from %s (%s),"
            " generation_id: %s",
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
        config = self._require_model_config(model_id)
        provider = self._get_provider(config.provider)

        return await provider.query_generation_cost(generation_id)
