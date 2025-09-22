"""Model manager with multi-provider support."""

from typing import TYPE_CHECKING, Any, Protocol, cast, Callable, Awaitable
import logging
from contextlib import asynccontextmanager
from config.settings import ModelConfig, SystemConfig
from .providers.base_model_provider import BaseModelProvider, GenerationMetadata
from .providers.providers import ProviderFactory

if TYPE_CHECKING:
    from models.base_types import BaseEnhancedModelInfo
else:
    from .base_types import BaseEnhancedModelInfo


class EnhancedModelProvider(Protocol):
    """Protocol for providers that support enhanced model information."""

    async def get_enhanced_models(self) -> list[BaseEnhancedModelInfo]: ...


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model inference via multiple providers."""

    def __init__(self, system_config: SystemConfig):
        self._system_config = system_config
        self._model_configs: dict[str, ModelConfig] = {}
        self._providers: dict[str, BaseModelProvider] = {}

        # Initialize providers lazily as needed

    def _get_provider(self, provider_name: str) -> BaseModelProvider:
        """Get or create a provider instance."""
        if provider_name not in self._providers:
            self._providers[provider_name] = ProviderFactory.create_provider(
                provider_name, self._system_config
            )
        return self._providers[provider_name]

    def register_model(self, model_id: str, config: ModelConfig) -> None:
        """Register a model configuration."""
        # Validate the provider exists
        try:
            provider = self._get_provider(config.provider)
            if not provider.validate_model_config(config):
                raise ValueError(f"Invalid model config for provider {config.provider}")
        except ValueError as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            raise

        self._model_configs[model_id] = config
        logger.info(f"Registered model {model_id}: {config.name} ({config.provider})")

    async def generate_response(
        self, model_id: str, messages: list[dict[str, str]], **overrides
    ) -> str:
        """Generate a response from the specified model."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        try:
            response = await provider.generate_response(config, messages, **overrides)
            logger.debug(
                f"Generated {len(response)} chars from {model_id} ({config.provider})"
            )
            return response

        except Exception as e:
            logger.error(f"Generation failed for {model_id} ({config.provider}): {e}")
            raise

    async def generate_response_stream(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        chunk_callback: Callable[[str, bool], Awaitable[None]],
        **overrides
    ) -> str:
        """Generate a streaming response from the specified model."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        try:
            # Check if provider supports streaming
            if provider.supports_streaming():
                response = await provider.generate_response_stream(
                    config, messages, chunk_callback, **overrides
                )
                logger.debug(
                    f"Generated {len(response)} chars via streaming from {model_id} ({config.provider})"
                )
            else:
                # Fallback to non-streaming for providers that don't support it
                response = await provider.generate_response(config, messages, **overrides)
                # Send all content at once via callback
                await chunk_callback(response, True)
                logger.debug(
                    f"Generated {len(response)} chars via fallback non-streaming from {model_id} ({config.provider})"
                )

            return response

        except Exception as e:
            logger.error(f"Streaming generation failed for {model_id} ({config.provider}): {e}")
            raise

    @asynccontextmanager
    async def model_session(self, model_id: str):
        """Create a session context for model operations."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        logger.debug(f"Starting session for model {model_id}")
        try:
            yield self
        finally:
            logger.debug(f"Ending session for model {model_id}")

    async def get_available_models(self) -> dict[str, list[str]]:
        """Get list of available models from all providers."""
        all_models = {}

        for provider_name in ProviderFactory.get_available_providers():
            try:
                provider = self._get_provider(provider_name)
                models = await provider.get_available_models()
                all_models[provider_name] = models
            except Exception as e:
                logger.error(f"Failed to get models from {provider_name}: {e}")
                all_models[provider_name] = []

        return all_models

    async def get_enhanced_models(self) -> list[dict[str, Any]]:
        """Get enhanced model information with metadata, filtering, and classification."""
        # Use the strongly typed version and convert to dict for API compatibility
        typed_models = await self.get_enhanced_models_typed()

        # Convert typed models to dict format for JSON serialization
        enhanced_models = []
        for model in typed_models:
            enhanced_models.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "description": (
                        model.description[:200] + "..."
                        if len(model.description) > 200
                        else model.description
                    ),
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
        enhanced_models = []

        for provider_name in ProviderFactory.get_available_providers():
            try:
                provider = self._get_provider(provider_name)

                # Check if provider supports enhanced model information
                if hasattr(provider, "get_enhanced_models"):
                    # Type-safe call using cast and protocol
                    enhanced_provider = cast(EnhancedModelProvider, provider)
                    provider_enhanced = await enhanced_provider.get_enhanced_models()
                    enhanced_models.extend(provider_enhanced)
                else:
                    # Fallback for providers without enhanced support
                    from .base_types import ModelWeightClass, ModelTier, ModelPricing

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
                            source_info={},
                        )
                        enhanced_models.append(fallback_model)

            except Exception as e:
                logger.error(f"Failed to get enhanced models from {provider_name}: {e}")

        # Filter out problematic models with overly aggressive safety filters
        BLACKLISTED_MODELS = {
            "meta-llama/llama-3.2-11b-vision-instruct",  # Returns safety codes instead of debate content
        }

        filtered_models = [
            model for model in enhanced_models
            if model.id not in BLACKLISTED_MODELS
        ]

        if len(filtered_models) != len(enhanced_models):
            blacklisted_count = len(enhanced_models) - len(filtered_models)
            logger.info(f"Filtered out {blacklisted_count} blacklisted model(s) with problematic safety filters")

        return filtered_models

    async def generate_response_with_metadata(
        self, model_id: str, messages: list[dict[str, str]], **overrides
    ) -> GenerationMetadata:
        """Generate a response with full metadata for cost tracking."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        metadata = await provider.generate_response_with_metadata(config, messages, **overrides)
        logger.debug(
            f"Generated {len(metadata.content)} chars with metadata from {model_id} ({config.provider}), "
            f"generation_id: {metadata.generation_id}"
        )
        return metadata

    async def generate_response_stream_with_metadata(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        chunk_callback: Callable[[str, bool], Awaitable[None]],
        **overrides
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
            f"Generated {len(metadata.content)} chars via streaming with metadata from {model_id} ({config.provider}), "
            f"generation_id: {metadata.generation_id}"
        )
        return metadata

    async def query_generation_cost(self, model_id: str, generation_id: str) -> float | None:
        """Query the cost for a specific generation using the appropriate provider."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)

        return await provider.query_generation_cost(generation_id)
