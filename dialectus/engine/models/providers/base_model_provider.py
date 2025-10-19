from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from config.settings import ModelConfig, SystemConfig
    from debate_engine.types import ChunkCallback
    from models.base_types import BaseEnhancedModelInfo


@dataclass
class GenerationMetadata:
    """Strongly-typed metadata returned from model generation for cost tracking."""

    content: str
    generation_id: str | None = None  # OpenRouter generation ID for cost queries
    cost: float | None = None  # Cost in USD if available immediately
    prompt_tokens: int | None = None  # Input tokens used
    completion_tokens: int | None = None  # Output tokens generated
    total_tokens: int | None = None  # Total tokens (prompt + completion)
    generation_time_ms: int | None = None  # Generation time in milliseconds
    model: str | None = None  # Actual model used (may differ from requested)
    provider: str | None = None  # Provider name


class BaseModelProvider(ABC):
    """Abstract base class for model providers."""

    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self._client: OpenAI | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""

    @abstractmethod
    async def get_available_models(self) -> list[str]:
        """Get list of available models from this provider."""

    @abstractmethod
    async def get_enhanced_models(self) -> list[BaseEnhancedModelInfo]:
        """Return rich metadata for models surfaced by this provider."""

    @abstractmethod
    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        **overrides: object,
    ) -> str:
        """Generate a response using this provider."""

    @abstractmethod
    def validate_model_config(self, model_config: ModelConfig) -> bool:
        """Validate that a model configuration is compatible with this provider."""

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses."""
        return False

    async def generate_response_stream(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        chunk_callback: ChunkCallback,
        **overrides: object,
    ) -> str:
        """Generate a streaming response using this provider."""
        complete_response = await self.generate_response(
            model_config, messages, **overrides
        )
        await chunk_callback(complete_response, True)
        return complete_response

    async def generate_response_with_metadata(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        **overrides: object,
    ) -> GenerationMetadata:
        """Generate a response with full metadata for cost tracking."""
        content = await self.generate_response(model_config, messages, **overrides)
        return GenerationMetadata(content=content)

    async def generate_response_stream_with_metadata(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        chunk_callback: ChunkCallback,
        **overrides: object,
    ) -> GenerationMetadata:
        """Generate a streaming response with full metadata for cost tracking."""
        content = await self.generate_response_stream(
            model_config, messages, chunk_callback, **overrides
        )
        return GenerationMetadata(content=content)

    async def query_generation_cost(self, generation_id: str) -> float | None:
        """Query the cost for a specific generation ID."""
        return None
