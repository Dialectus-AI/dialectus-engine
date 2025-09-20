from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Awaitable
from openai import OpenAI

if TYPE_CHECKING:
    from config.settings import SystemConfig, ModelConfig


class BaseModelProvider(ABC):
    """Abstract base class for model providers."""

    def __init__(self, system_config: "SystemConfig"):
        self.system_config = system_config
        self._client: OpenAI | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    async def get_available_models(self) -> list[str]:
        """Get list of available models from this provider."""
        pass

    @abstractmethod
    async def generate_response(
        self, model_config: "ModelConfig", messages: list[dict[str, str]], **overrides
    ) -> str:
        """Generate a response using this provider."""
        pass

    @abstractmethod
    def validate_model_config(self, model_config: "ModelConfig") -> bool:
        """Validate that a model configuration is compatible with this provider."""
        pass

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses."""
        return False

    async def generate_response_stream(
        self,
        model_config: "ModelConfig",
        messages: list[dict[str, str]],
        chunk_callback: Callable[[str, bool], Awaitable[None]],
        **overrides
    ) -> str:
        """
        Generate a streaming response using this provider.

        Args:
            model_config: Configuration for the model to use
            messages: List of messages for the conversation
            chunk_callback: Async callback function that receives (chunk_text, is_complete)
            **overrides: Additional parameters to override model config

        Returns:
            The complete response text

        Note:
            Default implementation falls back to non-streaming generate_response().
            Providers should override this method to implement true streaming.
        """
        # Default fallback: call non-streaming method and return all at once
        complete_response = await self.generate_response(model_config, messages, **overrides)
        await chunk_callback(complete_response, True)
        return complete_response
