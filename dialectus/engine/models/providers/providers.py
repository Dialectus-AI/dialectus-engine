from typing import TYPE_CHECKING

from .anthropic_provider import AnthropicProvider
from .base_model_provider import BaseModelProvider
from .ollama_provider import OllamaProvider
from .open_router_provider import OpenRouterProvider
from .openai_provider import OpenAIProvider

if TYPE_CHECKING:
    from config.settings import SystemConfig


class ProviderFactory:
    """Factory for creating model providers."""

    _providers = {
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def create_provider(
        cls, provider_name: str, system_config: "SystemConfig"
    ) -> BaseModelProvider:
        """Create a provider instance by name."""
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. Available:"
                f" {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(system_config)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())
