from typing import List, TYPE_CHECKING

from .base_model_provider import BaseModelProvider
from .open_router_provider import OpenRouterProvider

from .ollama_provider import OllamaProvider

if TYPE_CHECKING:
    from config.settings import SystemConfig


class ProviderFactory:
    """Factory for creating model providers."""

    _providers = {
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def create_provider(
        cls, provider_name: str, system_config: "SystemConfig"
    ) -> BaseModelProvider:
        """Create a provider instance by name."""
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. Available: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(system_config)

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())
