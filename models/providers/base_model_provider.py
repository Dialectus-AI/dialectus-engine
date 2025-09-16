from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
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
