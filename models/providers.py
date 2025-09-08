"""Abstract base classes and implementations for model providers."""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging
from openai import OpenAI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from config.settings import SystemConfig, ModelConfig

logger = logging.getLogger(__name__)


class BaseModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, system_config: "SystemConfig"):
        self.system_config = system_config
        self._client: Optional[OpenAI] = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models from this provider."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        model_config: "ModelConfig",
        messages: List[Dict[str, str]],
        **overrides
    ) -> str:
        """Generate a response using this provider."""
        pass
    
    @abstractmethod
    def validate_model_config(self, model_config: "ModelConfig") -> bool:
        """Validate that a model configuration is compatible with this provider."""
        pass


class OllamaProvider(BaseModelProvider):
    """Ollama model provider implementation."""
    
    def __init__(self, system_config: "SystemConfig"):
        super().__init__(system_config)
        self._client = OpenAI(
            base_url=f"{system_config.ollama_base_url}/v1",
            api_key="ollama"  # Ollama doesn't require real API key
        )
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            models = self._client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return []
    
    async def generate_response(
        self,
        model_config: "ModelConfig",
        messages: List[Dict[str, str]],
        **overrides
    ) -> str:
        """Generate a response using Ollama."""
        if not self._client:
            raise RuntimeError("Ollama client not initialized")
        
        # Prepare generation parameters
        params = {
            "model": model_config.name,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", model_config.max_tokens),
            "temperature": overrides.get("temperature", model_config.temperature),
        }
        
        # Add Ollama-specific parameters to extra_body
        ollama_config = self.system_config.ollama
        extra_body = {}
        if ollama_config.keep_alive is not None:
            extra_body["keep_alive"] = ollama_config.keep_alive
        if ollama_config.repeat_penalty is not None:
            extra_body["repeat_penalty"] = ollama_config.repeat_penalty
        if ollama_config.num_gpu_layers is not None:
            extra_body["num_gpu"] = ollama_config.num_gpu_layers
        if ollama_config.num_thread is not None:
            extra_body["num_thread"] = ollama_config.num_thread
        if ollama_config.main_gpu is not None:
            extra_body["main_gpu"] = ollama_config.main_gpu
        
        if extra_body:
            params["extra_body"] = extra_body
        
        try:
            response: "ChatCompletion" = self._client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
            
            logger.debug(f"Generated {len(content)} chars from Ollama model {model_config.name}")
            return content.strip()
            
        except Exception as e:
            logger.error(f"Ollama generation failed for {model_config.name}: {e}")
            raise
    
    def validate_model_config(self, model_config: "ModelConfig") -> bool:
        """Validate Ollama model configuration."""
        return model_config.provider == "ollama"


class OpenRouterProvider(BaseModelProvider):
    """OpenRouter model provider implementation."""
    
    def __init__(self, system_config: "SystemConfig"):
        super().__init__(system_config)
        
        # Get API key from config or environment
        api_key = (
            system_config.openrouter.api_key or 
            os.getenv("OPENROUTER_API_KEY")
        )
        
        if not api_key:
            logger.warning("No OpenRouter API key found. Set OPENROUTER_API_KEY or configure in system settings.")
            self._client = None
        else:
            # Prepare headers for OpenRouter
            headers = {}
            if system_config.openrouter.site_url:
                headers["HTTP-Referer"] = system_config.openrouter.site_url
            if system_config.openrouter.app_name:
                headers["X-Title"] = system_config.openrouter.app_name
            
            self._client = OpenAI(
                base_url=system_config.openrouter.base_url,
                api_key=api_key,
                timeout=system_config.openrouter.timeout,
                max_retries=system_config.openrouter.max_retries,
                default_headers=headers
            )
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from OpenRouter."""
        if not self._client:
            logger.warning("OpenRouter client not initialized - no API key")
            return []
        
        try:
            models = self._client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get OpenRouter models: {e}")
            return []
    
    async def generate_response(
        self,
        model_config: "ModelConfig",
        messages: List[Dict[str, str]],
        **overrides
    ) -> str:
        """Generate a response using OpenRouter."""
        if not self._client:
            raise RuntimeError("OpenRouter client not initialized - check API key")
        
        # Prepare generation parameters
        params = {
            "model": model_config.name,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", model_config.max_tokens),
            "temperature": overrides.get("temperature", model_config.temperature),
        }
        
        try:
            response: "ChatCompletion" = self._client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
            
            logger.debug(f"Generated {len(content)} chars from OpenRouter model {model_config.name}")
            return content.strip()
            
        except Exception as e:
            logger.error(f"OpenRouter generation failed for {model_config.name}: {e}")
            raise
    
    def validate_model_config(self, model_config: "ModelConfig") -> bool:
        """Validate OpenRouter model configuration."""
        return model_config.provider == "openrouter"


class ProviderFactory:
    """Factory for creating model providers."""
    
    _providers = {
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, system_config: "SystemConfig") -> BaseModelProvider:
        """Create a provider instance by name."""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(cls._providers.keys())}")
        
        provider_class = cls._providers[provider_name]
        return provider_class(system_config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())