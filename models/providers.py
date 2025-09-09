"""Abstract base classes and implementations for model providers."""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging
from openai import OpenAI
import httpx

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from config.settings import SystemConfig, ModelConfig
    from .base_types import ModelWeightClass, BaseEnhancedModelInfo
    from .openrouter_types import OpenRouterEnhancedModelInfo

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
        if not self._client:
            logger.error("Ollama client not initialized")
            return []
        
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
    
    async def get_enhanced_models(self) -> List["BaseEnhancedModelInfo"]:
        """Get enhanced model information for Ollama models."""
        basic_models = await self.get_available_models()
        enhanced_models = []
        
        from .base_types import BaseEnhancedModelInfo, ModelWeightClass, ModelTier, ModelPricing
        
        for model_id in basic_models:
            # Classify Ollama models based on name patterns
            weight_class = self._classify_ollama_model(model_id)
            tier = ModelTier.BUDGET  # Local models are always budget-friendly
            
            # Estimate parameters from model name
            estimated_params = self._estimate_ollama_params(model_id)
            
            # All Ollama models are free and text-only
            pricing = ModelPricing(
                prompt_cost_per_1k=0.0,
                completion_cost_per_1k=0.0,
                is_free=True,
                currency="USD"
            )
            
            # High value score for local models (free + private)
            value_score = 10.0 + (1.0 if weight_class == ModelWeightClass.HEAVYWEIGHT else 0.5)
            
            enhanced_model = BaseEnhancedModelInfo(
                id=model_id,
                name=model_id,
                provider='ollama',
                description=f"Local Ollama model: {model_id}",
                weight_class=weight_class,
                tier=tier,
                context_length=4096,  # Default, could be model-specific
                max_completion_tokens=2048,
                pricing=pricing,
                value_score=value_score,
                is_preview=False,
                is_text_only=True,
                estimated_params=estimated_params,
                source_info={'local_model': True}
            )
            
            enhanced_models.append(enhanced_model)
        
        return enhanced_models
    
    def _classify_ollama_model(self, model_id: str) -> "ModelWeightClass":
        """Classify Ollama model into weight class."""
        
        model_lower = model_id.lower()
        
        from .base_types import ModelWeightClass
        
        # Large models
        if any(size in model_lower for size in ['70b', '72b', '405b']):
            return ModelWeightClass.ULTRAWEIGHT
        elif any(size in model_lower for size in ['34b', '32b']):
            return ModelWeightClass.HEAVYWEIGHT
        elif any(size in model_lower for size in ['13b', '14b', '15b', '8b', '9b']):
            return ModelWeightClass.MIDDLEWEIGHT
        elif any(size in model_lower for size in ['7b', '3b', '1b']):
            return ModelWeightClass.LIGHTWEIGHT
        
        # Fallback based on model family
        if any(family in model_lower for family in ['llama', 'mistral', 'qwen']):
            return ModelWeightClass.MIDDLEWEIGHT
        
        return ModelWeightClass.LIGHTWEIGHT
    
    def _estimate_ollama_params(self, model_id: str) -> Optional[str]:
        """Estimate parameter count from Ollama model name."""
        model_lower = model_id.lower()
        
        # Extract parameter size from common patterns
        import re
        
        # Look for patterns like "7b", "13b", "70b", etc.
        param_match = re.search(r'(\d+(?:\.\d+)?)b', model_lower)
        if param_match:
            return param_match.group(1) + "B"
        
        # Look for specific model patterns
        if 'tiny' in model_lower:
            return "1B"
        elif 'small' in model_lower:
            return "7B"
        elif 'medium' in model_lower:
            return "13B"
        elif 'large' in model_lower:
            return "70B"
        
        return None


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
        """Get curated list of available models from OpenRouter with intelligent filtering."""
        enhanced_models = await self.get_enhanced_models()
        return [model.id for model in enhanced_models]
    
    async def get_enhanced_models(self) -> List["OpenRouterEnhancedModelInfo"]:
        """Get enhanced model information with filtering and classification."""
        if not self._client:
            logger.warning("OpenRouter client not initialized - no API key")
            return []
        
        try:
            from .openrouter_types import OpenRouterModelsResponse, OpenRouterModelFilter
            
            # Fetch raw models from OpenRouter API directly
            api_key = (
                self.system_config.openrouter.api_key or 
                os.getenv("OPENROUTER_API_KEY")
            )
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            if self.system_config.openrouter.site_url:
                headers["HTTP-Referer"] = self.system_config.openrouter.site_url
            if self.system_config.openrouter.app_name:
                headers["X-Title"] = self.system_config.openrouter.app_name
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.system_config.openrouter.base_url}/models",
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                models_response = OpenRouterModelsResponse(**response.json())
                
                # Apply intelligent filtering
                enhanced_models = OpenRouterModelFilter.filter_and_enhance_models(
                    models_response.data,
                    include_preview=True,  # Include for testing, but mark them clearly
                    max_cost_per_1k=0.02,  # $0.02 per 1K tokens max
                    min_context_length=4096,  # At least 4K context
                    max_models_per_tier=8  # Limit selection to avoid overwhelming UI
                )
                
                logger.info(f"OpenRouter: Filtered {len(models_response.data)} models down to {len(enhanced_models)} curated options")
                return enhanced_models
                
        except Exception as e:
            logger.error(f"Failed to get enhanced OpenRouter models: {e}")
            # Fallback to basic model list if enhanced filtering fails
            try:
                models = self._client.models.list()
                logger.warning("Using fallback basic model list from OpenRouter")
                return []
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
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