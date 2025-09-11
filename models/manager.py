"""Model manager with multi-provider support."""

from typing import Dict, List, Any, TYPE_CHECKING, Protocol, cast
import logging
from contextlib import asynccontextmanager
from config.settings import ModelConfig, SystemConfig
from models.providers import BaseModelProvider, ProviderFactory

if TYPE_CHECKING:
    from models.base_types import BaseEnhancedModelInfo

class EnhancedModelProvider(Protocol):
    """Protocol for providers that support enhanced model information."""
    async def get_enhanced_models(self) -> List["BaseEnhancedModelInfo"]:
        ...

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model inference via multiple providers."""
    
    def __init__(self, system_config: SystemConfig):
        self._system_config = system_config
        self._model_configs: Dict[str, ModelConfig] = {}
        self._providers: Dict[str, BaseModelProvider] = {}
        
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
        self, 
        model_id: str, 
        messages: List[Dict[str, str]],
        **overrides
    ) -> str:
        """Generate a response from the specified model."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model {model_id} not registered")
        
        config = self._model_configs[model_id]
        provider = self._get_provider(config.provider)
        
        try:
            response = await provider.generate_response(config, messages, **overrides)
            logger.debug(f"Generated {len(response)} chars from {model_id} ({config.provider})")
            return response
            
        except Exception as e:
            logger.error(f"Generation failed for {model_id} ({config.provider}): {e}")
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
    
    async def get_available_models(self) -> Dict[str, List[str]]:
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
    
    async def get_available_models_flat(self) -> List[str]:
        """Get flat list of available models (backward compatibility)."""
        all_models = await self.get_available_models()
        flat_list = []
        for models in all_models.values():
            flat_list.extend(models)
        return flat_list
    
    async def get_enhanced_models(self) -> List[Dict[str, Any]]:
        """Get enhanced model information with metadata, filtering, and classification."""
        enhanced_models = []
        
        for provider_name in ProviderFactory.get_available_providers():
            try:
                provider = self._get_provider(provider_name)
                
                # Check if provider supports enhanced model information
                if hasattr(provider, 'get_enhanced_models'):
                    # Type-safe call using cast and protocol
                    enhanced_provider = cast(EnhancedModelProvider, provider)
                    provider_enhanced = await enhanced_provider.get_enhanced_models()
                    
                    # Convert to dict format for JSON serialization
                    for model in provider_enhanced:
                        enhanced_models.append({
                            "id": model.id,
                            "name": model.name,
                            "provider": model.provider,
                            "description": model.description[:200] + "..." if len(model.description) > 200 else model.description,
                            "weight_class": model.weight_class.value,
                            "tier": model.tier.value,
                            "context_length": model.context_length,
                            "max_completion_tokens": model.max_completion_tokens,
                            "pricing": {
                                "prompt_cost_per_1k": model.pricing.prompt_cost_per_1k,
                                "completion_cost_per_1k": model.pricing.completion_cost_per_1k,
                                "avg_cost_per_1k": model.pricing.avg_cost_per_1k,
                                "is_free": model.pricing.is_free
                            },
                            "value_score": round(model.value_score, 2),
                            "is_preview": model.is_preview,
                            "is_text_only": model.is_text_only,
                            "estimated_params": model.estimated_params,
                            "display_name": model.display_name
                        })
                else:
                    # Fallback for providers without enhanced support
                    basic_models = await provider.get_available_models()
                    for model_id in basic_models:
                        enhanced_models.append({
                            "id": model_id,
                            "name": model_id,
                            "provider": provider_name,
                            "description": "Standard model",
                            "weight_class": "middleweight",
                            "tier": "balanced",
                            "context_length": 4096,
                            "max_completion_tokens": 1024,
                            "pricing": {
                                "prompt_cost_per_1k": 0.0,
                                "completion_cost_per_1k": 0.0,
                                "avg_cost_per_1k": 0.0,
                                "is_free": True
                            },
                            "value_score": 5.0,
                            "is_preview": False,
                            "is_text_only": True,
                            "estimated_params": None,
                            "display_name": model_id
                        })
                        
            except Exception as e:
                logger.error(f"Failed to get enhanced models from {provider_name}: {e}")
        
        return enhanced_models