"""Model manager with Ollama integration."""

from typing import Dict, List, Optional, Any, TYPE_CHECKING, AsyncContextManager
import logging
from contextlib import asynccontextmanager
from openai import OpenAI
from config.settings import ModelConfig, SystemConfig

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model inference via Ollama."""
    
    def __init__(self, system_config: SystemConfig):
        self._system_config = system_config
        self._client = OpenAI(
            base_url=f"{system_config.ollama_base_url}/v1",
            api_key="ollama"  # Ollama doesn't require real API key
        )
        self._model_configs: Dict[str, ModelConfig] = {}
        
    def register_model(self, model_id: str, config: ModelConfig) -> None:
        """Register a model configuration."""
        self._model_configs[model_id] = config
        logger.info(f"Registered model {model_id}: {config.name}")
    
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
        
        # Prepare generation parameters
        params = {
            "model": config.name,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", config.max_tokens),
            "temperature": overrides.get("temperature", config.temperature),
        }
        
        # Add Ollama-specific parameters to extra_body
        ollama_config = self._system_config.ollama
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
            
            logger.debug(f"Generated {len(content)} chars from {model_id}")
            return content.strip()
            
        except Exception as e:
            logger.error(f"Generation failed for {model_id}: {e}")
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
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            models = self._client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []