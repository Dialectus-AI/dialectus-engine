from typing import Dict, List, Optional, TYPE_CHECKING
from openai import OpenAI
from .base_model_provider import BaseModelProvider
import logging

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from config.settings import SystemConfig, ModelConfig
    from models.base_types import ModelWeightClass, BaseEnhancedModelInfo

logger = logging.getLogger(__name__)


class OllamaProvider(BaseModelProvider):
    """Ollama model provider implementation."""

    def __init__(self, system_config: "SystemConfig"):
        super().__init__(system_config)
        self._client = OpenAI(
            base_url=f"{system_config.ollama_base_url}/v1",
            api_key="ollama",  # Ollama doesn't require real API key
            timeout=120.0,  # Increased timeout to allow for model loading and generation
        )
        self._ollama_base_url = system_config.ollama_base_url

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def is_running(self) -> bool:
        """Fast health check to see if Ollama server is running."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"{self._ollama_base_url}/api/tags")
                response.raise_for_status()
                return True
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        # Fast health check first
        if not await self.is_running():
            logger.error("Failed to get Ollama models: Connection error.")
            return []

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
        self, model_config: "ModelConfig", messages: list[dict[str, str]], **overrides
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

            logger.debug(
                f"Generated {len(content)} chars from Ollama model {model_config.name}"
            )
            return content.strip()

        except Exception as e:
            logger.error(f"Ollama generation failed for {model_config.name}: {e}")
            raise

    def validate_model_config(self, model_config: "ModelConfig") -> bool:
        """Validate Ollama model configuration."""
        return model_config.provider == "ollama"

    async def get_enhanced_models(self) -> list["BaseEnhancedModelInfo"]:
        """Get enhanced model information for Ollama models."""
        basic_models = await self.get_available_models()
        enhanced_models = []

        from models.base_types import (
            BaseEnhancedModelInfo,
            ModelWeightClass,
            ModelTier,
            ModelPricing,
        )

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
                currency="USD",
            )

            # High value score for local models (free + private)
            value_score = 10.0 + (
                1.0 if weight_class == ModelWeightClass.HEAVYWEIGHT else 0.5
            )

            enhanced_model = BaseEnhancedModelInfo(
                id=model_id,
                name=model_id,
                provider="ollama",
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
                source_info={"local_model": True},
            )

            enhanced_models.append(enhanced_model)

        return enhanced_models

    def _classify_ollama_model(self, model_id: str) -> "ModelWeightClass":
        """Classify Ollama model into weight class."""

        model_lower = model_id.lower()

        from models.base_types import ModelWeightClass

        # Large models
        if any(size in model_lower for size in ["70b", "72b", "405b"]):
            return ModelWeightClass.ULTRAWEIGHT
        elif any(size in model_lower for size in ["34b", "32b"]):
            return ModelWeightClass.HEAVYWEIGHT
        elif any(size in model_lower for size in ["13b", "14b", "15b", "8b", "9b"]):
            return ModelWeightClass.MIDDLEWEIGHT
        elif any(size in model_lower for size in ["7b", "3b", "1b"]):
            return ModelWeightClass.LIGHTWEIGHT

        # Fallback based on model family
        if any(family in model_lower for family in ["llama", "mistral", "qwen"]):
            return ModelWeightClass.MIDDLEWEIGHT

        return ModelWeightClass.LIGHTWEIGHT

    def _estimate_ollama_params(self, model_id: str) -> str | None:
        """Estimate parameter count from Ollama model name."""
        model_lower = model_id.lower()

        # Extract parameter size from common patterns
        import re

        # Look for patterns like "7b", "13b", "70b", etc.
        param_match = re.search(r"(\d+(?:\.\d+)?)b", model_lower)
        if param_match:
            return param_match.group(1) + "B"

        # Look for specific model patterns
        if "tiny" in model_lower:
            return "1B"
        elif "small" in model_lower:
            return "7B"
        elif "medium" in model_lower:
            return "13B"
        elif "large" in model_lower:
            return "70B"

        return None
