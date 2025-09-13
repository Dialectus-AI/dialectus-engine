import os
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING, Optional, ClassVar
import logging
from openai import OpenAI
import httpx

from .base_model_provider import BaseModelProvider

if TYPE_CHECKING:
    from config.settings import SystemConfig, ModelConfig
    from models.openrouter_types import OpenRouterEnhancedModelInfo

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseModelProvider):
    """OpenRouter model provider implementation."""
    
    # Class-level rate limiting to prevent 429 errors
    _last_request_time: ClassVar[Optional[float]] = None
    _request_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _min_request_interval: ClassVar[float] = 3.0  # Minimum 3 seconds between requests to prevent 429 errors

    def __init__(self, system_config: "SystemConfig"):
        super().__init__(system_config)

        # Get API key from config or environment
        api_key = system_config.openrouter.api_key or os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            logger.warning(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY or configure in system settings."
            )
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
                default_headers=headers,
            )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def _rate_limit_request(self) -> None:
        """Ensure minimum time between requests to avoid 429 errors."""
        async with self._request_lock:
            current_time = time.time()
            
            if self._last_request_time is not None:
                time_since_last = current_time - self._last_request_time
                if time_since_last < self._min_request_interval:
                    sleep_time = self._min_request_interval - time_since_last
                    logger.debug(f"Rate limiting: waiting {sleep_time:.2f}s before next OpenRouter request")
                    await asyncio.sleep(sleep_time)
            
            # Update the class variable to track last request time
            OpenRouterProvider._last_request_time = time.time()

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
            from models.openrouter_types import (
                OpenRouterModelsResponse,
                OpenRouterModelFilter,
            )
            from models.cache_manager import cache_manager

            # Check cache first (6 hour default TTL)
            cached_models = cache_manager.get("openrouter", "models")
            if cached_models is not None:
                # Convert cached dictionaries back to model objects
                from models.openrouter_types import OpenRouterEnhancedModelInfo

                enhanced_models = []
                for model_dict in cached_models:
                    try:
                        # Use model_validate for proper type conversion from cache
                        enhanced_model = OpenRouterEnhancedModelInfo.model_validate(model_dict)
                        enhanced_models.append(enhanced_model)
                    except Exception as e:
                        logger.warning(f"Failed to reconstruct cached OpenRouter model {model_dict.get('id', 'unknown')}: {e}")
                        # Continue with other models rather than failing entirely
                        continue
                
                if enhanced_models:
                    logger.info(
                        f"Using cached OpenRouter models ({len(enhanced_models)} models)"
                    )
                    return enhanced_models
                else:
                    logger.warning("All cached models failed to reconstruct, fetching fresh data...")
                    # Clear corrupted cache and fall through to fresh API call
                    cache_manager.delete("openrouter", "models")

            # Cache miss - fetch fresh data from OpenRouter API
            logger.info("Fetching fresh OpenRouter models from API...")

            api_key = self.system_config.openrouter.api_key or os.getenv(
                "OPENROUTER_API_KEY"
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            if self.system_config.openrouter.site_url:
                headers["HTTP-Referer"] = self.system_config.openrouter.site_url
            if self.system_config.openrouter.app_name:
                headers["X-Title"] = self.system_config.openrouter.app_name

            # Apply rate limiting before API request
            await self._rate_limit_request()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.system_config.openrouter.base_url}/models",
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()

                models_response = OpenRouterModelsResponse(**response.json())

                # Apply intelligent filtering
                enhanced_models = OpenRouterModelFilter.filter_and_enhance_models(
                    models_response.data,
                    include_preview=True,  # Include for testing, but mark them clearly
                    max_cost_per_1k=0.02,  # $0.02 per 1K tokens max
                    min_context_length=4096,  # At least 4K context
                    max_models_per_tier=8,  # Limit selection to avoid overwhelming UI
                )

                # Cache the enhanced models for 6 hours
                cache_manager.set("openrouter", "models", enhanced_models, ttl_hours=6)

                logger.info(
                    f"OpenRouter: Fetched and cached {len(models_response.data)} models, filtered down to {len(enhanced_models)} curated options"
                )
                return enhanced_models

        except Exception as e:
            logger.error(f"Failed to get enhanced OpenRouter models: {e}")
            # Fallback to basic model list if enhanced filtering fails
            try:
                _ = self._client.models.list()
                logger.warning("Using fallback basic model list from OpenRouter")
                return []
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return []

    async def generate_response(
        self, model_config: "ModelConfig", messages: List[Dict[str, str]], **overrides
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
            # For OpenRouter, we need to make direct HTTP requests to support reasoning parameter
            import httpx

            api_key = self.system_config.openrouter.api_key or os.getenv(
                "OPENROUTER_API_KEY"
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            if self.system_config.openrouter.site_url:
                headers["HTTP-Referer"] = self.system_config.openrouter.site_url
            if self.system_config.openrouter.app_name:
                headers["X-Title"] = self.system_config.openrouter.app_name

            # Build the payload with reasoning parameter at top level
            payload = {
                "model": params["model"],
                "messages": params["messages"],
                "max_tokens": params["max_tokens"],
                "temperature": params["temperature"],
                "reasoning": {"exclude": True},
            }

            # Apply rate limiting before API request
            await self._rate_limit_request()
            
            async with httpx.AsyncClient() as client:
                http_response = await client.post(
                    f"{self.system_config.openrouter.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.system_config.openrouter.timeout,
                )
                http_response.raise_for_status()
                response_data = http_response.json()

                content = response_data["choices"][0]["message"]["content"] or ""

            # Enhanced logging for debugging empty responses
            if not content.strip():
                logger.warning(
                    f"OpenRouter model {model_config.name} returned empty content. "
                    f"Raw response: {repr(content)}, Response data: {response_data}"
                )
            else:
                logger.debug(
                    f"Generated {len(content)} chars from OpenRouter model {model_config.name}"
                )

            return content.strip()

        except Exception as e:
            logger.error(f"OpenRouter generation failed for {model_config.name}: {e}")
            raise

    def validate_model_config(self, model_config: "ModelConfig") -> bool:
        """Validate OpenRouter model configuration."""
        return model_config.provider == "openrouter"
