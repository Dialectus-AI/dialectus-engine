from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar, cast

import httpx
from httpx import HTTPStatusError
from openai import OpenAI

from dialectus.engine.models.base_types import BaseEnhancedModelInfo

from .openai_compatible_provider import OpenAICompatibleProviderBase
from .exceptions import ProviderRateLimitError
from .openrouter_generation_types import (
    OpenRouterGenerationApiResponse,
)

if TYPE_CHECKING:
    from dialectus.engine.config.settings import ModelConfig, SystemConfig
    from dialectus.engine.models.openrouter.openrouter_enhanced_model_info import (
        OpenRouterEnhancedModelInfo,
    )

logger = logging.getLogger(__name__)


class OpenRouterProvider(OpenAICompatibleProviderBase):
    """OpenRouter model provider implementation.

    Uses OpenRouter's OpenAI-compatible API for all operations.
    Provides dynamic model catalog fetching and post-generation cost queries.
    """

    # Class-level rate limiting to prevent 429 errors (3 seconds between requests)
    _min_request_interval: ClassVar[float] = 3.0

    def __init__(self, system_config: SystemConfig):
        super().__init__(system_config)

        # Get API key from config or environment
        api_key = self._get_api_key()

        if not api_key:
            logger.warning(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY or configure in"
                " system settings."
            )
            self._client = None
        else:
            # Prepare headers for OpenRouter
            headers: dict[str, str] = {}

            # Get site_url from environment or config (environment takes precedence)
            site_url = self._get_site_url()
            if site_url:
                headers["HTTP-Referer"] = site_url
                logger.info(f"OpenRouter HTTP-Referer header set to: {site_url}")
            else:
                logger.warning(
                    "OpenRouter site_url not configured - this may trigger stricter"
                    " rate limits. Set OPENROUTER_SITE_URL env var."
                )

            if system_config.openrouter.app_name:
                headers["X-Title"] = system_config.openrouter.app_name

            self._client = OpenAI(
                base_url=system_config.openrouter.base_url,
                api_key=api_key,
                timeout=system_config.openrouter.timeout,
                max_retries=system_config.openrouter.max_retries,
                default_headers=headers,
            )

    def _get_api_key(self) -> str | None:
        """Get OpenRouter API key from environment or config.

        Returns:
            API key string if found, None otherwise
        """
        return os.getenv("OPENROUTER_API_KEY") or self.system_config.openrouter.api_key

    def _get_site_url(self) -> str | None:
        """Get OpenRouter site URL from environment or config.

        Site URL is sent as HTTP-Referer header for attribution and rate
        limiting. Environment variable takes precedence over config.

        Returns:
            Site URL string if found, None otherwise
        """
        return (
            os.getenv("OPENROUTER_SITE_URL") or self.system_config.openrouter.site_url
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def get_available_models(self) -> list[str]:
        """Get curated list of available models from OpenRouter.

        Uses intelligent filtering to return high-quality models only.
        """
        enhanced_models = await self.get_enhanced_models()
        return [model.id for model in enhanced_models]

    async def get_enhanced_models(self) -> list[BaseEnhancedModelInfo]:
        """Get enhanced model information with filtering and classification."""
        if not self._client:
            logger.warning("OpenRouter client not initialized - no API key")
            return []

        try:
            from dialectus.engine.models.cache_manager import cache_manager
            from dialectus.engine.models.openrouter.openrouter_model_filter import (
                OpenRouterModelFilter,
            )
            from dialectus.engine.models.openrouter.openrouter_models_response import (
                OpenRouterModelsResponse,
            )

            # Check cache first (6 hour default TTL)
            cached_models = cache_manager.get("openrouter", "models")
            enhanced_models: list[OpenRouterEnhancedModelInfo] = []
            if isinstance(cached_models, list):
                from dialectus.engine.models.openrouter import (
                    openrouter_enhanced_model_info,
                )

                OpenRouterEnhancedModelInfo = (
                    openrouter_enhanced_model_info.OpenRouterEnhancedModelInfo
                )

                cached_list = cast(list[object], cached_models)
                for model_candidate in cached_list:
                    if not isinstance(model_candidate, dict):
                        continue
                    model_dict = cast(dict[str, object], model_candidate)
                    try:
                        enhanced_model = OpenRouterEnhancedModelInfo.model_validate(
                            model_dict
                        )
                        enhanced_models.append(enhanced_model)
                    except Exception as exc:
                        model_id = str(model_dict.get("id", "unknown"))
                        logger.warning(
                            "Failed to reconstruct cached OpenRouter model %s: %s",
                            model_id,
                            exc,
                        )
                        continue

                if enhanced_models:
                    logger.info(
                        "Using cached OpenRouter models (%s models)",
                        len(enhanced_models),
                    )
                    return cast(list[BaseEnhancedModelInfo], enhanced_models)

                logger.warning(
                    "All cached models failed to reconstruct, fetching fresh data..."
                )
                cache_manager.invalidate("openrouter", "models")
            elif cached_models is not None:
                logger.warning(
                    "OpenRouter model cache contains unexpected type: %s",
                    type(cached_models).__name__,
                )

            # Cache miss - fetch fresh data from OpenRouter API
            logger.info("Fetching fresh OpenRouter models from API...")

            api_key = self._get_api_key()

            headers: dict[str, str] = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            site_url = self._get_site_url()
            if site_url:
                headers["HTTP-Referer"] = site_url
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

                # Cache enhanced models (1 hour, OpenRouter updates slowly)
                cache_manager.set("openrouter", "models", enhanced_models, ttl_hours=1)

                logger.info(
                    f"OpenRouter: Fetched and cached {len(models_response.data)}"
                    f" models, filtered down to {len(enhanced_models)} curated options"
                )
                return cast(list[BaseEnhancedModelInfo], enhanced_models)

        except HTTPStatusError as http_err:
            status = http_err.response.status_code if http_err.response else None
            if status == 429:
                detail = "OpenRouter rate limited the model discovery request."
                raise ProviderRateLimitError(
                    provider="openrouter",
                    model=None,
                    status_code=429,
                    detail=detail,
                ) from http_err
            logger.error("OpenRouter models request failed with HTTP %s", status)
            raise
        except Exception as e:
            logger.error(f"Failed to get enhanced OpenRouter models: {e}")
            raise  # Fail fast - don't hide errors from the frontend

    def validate_model_config(self, model_config: ModelConfig) -> bool:
        """Validate OpenRouter model configuration."""
        return model_config.provider == "openrouter"

    async def query_generation_cost(self, generation_id: str) -> float:
        """Query OpenRouter for the cost of a specific generation."""
        if not self._client:
            raise RuntimeError(
                "OpenRouter client not initialized - this should not happen if we got a"
                " generation_id"
            )

        if not generation_id:
            raise ValueError("generation_id is required for cost queries")

        try:
            api_key = self._get_api_key()
            if not api_key:
                raise RuntimeError(
                    "OpenRouter API key missing - cannot query generation cost"
                )

            headers: dict[str, str] = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            site_url = self._get_site_url()
            if site_url:
                headers["HTTP-Referer"] = site_url
            if self.system_config.openrouter.app_name:
                headers["X-Title"] = self.system_config.openrouter.app_name

            # Apply rate limiting before API request
            await self._rate_limit_request()

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.system_config.openrouter.base_url}/generation",
                    params={"id": generation_id},
                    headers=headers,
                    timeout=10.0,  # Shorter timeout for cost queries
                )
                response.raise_for_status()

                cost_data: OpenRouterGenerationApiResponse = response.json()
                total_cost = cost_data["data"]["total_cost"]

                logger.debug(
                    f"Retrieved cost for generation {generation_id}: ${total_cost}"
                )
                return total_cost

        except HTTPStatusError as http_err:
            status = http_err.response.status_code if http_err.response else None
            if status == 429:
                detail = "OpenRouter rate limited the cost query request."
                raise ProviderRateLimitError(
                    provider="openrouter",
                    model=None,
                    status_code=429,
                    detail=detail,
                ) from http_err
            logger.error(
                "OpenRouter cost query HTTP failure for %s: status %s",
                generation_id,
                status,
            )
            raise
        except Exception as e:
            logger.error(f"Failed to query cost for generation {generation_id}: {e}")
            raise
