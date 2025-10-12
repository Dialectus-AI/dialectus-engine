from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, ClassVar, cast

import httpx
from httpx import HTTPStatusError
from openai import OpenAI

from dialectus.engine.models.base_types import BaseEnhancedModelInfo

from .base_model_provider import BaseModelProvider, GenerationMetadata
from .exceptions import ProviderRateLimitError
from .openrouter_generation_types import (
    OpenRouterChatCompletionResponse,
    OpenRouterGenerationApiResponse,
)

if TYPE_CHECKING:
    from dialectus.engine.config.settings import ModelConfig, SystemConfig
    from dialectus.engine.models.openrouter.openrouter_enhanced_model_info import (
        OpenRouterEnhancedModelInfo,
    )

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseModelProvider):
    """OpenRouter model provider implementation."""

    # Class-level rate limiting to prevent 429 errors
    _last_request_time: ClassVar[float | None] = None
    _request_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _min_request_interval: ClassVar[float] = (
        3.0  # Minimum 3 seconds between requests to prevent 429 errors
    )

    def __init__(self, system_config: SystemConfig):
        super().__init__(system_config)

        # Get API key from config or environment
        api_key = self._get_api_key()

        if not api_key:
            logger.warning(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY or configure in system settings."
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
                    "OpenRouter site_url not configured - this may trigger stricter rate limits. Set OPENROUTER_SITE_URL env var."
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

        Site URL is sent as HTTP-Referer header for OpenRouter attribution and rate limiting.
        Environment variable takes precedence over config.

        Returns:
            Site URL string if found, None otherwise
        """
        return (
            os.getenv("OPENROUTER_SITE_URL") or self.system_config.openrouter.site_url
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
                    logger.debug(
                        f"Rate limiting: waiting {sleep_time:.2f}s before next OpenRouter request"
                    )
                    await asyncio.sleep(sleep_time)

            # Update the class variable to track last request time
            OpenRouterProvider._last_request_time = time.time()

    async def get_available_models(self) -> list[str]:
        """Get curated list of available models from OpenRouter with intelligent filtering."""
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
                from dialectus.engine.models.openrouter.openrouter_enhanced_model_info import (
                    OpenRouterEnhancedModelInfo,
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

                # Cache the enhanced models for 1 hour (OpenRouter doesn't update frequently)
                cache_manager.set("openrouter", "models", enhanced_models, ttl_hours=1)

                logger.info(
                    f"OpenRouter: Fetched and cached {len(models_response.data)} models, filtered down to {len(enhanced_models)} curated options"
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

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        **overrides: object,
    ) -> str:
        """Generate a response using OpenRouter."""
        if not self._client:
            raise RuntimeError("OpenRouter client not initialized - check API key")

        max_tokens = model_config.max_tokens
        max_tokens_override = overrides.get("max_tokens")
        if isinstance(max_tokens_override, int):
            max_tokens = max_tokens_override

        temperature = model_config.temperature
        temperature_override = overrides.get("temperature")
        if isinstance(temperature_override, (int, float)):
            temperature = float(temperature_override)

        try:
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

            payload: dict[str, object] = {
                "model": model_config.name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning": {"exclude": True},
            }

            await self._rate_limit_request()

            async with httpx.AsyncClient() as client:
                http_response = await client.post(
                    f"{self.system_config.openrouter.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.system_config.openrouter.timeout,
                )
                http_response.raise_for_status()
                response_json = http_response.json()

            response_data = cast(OpenRouterChatCompletionResponse, response_json)
            content = ""
            choices_raw = response_data["choices"]
            if choices_raw:
                first_choice = choices_raw[0]
                message_data = first_choice.get("message")
                if isinstance(message_data, dict):
                    message_dict = cast(dict[str, object], message_data)
                    content_value = message_dict.get("content")
                    if isinstance(content_value, str):
                        content = content_value

            if not content.strip():
                logger.warning(
                    "OpenRouter model %s returned empty content. Response data: %s",
                    model_config.name,
                    response_data,
                )
            else:
                logger.debug(
                    "Generated %s chars from OpenRouter model %s",
                    len(content),
                    model_config.name,
                )

            return content.strip()

        except HTTPStatusError as http_err:
            status = http_err.response.status_code if http_err.response else None
            if status == 429:
                detail = "OpenRouter rate limited the request."
                if ":free" in model_config.name:
                    detail += " Free-tier routes (suffix ':free') require sufficient balance on OpenRouter."
                raise ProviderRateLimitError(
                    provider="openrouter",
                    model=model_config.name,
                    status_code=429,
                    detail=detail,
                ) from http_err
            logger.error(
                "OpenRouter generation HTTP failure for %s: status %s",
                model_config.name,
                status,
            )
            raise
        except Exception as exc:
            logger.error(
                "OpenRouter generation failed for %s: %s", model_config.name, exc
            )
            raise

    def validate_model_config(self, model_config: ModelConfig) -> bool:
        """Validate OpenRouter model configuration."""
        return model_config.provider == "openrouter"

    def supports_streaming(self) -> bool:
        """OpenRouter supports streaming responses."""
        return True

    async def generate_response_stream(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        chunk_callback: Callable[[str, bool], Awaitable[None]],
        **overrides: object,
    ) -> str:
        """Generate a streaming response using OpenRouter with SSE."""
        if not self._client:
            raise RuntimeError("OpenRouter client not initialized - check API key")

        max_tokens = model_config.max_tokens
        max_tokens_override = overrides.get("max_tokens")
        if isinstance(max_tokens_override, int):
            max_tokens = max_tokens_override

        temperature = model_config.temperature
        temperature_override = overrides.get("temperature")
        if isinstance(temperature_override, (int, float)):
            temperature = float(temperature_override)

        try:
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

            payload: dict[str, object] = {
                "model": model_config.name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "reasoning": {"exclude": True},
            }

            await self._rate_limit_request()

            complete_content = ""
            stream_finished = False

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.system_config.openrouter.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.system_config.openrouter.timeout,
                ) as response:
                    response.raise_for_status()

                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk

                        while True:
                            line_end = buffer.find("\n")
                            if line_end == -1:
                                break

                            line = buffer[:line_end].strip()
                            buffer = buffer[line_end + 1 :]

                            if not line or line.startswith(":"):
                                continue

                            if not line.startswith("data: "):
                                continue

                            data = line[6:]
                            if data == "[DONE]":
                                await chunk_callback("", True)
                                stream_finished = True
                                break

                            try:
                                parsed_obj = json.loads(data)
                            except json.JSONDecodeError:
                                logger.debug(
                                    "Skipping invalid JSON in stream: %s...",
                                    data[:100],
                                )
                                continue
                            except Exception as exc:
                                logger.error(
                                    "Error processing stream chunk: %s",
                                    exc,
                                )
                                continue

                            if not isinstance(parsed_obj, dict):
                                continue

                            parsed_dict = cast(dict[str, object], parsed_obj)

                            error_obj = parsed_dict.get("error")
                            if isinstance(error_obj, dict):
                                error_dict = cast(dict[str, object], error_obj)
                                message_value = error_dict.get("message")
                                error_msg = (
                                    message_value
                                    if isinstance(message_value, str)
                                    else "Unknown streaming error"
                                )
                                logger.error(
                                    "OpenRouter streaming error: %s", error_msg
                                )
                                raise RuntimeError(f"Streaming error: {error_msg}")

                            choices_value = parsed_dict.get("choices")
                            if not isinstance(choices_value, list) or not choices_value:
                                continue

                            choice_candidates = cast(list[object], choices_value)
                            typed_choices: list[dict[str, object]] = []
                            for choice_candidate in choice_candidates:
                                if isinstance(choice_candidate, dict):
                                    typed_choices.append(
                                        cast(dict[str, object], choice_candidate)
                                    )

                            if not typed_choices:
                                continue

                            first_choice = typed_choices[0]

                            delta_obj = first_choice.get("delta")
                            if isinstance(delta_obj, dict):
                                delta_dict = cast(dict[str, object], delta_obj)
                                content_value = delta_dict.get("content")
                                if isinstance(content_value, str) and content_value:
                                    complete_content += content_value
                                    await chunk_callback(content_value, False)

                            finish_value = first_choice.get("finish_reason")
                            if isinstance(finish_value, str) and finish_value:
                                await chunk_callback("", True)
                                stream_finished = True
                                break
                        if stream_finished:
                            break

            logger.debug(
                "OpenRouter streaming completed: %s chars from %s",
                len(complete_content),
                model_config.name,
            )
            return complete_content.strip()

        except HTTPStatusError as http_err:
            status = http_err.response.status_code if http_err.response else None
            if status == 429:
                detail = "OpenRouter rate limited the request."
                if ":free" in model_config.name:
                    detail += " Free-tier routes (suffix ':free') require sufficient balance on OpenRouter."
                raise ProviderRateLimitError(
                    provider="openrouter",
                    model=model_config.name,
                    status_code=429,
                    detail=detail,
                ) from http_err
            logger.error(
                "OpenRouter streaming HTTP failure for %s: status %s",
                model_config.name,
                status,
            )
            raise
        except Exception as exc:
            logger.error(
                "OpenRouter streaming failed for %s: %s", model_config.name, exc
            )
            raise

    async def generate_response_with_metadata(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        **overrides: object,
    ) -> GenerationMetadata:
        """Generate response with full metadata including generation ID for cost tracking."""
        if not self._client:
            raise RuntimeError("OpenRouter client not initialized - check API key")

        max_tokens = model_config.max_tokens
        max_tokens_override = overrides.get("max_tokens")
        if isinstance(max_tokens_override, int):
            max_tokens = max_tokens_override

        temperature = model_config.temperature
        temperature_override = overrides.get("temperature")
        if isinstance(temperature_override, (int, float)):
            temperature = float(temperature_override)

        try:
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

            payload: dict[str, object] = {
                "model": model_config.name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning": {"exclude": True},
            }

            await self._rate_limit_request()

            start_time = time.time()

            async with httpx.AsyncClient() as client:
                http_response = await client.post(
                    f"{self.system_config.openrouter.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.system_config.openrouter.timeout,
                )
                http_response.raise_for_status()
                response_json = http_response.json()

            response_data = cast(OpenRouterChatCompletionResponse, response_json)
            generation_time_ms = int((time.time() - start_time) * 1000)

            content = ""
            choices_raw = response_data["choices"]
            if choices_raw:
                first_choice = choices_raw[0]
                message_data = first_choice.get("message")
                if isinstance(message_data, dict):
                    message_dict = cast(dict[str, object], message_data)
                    content_value = message_dict.get("content")
                    if isinstance(content_value, str):
                        content = content_value

            generation_id = response_data["id"]
            if not generation_id:
                raise RuntimeError(
                    f"OpenRouter response missing generation ID: {response_data}"
                )

            usage_data = response_data["usage"]
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None
            if usage_data is not None:
                prompt_tokens = usage_data["prompt_tokens"]
                completion_tokens = usage_data["completion_tokens"]
                total_tokens = usage_data["total_tokens"]

            logger.debug(
                "Generated %s chars from OpenRouter %s, generation_id: %s, tokens: %s",
                len(content),
                model_config.name,
                generation_id,
                total_tokens,
            )

            return GenerationMetadata(
                content=content,
                generation_id=generation_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time_ms,
                model=model_config.name,
                provider="openrouter",
            )

        except HTTPStatusError as http_err:
            status = http_err.response.status_code if http_err.response else None
            if status == 429:
                detail = "OpenRouter rate limited the request."
                if ":free" in model_config.name:
                    detail += " Free-tier routes (suffix ':free') require sufficient balance on OpenRouter."
                raise ProviderRateLimitError(
                    provider="openrouter",
                    model=model_config.name,
                    status_code=429,
                    detail=detail,
                ) from http_err
            logger.error(
                "OpenRouter metadata generation HTTP failure for %s: status %s",
                model_config.name,
                status,
            )
            raise
        except Exception as exc:
            logger.error(
                "OpenRouter metadata generation failed for %s: %s",
                model_config.name,
                exc,
            )
            raise

    async def query_generation_cost(self, generation_id: str) -> float:
        """Query OpenRouter for the cost of a specific generation. Fails fast on errors."""
        if not self._client:
            raise RuntimeError(
                "OpenRouter client not initialized - this should not happen if we got a generation_id"
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
