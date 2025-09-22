import os
import asyncio
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING, Optional, ClassVar, Callable, Awaitable
import logging
from openai import OpenAI
import httpx

from .base_model_provider import BaseModelProvider, GenerationMetadata
from .openrouter_generation_types import OpenRouterChatCompletionResponse, OpenRouterGenerationApiResponse

if TYPE_CHECKING:
    from config.settings import SystemConfig, ModelConfig
    from models.openrouter_types import OpenRouterEnhancedModelInfo

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseModelProvider):
    """OpenRouter model provider implementation."""

    # Class-level rate limiting to prevent 429 errors
    _last_request_time: ClassVar[float | None] = None
    _request_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _min_request_interval: ClassVar[float] = (
        3.0  # Minimum 3 seconds between requests to prevent 429 errors
    )

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

    async def get_enhanced_models(self) -> list["OpenRouterEnhancedModelInfo"]:
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
                        enhanced_model = OpenRouterEnhancedModelInfo.model_validate(
                            model_dict
                        )
                        enhanced_models.append(enhanced_model)
                    except Exception as e:
                        logger.warning(
                            f"Failed to reconstruct cached OpenRouter model {model_dict.get('id', 'unknown')}: {e}"
                        )
                        # Continue with other models rather than failing entirely
                        continue

                if enhanced_models:
                    logger.info(
                        f"Using cached OpenRouter models ({len(enhanced_models)} models)"
                    )
                    return enhanced_models
                else:
                    logger.warning(
                        "All cached models failed to reconstruct, fetching fresh data..."
                    )
                    # Clear corrupted cache and fall through to fresh API call
                    cache_manager.invalidate("openrouter", "models")

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

                # Cache the enhanced models for 1 hour (OpenRouter doesn't update frequently)
                cache_manager.set("openrouter", "models", enhanced_models, ttl_hours=1)

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
        self, model_config: "ModelConfig", messages: list[dict[str, str]], **overrides
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

    def supports_streaming(self) -> bool:
        """OpenRouter supports streaming responses."""
        return True

    async def generate_response_stream(
        self,
        model_config: "ModelConfig",
        messages: list[dict[str, str]],
        chunk_callback: Callable[[str, bool], Awaitable[None]],
        **overrides
    ) -> str:
        """Generate a streaming response using OpenRouter with SSE."""
        if not self._client:
            raise RuntimeError("OpenRouter client not initialized - check API key")

        # Prepare generation parameters
        params = {
            "model": model_config.name,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", model_config.max_tokens),
            "temperature": overrides.get("temperature", model_config.temperature),
            "stream": True,  # Enable streaming
        }

        try:
            api_key = self.system_config.openrouter.api_key or os.getenv("OPENROUTER_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            if self.system_config.openrouter.site_url:
                headers["HTTP-Referer"] = self.system_config.openrouter.site_url
            if self.system_config.openrouter.app_name:
                headers["X-Title"] = self.system_config.openrouter.app_name

            # Build the payload with reasoning parameter and streaming
            payload = {
                "model": params["model"],
                "messages": params["messages"],
                "max_tokens": params["max_tokens"],
                "temperature": params["temperature"],
                "stream": True,
                "reasoning": {"exclude": True},
            }

            # Apply rate limiting before API request
            await self._rate_limit_request()

            complete_content = ""

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.system_config.openrouter.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.system_config.openrouter.timeout,
                ) as response:
                    response.raise_for_status()

                    # Process SSE stream
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk

                        # Process complete lines from buffer
                        while True:
                            line_end = buffer.find('\n')
                            if line_end == -1:
                                break

                            line = buffer[:line_end].strip()
                            buffer = buffer[line_end + 1:]

                            # Skip empty lines and comments
                            if not line or line.startswith(':'):
                                continue

                            # Process SSE data lines
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix

                                # Check for end of stream
                                if data == '[DONE]':
                                    break

                                try:
                                    # Parse the JSON chunk
                                    parsed = json.loads(data)

                                    # Check for errors in the stream
                                    if 'error' in parsed:
                                        error_msg = parsed['error'].get('message', 'Unknown streaming error')
                                        logger.error(f"OpenRouter streaming error: {error_msg}")
                                        raise RuntimeError(f"Streaming error: {error_msg}")

                                    # Extract content from the delta
                                    choices = parsed.get('choices', [])
                                    if choices and 'delta' in choices[0]:
                                        delta = choices[0]['delta']
                                        content_chunk = delta.get('content', '')

                                        if content_chunk:
                                            complete_content += content_chunk

                                            # Call the chunk callback with the new content
                                            await chunk_callback(content_chunk, False)

                                        # Check if this is the final chunk
                                        finish_reason = choices[0].get('finish_reason')
                                        if finish_reason:
                                            # Final callback to indicate completion
                                            await chunk_callback("", True)
                                            break

                                except json.JSONDecodeError as e:
                                    # Skip invalid JSON - this can happen with SSE comments
                                    logger.debug(f"Skipping invalid JSON in stream: {data[:100]}...")
                                    continue
                                except Exception as e:
                                    logger.error(f"Error processing stream chunk: {e}")
                                    # Continue processing other chunks rather than failing entirely
                                    continue

            logger.debug(f"OpenRouter streaming completed: {len(complete_content)} chars from {model_config.name}")
            return complete_content.strip()

        except Exception as e:
            logger.error(f"OpenRouter streaming failed for {model_config.name}: {e}")
            raise

    async def generate_response_with_metadata(
        self, model_config: "ModelConfig", messages: list[dict[str, str]], **overrides
    ) -> GenerationMetadata:
        """Generate response with full metadata including generation ID for cost tracking."""
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
            api_key = self.system_config.openrouter.api_key or os.getenv("OPENROUTER_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            if self.system_config.openrouter.site_url:
                headers["HTTP-Referer"] = self.system_config.openrouter.site_url
            if self.system_config.openrouter.app_name:
                headers["X-Title"] = self.system_config.openrouter.app_name

            # Build the payload with reasoning parameter
            payload = {
                "model": params["model"],
                "messages": params["messages"],
                "max_tokens": params["max_tokens"],
                "temperature": params["temperature"],
                "reasoning": {"exclude": True},
            }

            # Apply rate limiting before API request
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
                response_data: OpenRouterChatCompletionResponse = http_response.json()

            generation_time_ms = int((time.time() - start_time) * 1000)
            content = response_data["choices"][0]["message"]["content"] or ""
            generation_id = response_data.get("id")  # This is the key field we need

            # Fail fast if we don't get a generation ID from OpenRouter
            if not generation_id:
                raise RuntimeError(f"OpenRouter response missing generation ID: {response_data}")

            # Extract usage information if available
            usage = response_data.get("usage")
            prompt_tokens = usage.get("prompt_tokens") if usage else None
            completion_tokens = usage.get("completion_tokens") if usage else None
            total_tokens = usage.get("total_tokens") if usage else None

            logger.debug(
                f"Generated {len(content)} chars from OpenRouter {model_config.name}, "
                f"generation_id: {generation_id}, tokens: {total_tokens}"
            )

            return GenerationMetadata(
                content=content,
                generation_id=generation_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time_ms,
                model=model_config.name,
                provider="openrouter"
            )

        except Exception as e:
            logger.error(f"OpenRouter metadata generation failed for {model_config.name}: {e}")
            raise

    async def query_generation_cost(self, generation_id: str) -> float:
        """Query OpenRouter for the cost of a specific generation. Fails fast on errors."""
        if not self._client:
            raise RuntimeError("OpenRouter client not initialized - this should not happen if we got a generation_id")

        if not generation_id:
            raise ValueError("generation_id is required for cost queries")

        try:
            api_key = self.system_config.openrouter.api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OpenRouter API key missing - cannot query generation cost")

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
                    f"{self.system_config.openrouter.base_url}/generation",
                    params={"id": generation_id},
                    headers=headers,
                    timeout=10.0,  # Shorter timeout for cost queries
                )
                response.raise_for_status()

                cost_data: OpenRouterGenerationApiResponse = response.json()

                if "data" not in cost_data:
                    raise RuntimeError(f"Invalid cost response format: {cost_data}")

                total_cost = cost_data["data"]["total_cost"]

                logger.debug(f"Retrieved cost for generation {generation_id}: ${total_cost}")
                return total_cost

        except Exception as e:
            logger.error(f"Failed to query cost for generation {generation_id}: {e}")
            raise
