from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Unpack, cast

import httpx
from openai import OpenAI

from .openai_compatible_provider import OpenAICompatibleProviderBase
from .base_model_provider import ChatMessage, ModelOverrides

if TYPE_CHECKING:
    from config.settings import ModelConfig, SystemConfig
    from debate_engine.types import ChunkCallback
    from models.base_types import BaseEnhancedModelInfo, ModelWeightClass
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class OllamaProvider(OpenAICompatibleProviderBase):
    """Ollama model provider implementation.

    Ollama provides local LLM hosting with OpenAI-compatible API.
    This provider handles Ollama-specific configuration (keep_alive,
    repeat_penalty, GPU settings) and uses the native Ollama streaming API
    for better performance.
    """

    # No rate limiting for local Ollama server
    _min_request_interval = 0.0

    def __init__(self, system_config: SystemConfig):
        super().__init__(system_config)
        self._client = OpenAI(
            base_url=f"{system_config.ollama_base_url}/v1",
            api_key="ollama",  # Ollama doesn't require real API key
            timeout=120.0,  # Increased timeout for model loading
        )
        self._ollama_base_url = system_config.ollama_base_url

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def is_running(self) -> bool:
        """Fast health check to see if Ollama server is running."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"{self._ollama_base_url}/api/tags")
                response.raise_for_status()
                return True
        except httpx.TimeoutException as e:
            logger.error(f"Ollama health check timeout after 1s: {e}")
            return False
        except httpx.ConnectError as e:
            logger.error(f"Ollama health check connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {type(e).__name__}: {e}")
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

    def _get_ollama_extra_body(self) -> dict[str, object]:
        """Build Ollama-specific extra_body parameters from config.

        Returns:
            Dictionary of Ollama-specific parameters for extra_body
        """
        ollama_config = self.system_config.ollama
        extra_body: dict[str, object] = {}
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
        return extra_body

    async def generate_response(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        **overrides: Unpack[ModelOverrides],
    ) -> str:
        """Generate a response using Ollama with provider-specific options.

        Overrides base implementation to add Ollama-specific extra_body params.
        """
        if not self._client:
            raise RuntimeError("Ollama client not initialized")

        max_tokens, temperature = self._apply_overrides(model_config, overrides)
        extra_body = self._get_ollama_extra_body()

        try:
            await self._rate_limit_request()

            chat_messages = cast(list["ChatCompletionMessageParam"], messages)

            response: ChatCompletion
            if extra_body:
                response = self._client.chat.completions.create(
                    model=model_config.name,
                    messages=chat_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                )
            else:
                response = self._client.chat.completions.create(
                    model=model_config.name,
                    messages=chat_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            first_choice = response.choices[0]
            message = first_choice.message
            content_text = self._coerce_content_to_text(
                message.content if message.content is not None else ""
            )

            if not content_text.strip():
                logger.warning(
                    "Ollama model %s returned empty content", model_config.name
                )
            else:
                logger.debug(
                    "Generated %s chars from Ollama model %s",
                    len(content_text),
                    model_config.name,
                )

            return content_text.strip()

        except Exception as exc:
            self._handle_rate_limit_error(exc, model_config.name, "request")
            logger.error(
                "Ollama generation failed for %s: %s",
                model_config.name,
                exc,
            )
            raise

    def validate_model_config(self, model_config: ModelConfig) -> bool:
        """Validate Ollama model configuration."""
        return model_config.provider == "ollama"

    async def generate_response_stream(
        self,
        model_config: ModelConfig,
        messages: list[ChatMessage],
        chunk_callback: ChunkCallback,
        **overrides: Unpack[ModelOverrides],
    ) -> str:
        """Generate a streaming response using Ollama native API.

        Note: Uses Ollama's native /api/chat endpoint instead of OpenAI SDK
        for better streaming performance with Ollama.
        """
        if not self._client:
            raise RuntimeError("Ollama client not initialized")

        max_tokens, temperature = self._apply_overrides(model_config, overrides)
        extra_options = self._get_ollama_extra_body()

        try:
            await self._rate_limit_request()

            complete_content = ""

            async with httpx.AsyncClient() as client:
                options: dict[str, object] = {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
                if extra_options:
                    options.update(extra_options)

                payload: dict[str, object] = {
                    "model": model_config.name,
                    "messages": messages,
                    "stream": True,
                    "options": options,
                }

                async with client.stream(
                    "POST",
                    f"{self._ollama_base_url}/api/chat",
                    json=payload,
                    timeout=120.0,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug(
                                "Skipping invalid JSON in Ollama stream: %s...",
                                line[:100],
                            )
                            continue
                        except Exception as exc:
                            logger.error(
                                "Error processing Ollama stream chunk: %s",
                                exc,
                            )
                            continue

                        if not isinstance(parsed, dict):
                            continue

                        parsed_dict = cast(dict[str, object], parsed)

                        error_value = parsed_dict.get("error")
                        if error_value is not None:
                            error_text = str(error_value)
                            logger.error("Ollama streaming error: %s", error_text)
                            raise RuntimeError(f"Streaming error: {error_text}")

                        message_obj = parsed_dict.get("message")
                        if isinstance(message_obj, dict):
                            message_dict = cast(dict[str, object], message_obj)
                            content_value = message_dict.get("content")
                            if isinstance(content_value, str) and content_value:
                                complete_content += content_value
                                await chunk_callback(content_value, False)

                        done_value = parsed_dict.get("done")
                        if isinstance(done_value, bool) and done_value:
                            await chunk_callback("", True)
                            break

            logger.debug(
                "Ollama streaming completed: %s chars from %s",
                len(complete_content),
                model_config.name,
            )
            return complete_content.strip()
        except Exception as exc:
            logger.error(
                "Ollama streaming failed for %s: %s",
                model_config.name,
                exc,
            )
            raise

    async def get_enhanced_models(self) -> list[BaseEnhancedModelInfo]:
        """Get enhanced model information for Ollama models."""
        basic_models = await self.get_available_models()
        enhanced_models: list[BaseEnhancedModelInfo] = []

        from dialectus.engine.models.base_types import (
            BaseEnhancedModelInfo,
            ModelPricing,
            ModelTier,
            ModelWeightClass,
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

    def _classify_ollama_model(self, model_id: str) -> ModelWeightClass:
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
