"""Model and provider management endpoints."""

import logging
from typing import Any

from fastapi import HTTPException, APIRouter

from config.settings import get_default_config
from models.manager import ModelManager
from models.providers import ProviderFactory, OllamaProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/models")
async def get_models():
    """Get available models from all providers with enhanced metadata."""
    try:
        config = get_default_config()
        model_manager = ModelManager(config.system)

        # Get enhanced models first (this includes all the data we need)
        enhanced_models = await model_manager.get_enhanced_models()

        # Build other formats from enhanced_models to avoid redundant API calls
        models_by_provider = {}
        flat_models = []
        formatted_models = []

        for model in enhanced_models:
            # Build models_by_provider
            provider = model["provider"]
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model["name"])

            # Build flat_models
            flat_models.append(model["name"])

            # Build formatted_models
            formatted_models.append(
                {
                    "id": model["id"],
                    "name": model["name"],
                    "provider": model["provider"],
                }
            )

        return {
            "models": flat_models,  # Backward compatibility
            "models_detailed": formatted_models,  # Basic format with provider info
            "models_enhanced": enhanced_models,  # Full enhanced metadata
            "models_by_provider": models_by_provider,  # Grouped by provider
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers")
async def get_providers():
    """Get available model providers and their status."""
    try:
        providers = []

        for provider_name in ProviderFactory.get_available_providers():
            provider_info: dict[str, Any] = {
                "name": provider_name,
                "status": "available",
            }

            # Add provider-specific status information
            if provider_name == "openrouter":
                config = get_default_config()
                import os

                api_key_configured = bool(
                    config.system.openrouter.api_key or os.getenv("OPENROUTER_API_KEY")
                )
                provider_info["api_key_configured"] = api_key_configured
                if not api_key_configured:
                    provider_info["status"] = "requires_api_key"
            elif provider_name == "ollama":
                # Use the provider's own health check method
                try:
                    config = get_default_config()
                    provider = ProviderFactory.create_provider(
                        provider_name, config.system
                    )
                    # Type-safe check for OllamaProvider
                    if provider.provider_name == "ollama":
                        if (
                            isinstance(provider, OllamaProvider)
                            and await provider.is_running()
                        ):
                            provider_info["status"] = "available"
                            provider_info["ollama_running"] = True
                        else:
                            provider_info["status"] = "offline"
                            provider_info["ollama_running"] = False
                except Exception as e:
                    logger.debug(f"Ollama provider check failed: {e}")
                    provider_info["status"] = "offline"
                    provider_info["ollama_running"] = False

            providers.append(provider_info)

        return {"providers": providers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ollama/health")
async def ollama_health():
    """Quick health check for Ollama service."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                version_data = response.json()
                return {
                    "status": "available",
                    "running": True,
                    "version": version_data.get("version", "unknown"),
                }
            else:
                return {
                    "status": "offline",
                    "running": False,
                    "error": f"HTTP {response.status_code}",
                }
    except Exception as e:
        return {"status": "offline", "running": False, "error": str(e)}