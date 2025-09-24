"""System health, formats, and cache management endpoints."""

import logging

from fastapi import HTTPException, APIRouter

from formats import format_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"isAlive": True}


@router.get("/formats")
async def get_formats():
    """Get available debate formats."""
    return {"formats": format_registry.get_format_descriptions()}


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        from models.cache_manager import cache_manager

        stats = cache_manager.get_cache_stats()
        return {"cache_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/cleanup")
async def cleanup_cache():
    """Clean up expired cache entries."""
    try:
        from models.cache_manager import cache_manager

        cleaned_count = cache_manager.cleanup_expired()
        return {"message": f"Cleaned up {cleaned_count} expired cache entries"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/models")
async def invalidate_models_cache():
    """Force invalidate the models cache."""
    try:
        from models.cache_manager import cache_manager

        invalidated = cache_manager.invalidate("openrouter", "models")
        if invalidated:
            return {"message": "Models cache invalidated successfully"}
        else:
            return {"message": "No models cache found to invalidate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))