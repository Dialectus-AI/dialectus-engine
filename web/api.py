"""FastAPI web application for Dialectus AI Debate System."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.debate_manager import DebateManager

# Stable endpoint routers (shared across versions)
from web.endpoints.models import router as models_router
from web.endpoints.system import router as system_router
from web.endpoints.transcripts import router as transcripts_router

# Version-specific endpoint routers
from web.endpoints.v1.auth import router as auth_v1_router
from web.endpoints.v1.debates import router as debates_v1_router, ws_router as debates_ws_v1_router
from web.endpoints.v1.tournaments import router as tournaments_v1_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Outputs to console
)

logger: logging.Logger = logging.getLogger(__name__)

logger.info("api.py module loaded")

# Cache cleanup scheduler
cleanup_task: asyncio.Task[None] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown."""
    global cleanup_task

    logger.info("Starting cache cleanup scheduler...")
    cleanup_task = asyncio.create_task(cache_cleanup_scheduler())

    from debate_engine.database.database import DatabaseManager
    from config.settings import get_default_config
    from web.email_service import initialize_email_service

    # This will create the database with all tables including auth tables
    DatabaseManager()

    # Initialize email service from config
    config = get_default_config()
    initialize_email_service(config)

    yield

    # Shutdown: Cancel cache cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Cache cleanup scheduler stopped")


async def cache_cleanup_scheduler() -> None:
    """Background task to periodically clean up expired cache entries."""
    while True:
        try:
            # Run cleanup every hour
            await asyncio.sleep(3600)

            from models.cache_manager import cache_manager

            cleaned_count: int = cache_manager.cleanup_expired()

            if cleaned_count > 0:
                logger.info(
                    f"Scheduled cache cleanup: removed {cleaned_count} expired entries"
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cache cleanup scheduler: {e}")


def get_allowed_origins() -> list[str] | None:
    """Get CORS origins from environment or use development defaults."""
    env_origins: str | None = os.environ.get("ALLOWED_ORIGINS")
    if env_origins:
        origins = [origin.strip() for origin in env_origins.split(",")]
        return origins
    return None


# FastAPI app
app: FastAPI = FastAPI(
    title="Dialectus AI Debate System",
    description="Web interface for local AI model debates",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware setup
allowed_origins: list[str] | None = get_allowed_origins()

if allowed_origins:

    logging.info(f"Setting CORS allowed origins: {allowed_origins}")

    # Production: Use specific origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:

    logging.info("No ALLOWED_ORIGINS set, using development CORS settings")

    # Development: Allow any localhost/127.0.0.1
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Always allow prod origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dialectus.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global debate manager
debate_manager: DebateManager = DebateManager()

# Include v1 API endpoints
# Stable endpoints (shared across versions)
app.include_router(models_router, prefix="/v1")
app.include_router(system_router, prefix="/v1")
app.include_router(transcripts_router, prefix="/v1")

# Version-specific endpoints
app.include_router(auth_v1_router, prefix="/v1")
app.include_router(debates_v1_router, prefix="/v1")
app.include_router(debates_ws_v1_router, prefix="/v1")
app.include_router(tournaments_v1_router, prefix="/v1")