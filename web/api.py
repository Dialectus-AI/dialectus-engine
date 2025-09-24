"""FastAPI web application for Dialectus AI Debate System."""

import asyncio
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.debate_manager import DebateManager
from web.auth_database import AuthDatabaseManager

# Import endpoint routers
from web.endpoints.models import router as models_router
from web.endpoints.system import router as system_router
from web.endpoints.debates import router as debates_router, ws_router as debates_ws_router
from web.endpoints.tournaments import router as tournaments_router
from web.endpoints.transcripts import router as transcripts_router
from web.endpoints.auth import router as auth_router

logger = logging.getLogger(__name__)

# Cache cleanup scheduler
cleanup_task = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global cleanup_task

    # Startup: Start cache cleanup scheduler
    logger.info("Starting cache cleanup scheduler...")
    cleanup_task = asyncio.create_task(cache_cleanup_scheduler())

    # Initialize auth database - ensure database is created with schema
    from debate_engine.database.database import DatabaseManager
    # This will create the database with all tables including auth tables
    DatabaseManager()  # Initialize to ensure database exists with schema

    yield

    # Shutdown: Cancel cache cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Cache cleanup scheduler stopped")


async def cache_cleanup_scheduler():
    """Background task to periodically clean up expired cache entries."""
    while True:
        try:
            # Run cleanup every hour
            await asyncio.sleep(3600)  # 1 hour

            from models.cache_manager import cache_manager

            cleaned_count = cache_manager.cleanup_expired()

            if cleaned_count > 0:
                logger.info(
                    f"Scheduled cache cleanup: removed {cleaned_count} expired entries"
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cache cleanup scheduler: {e}")


# FastAPI app
app = FastAPI(
    title="Dialectus AI Debate System",
    description="Web interface for local AI model debates",
    version="1.0.0",
    lifespan=lifespan,
)

# Global debate manager
debate_manager = DebateManager()

# Include endpoint routers
app.include_router(models_router)
app.include_router(system_router)
app.include_router(debates_router)
app.include_router(debates_ws_router)
app.include_router(tournaments_router)
app.include_router(transcripts_router)
app.include_router(auth_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)