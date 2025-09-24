"""Tournament management endpoints."""

import logging

from fastapi import HTTPException, APIRouter

from config.settings import get_default_config
from models.manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Tournament endpoints
tournament_manager = None
tournament_api = None


def get_tournament_api():
    """Get or create tournament API instance."""
    global tournament_manager, tournament_api
    if tournament_api is None:
        # Import debate manager from main api
        from web import api
        debate_manager = api.debate_manager

        config = get_default_config()
        model_manager = ModelManager(config.system)

        from tournaments import (
            TournamentManager,
            TournamentAPI,
            patch_debate_manager_for_tournaments,
        )

        tournament_manager = TournamentManager(
            db_path="tournaments.db",
            model_manager=model_manager,
            debate_manager=debate_manager,
        )
        tournament_api = TournamentAPI(tournament_manager)

        # Patch debate manager for tournament callbacks
        patch_debate_manager_for_tournaments(debate_manager, tournament_manager)

    return tournament_api


@router.post("/tournaments")
async def create_tournament(request: dict):
    """Create a new tournament."""
    from tournaments.models import TournamentCreateRequest

    try:
        # Convert dict to TournamentCreateRequest
        tournament_request = TournamentCreateRequest(**request)
        api = get_tournament_api()
        return await api.create_tournament(tournament_request)
    except Exception as e:
        logger.error(f"Failed to create tournament: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tournaments")
async def list_tournaments(limit: int | None = None, offset: int = 0):
    """List all tournaments."""
    api = get_tournament_api()
    return await api.list_tournaments(limit, offset)


@router.get("/tournaments/{tournament_id}")
async def get_tournament(tournament_id: int):
    """Get tournament details."""
    api = get_tournament_api()
    return await api.get_tournament(tournament_id)


@router.post("/tournaments/{tournament_id}/start")
async def start_tournament(tournament_id: int):
    """Start a tournament."""
    api = get_tournament_api()
    return await api.start_tournament(tournament_id)


@router.post("/tournaments/{tournament_id}/advance")
async def advance_tournament(tournament_id: int):
    """Advance tournament to next round."""
    api = get_tournament_api()
    return await api.advance_tournament(tournament_id)


@router.get("/tournaments/{tournament_id}/bracket")
async def get_tournament_bracket(tournament_id: int):
    """Get tournament bracket visualization data."""
    api = get_tournament_api()
    return await api.get_bracket(tournament_id)


@router.get("/tournaments/{tournament_id}/matches")
async def get_tournament_matches(tournament_id: int, round_number: int | None = None):
    """Get tournament matches, optionally filtered by round."""
    api = get_tournament_api()
    return await api.get_matches(tournament_id, round_number)


@router.post("/tournaments/{tournament_id}/cancel")
async def cancel_tournament(tournament_id: int):
    """Cancel a tournament."""
    api = get_tournament_api()
    return await api.cancel_tournament(tournament_id)


@router.delete("/tournaments/{tournament_id}")
async def delete_tournament(tournament_id: int):
    """Delete a tournament and all related data."""
    api = get_tournament_api()
    return await api.delete_tournament(tournament_id)


@router.get("/tournaments/{tournament_id}/rounds/{round_number}/status")
async def get_round_status(tournament_id: int, round_number: int):
    """Get status of all matches in a specific round."""
    api = get_tournament_api()
    return await api.get_round_status(tournament_id, round_number)


@router.get("/tournaments/{tournament_id}/participants")
async def get_tournament_participants(tournament_id: int):
    """Get all participants in a tournament."""
    api = get_tournament_api()
    return await api.get_tournament_participants(tournament_id)