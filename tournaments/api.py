"""Tournament API endpoints."""

import logging
from typing import Any

from fastapi import HTTPException

from .manager import TournamentManager
from .models import (
    TournamentCreateRequest,
    TournamentStatus,
)

logger = logging.getLogger(__name__)


class TournamentAPI:
    """FastAPI endpoint handlers for tournament operations."""

    def __init__(self, tournament_manager: TournamentManager):
        self.manager = tournament_manager

    async def create_tournament(
        self, request: TournamentCreateRequest
    ) -> dict[str, Any]:
        """Create a new tournament."""
        try:
            # Validate bracket size
            valid_sizes = [4, 8, 16, 32, 64]
            if request.bracket_size not in valid_sizes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid bracket size. Must be one of: {valid_sizes}",
                )

            # Create tournament
            tournament_id = await self.manager.create_tournament(request)

            return {
                "tournament_id": tournament_id,
                "message": f"Tournament '{request.name}' created successfully",
                "bracket_size": request.bracket_size,
                "weight_class": request.weight_class.value,
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to create tournament: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def list_tournaments(
        self, limit: int | None = None, offset: int = 0
    ) -> dict[str, Any]:
        """List all tournaments."""
        try:
            tournaments = self.manager.db.list_tournaments(limit, offset)

            return {
                "tournaments": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "weight_class": t.weight_class.value,
                        "status": t.status.value,
                        "bracket_size": t.bracket_size,
                        "current_round": t.current_round,
                        "total_rounds": t.total_rounds,
                        "created_at": (
                            t.created_at.isoformat() if t.created_at else None
                        ),
                        "winner_model_id": t.winner_model_id,
                    }
                    for t in tournaments
                ],
                "count": len(tournaments),
            }

        except Exception as e:
            logger.error(f"Failed to list tournaments: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_tournament(self, tournament_id: int) -> dict[str, Any]:
        """Get tournament details."""
        try:
            tournament = self.manager.get_tournament_status(tournament_id)
            if not tournament:
                raise HTTPException(status_code=404, detail="Tournament not found")

            return {
                "id": tournament.id,
                "name": tournament.name,
                "weight_class": tournament.weight_class.value,
                "format": tournament.format,
                "word_limit": tournament.word_limit,
                "status": tournament.status.value,
                "bracket_size": tournament.bracket_size,
                "current_round": tournament.current_round,
                "total_rounds": tournament.total_rounds,
                "created_at": (
                    tournament.created_at.isoformat() if tournament.created_at else None
                ),
                "started_at": (
                    tournament.started_at.isoformat() if tournament.started_at else None
                ),
                "completed_at": (
                    tournament.completed_at.isoformat()
                    if tournament.completed_at
                    else None
                ),
                "winner_model_id": tournament.winner_model_id,
                "tournament_metadata": tournament.tournament_metadata,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get tournament {tournament_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def start_tournament(self, tournament_id: int) -> dict[str, Any]:
        """Start a tournament."""
        try:
            success = await self.manager.start_tournament(tournament_id)
            if not success:
                raise HTTPException(
                    status_code=400, detail="Failed to start tournament"
                )

            return {
                "tournament_id": tournament_id,
                "message": "Tournament started successfully",
                "status": "in_progress",
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to start tournament {tournament_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def advance_tournament(self, tournament_id: int) -> dict[str, Any]:
        """Advance tournament to next round."""
        try:
            success = await self.manager.advance_tournament(tournament_id)
            if not success:
                raise HTTPException(
                    status_code=400, detail="Cannot advance tournament at this time"
                )

            # Get updated tournament status
            tournament = self.manager.get_tournament_status(tournament_id)
            status = tournament.status.value if tournament else "unknown"

            return {
                "tournament_id": tournament_id,
                "message": "Tournament advanced successfully",
                "status": status,
                "current_round": tournament.current_round if tournament else None,
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to advance tournament {tournament_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_bracket(self, tournament_id: int) -> dict[str, Any]:
        """Get tournament bracket visualization data."""
        try:
            bracket_data = self.manager.get_bracket_view(tournament_id)
            if not bracket_data:
                raise HTTPException(status_code=404, detail="Tournament not found")

            return {
                "tournament": {
                    "id": bracket_data.tournament.id,
                    "name": bracket_data.tournament.name,
                    "weight_class": bracket_data.tournament.weight_class.value,
                    "status": bracket_data.tournament.status.value,
                    "current_round": bracket_data.tournament.current_round,
                    "total_rounds": bracket_data.tournament.total_rounds,
                },
                "participants": [
                    {
                        "id": p.id,
                        "model_id": p.model_id,
                        "model_name": p.model_name,
                        "seed_number": p.seed_number,
                        "eliminated_in_round": p.eliminated_in_round,
                        "is_active": p.eliminated_in_round is None,
                    }
                    for p in bracket_data.participants
                ],
                "matches": [
                    {
                        "id": m.id,
                        "round_number": m.round_number,
                        "match_number": m.match_number,
                        "model_a_id": m.model_a_id,
                        "model_b_id": m.model_b_id,
                        "winner_model_id": m.winner_model_id,
                        "debate_id": m.debate_id,
                        "topic": m.topic,
                        "status": m.status.value,
                        "is_bye": m.model_b_id is None,
                    }
                    for m in bracket_data.matches
                ],
                "judges": [
                    {
                        "id": j.id,
                        "judge_model_id": j.judge_model_id,
                        "judge_provider": j.judge_provider,
                    }
                    for j in bracket_data.judges
                ],
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get bracket for tournament {tournament_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_matches(
        self, tournament_id: int, round_number: int | None = None
    ) -> dict[str, Any]:
        """Get tournament matches, optionally filtered by round."""
        try:
            matches = self.manager.db.get_matches(tournament_id, round_number)

            return {
                "tournament_id": tournament_id,
                "round_number": round_number,
                "matches": [
                    {
                        "id": m.id,
                        "round_number": m.round_number,
                        "match_number": m.match_number,
                        "model_a_id": m.model_a_id,
                        "model_b_id": m.model_b_id,
                        "winner_model_id": m.winner_model_id,
                        "debate_id": m.debate_id,
                        "topic": m.topic,
                        "status": m.status.value,
                        "is_bye": m.model_b_id is None,
                    }
                    for m in matches
                ],
                "count": len(matches),
            }

        except Exception as e:
            logger.error(f"Failed to get matches for tournament {tournament_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def cancel_tournament(self, tournament_id: int) -> dict[str, Any]:
        """Cancel a tournament."""
        try:
            # Update tournament status to cancelled
            success = self.manager.db.update_tournament_status(
                tournament_id, TournamentStatus.CANCELLED
            )

            if not success:
                raise HTTPException(status_code=404, detail="Tournament not found")

            return {
                "tournament_id": tournament_id,
                "message": "Tournament cancelled successfully",
                "status": "cancelled",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel tournament {tournament_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def delete_tournament(self, tournament_id: int) -> dict[str, Any]:
        """Delete a tournament and all related data."""
        try:
            success = self.manager.db.delete_tournament(tournament_id)
            if not success:
                raise HTTPException(status_code=404, detail="Tournament not found")

            return {
                "tournament_id": tournament_id,
                "message": "Tournament deleted successfully",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete tournament {tournament_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_round_status(
        self, tournament_id: int, round_number: int
    ) -> dict[str, Any]:
        """Get status of all matches in a specific round."""
        try:
            round_status = self.manager.db.get_round_status(tournament_id, round_number)

            return {
                "tournament_id": tournament_id,
                "round_number": round_number,
                "total_matches": round_status["total_matches"],
                "completed_matches": round_status["completed_matches"],
                "pending_matches": round_status["pending_matches"],
                "in_progress_matches": round_status["in_progress_matches"],
                "all_completed": round_status["all_completed"],
                "completion_percentage": (
                    round_status["completed_matches"]
                    / round_status["total_matches"]
                    * 100
                    if round_status["total_matches"] > 0
                    else 0
                ),
            }

        except Exception as e:
            logger.error(
                f"Failed to get round status for tournament {tournament_id}, round {round_number}: {e}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_tournament_participants(self, tournament_id: int) -> dict[str, Any]:
        """Get all participants in a tournament."""
        try:
            participants = self.manager.db.get_participants(tournament_id)

            return {
                "tournament_id": tournament_id,
                "participants": [
                    {
                        "id": p.id,
                        "model_id": p.model_id,
                        "model_name": p.model_name,
                        "seed_number": p.seed_number,
                        "eliminated_in_round": p.eliminated_in_round,
                        "is_active": p.eliminated_in_round is None,
                    }
                    for p in participants
                ],
                "count": len(participants),
                "active_count": len(
                    [p for p in participants if p.eliminated_in_round is None]
                ),
            }

        except Exception as e:
            logger.error(
                f"Failed to get participants for tournament {tournament_id}: {e}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")
