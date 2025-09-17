"""Tournament system for AI debate competitions."""

from .manager import TournamentManager
from .database import TournamentDatabaseManager
from .api import TournamentAPI
from .debate_integration import TournamentDebateCallback, patch_debate_manager_for_tournaments
from .models import (
    Tournament,
    TournamentParticipant,
    TournamentMatch,
    TournamentJudge,
    TournamentStatus,
    TournamentCreateRequest,
    TournamentSummary,
    WeightClass,
    BracketData,
    MatchStatus,
    MatchResult,
    RoundStatus,
    TournamentProgressUpdate,
)

__all__ = [
    "TournamentManager",
    "TournamentDatabaseManager",
    "TournamentAPI",
    "TournamentDebateCallback",
    "patch_debate_manager_for_tournaments",
    "Tournament",
    "TournamentParticipant",
    "TournamentMatch",
    "TournamentJudge",
    "TournamentStatus",
    "TournamentCreateRequest",
    "TournamentSummary",
    "WeightClass",
    "BracketData",
    "MatchStatus",
    "MatchResult",
    "RoundStatus",
    "TournamentProgressUpdate",
]
