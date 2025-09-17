"""Tournament system data models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WeightClass(Enum):
    """Tournament weight classes based on model cost."""

    FREE = "free"  # 🆓 Ollama + legitimate free OpenRouter (no preview)
    BUDGET = "budget"  # 💰 $0 - $0.001/1K tokens
    ECONOMY = "economy"  # 💼 $0.001 - $0.005/1K tokens
    PREMIUM = "premium"  # 💎 $0.005 - $0.02/1K tokens
    ELITE = "elite"  # 👑 $0.02+/1K tokens


class TournamentStatus(Enum):
    """Tournament execution status."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MatchStatus(Enum):
    """Individual match status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BYE = "bye"


class TournamentCreateRequest(BaseModel):
    """Request to create a new tournament."""

    name: str = Field(..., description="Tournament name")
    weight_class: WeightClass = Field(..., description="Model weight class")
    format: str = Field(default="oxford", description="Debate format")
    word_limit: int = Field(default=500, description="Word limit per speech")
    bracket_size: int = Field(
        default=8, description="Tournament bracket size (4, 8, 16, 32, 64)"
    )
    judge_models: list[str] = Field(
        default=["llama3.1:8b"], description="Judge model IDs"
    )
    judge_provider: str = Field(default="ollama", description="Judge provider")


class TournamentJudge(BaseModel):
    """Tournament judge configuration."""

    id: int | None = None
    tournament_id: int
    judge_model_id: str
    judge_provider: str


class TournamentParticipant(BaseModel):
    """Tournament participant (seeded model)."""

    id: int | None = None
    tournament_id: int
    model_id: str
    model_name: str
    seed_number: int  # 1-64 seeding position
    eliminated_in_round: int | None = None  # NULL if still active


class TournamentMatch(BaseModel):
    """Individual tournament match."""

    id: int | None = None
    tournament_id: int
    round_number: int
    match_number: int  # Match within round
    model_a_id: str
    model_b_id: str | None = None  # NULL for bye matches
    winner_model_id: str | None = None  # NULL until debate completes
    debate_id: str | None = None  # Links to debates table
    topic: str  # Unique topic per match
    status: MatchStatus


class Tournament(BaseModel):
    """Complete tournament information."""

    id: int | None = None
    name: str
    weight_class: WeightClass
    format: str
    word_limit: int
    status: TournamentStatus
    bracket_size: int
    current_round: int = 1
    total_rounds: int
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    winner_model_id: str | None = None
    tournament_metadata: dict[str, Any] | None = None


class BracketData(BaseModel):
    """Tournament bracket visualization data."""

    tournament: Tournament
    participants: list[TournamentParticipant]
    matches: list[TournamentMatch]
    judges: list[TournamentJudge]


class TournamentSummary(BaseModel):
    """Tournament summary for listing."""

    id: int
    name: str
    weight_class: WeightClass
    status: TournamentStatus
    bracket_size: int
    current_round: int
    total_rounds: int
    created_at: datetime
    winner_model_id: str | None = None


class MatchResult(BaseModel):
    """Result of a completed match."""

    match_id: int
    winner_model_id: str
    debate_id: str
    completed_at: datetime


class RoundStatus(BaseModel):
    """Status of all matches in a round."""

    round_number: int
    total_matches: int
    completed_matches: int
    pending_matches: int
    in_progress_matches: int
    all_completed: bool


class TournamentProgressUpdate(BaseModel):
    """WebSocket update for tournament progression."""

    tournament_id: int
    type: str  # 'match_started', 'match_completed', 'round_completed', 'tournament_completed'
    round_number: int
    match_id: int | None = None
    winner_model_id: str | None = None
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
