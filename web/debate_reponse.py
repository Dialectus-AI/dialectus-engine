from pydantic import BaseModel
from config.settings import ModelConfig


class DebateResponse(BaseModel):
    """Response model for debate information."""

    id: str
    topic: str
    format: str
    status: str
    current_round: int
    current_phase: str
    message_count: int
    # Full configuration details for frontend
    word_limit: int | None = None
    models: dict[str, ModelConfig] | None = None
    judge_models: list[str] | None = None
    side_labels: dict[str, str] | None = None  # Format-specific participant labels
