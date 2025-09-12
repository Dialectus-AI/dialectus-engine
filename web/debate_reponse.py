from typing import Dict, Optional, List
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
    word_limit: Optional[int] = None
    models: Optional[Dict[str, ModelConfig]] = None
    judging_method: Optional[str] = None
    judge_models: Optional[List[str]] = None
    side_labels: Optional[Dict[str, str]] = None  # Format-specific participant labels
