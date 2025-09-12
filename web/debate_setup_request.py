from typing import Dict, Optional
from pydantic import BaseModel
from config.settings import ModelConfig


class DebateSetupRequest(BaseModel):
    """Request model for creating a new debate."""

    topic: str
    format: str = "oxford"
    word_limit: int = 200
    models: Dict[str, ModelConfig]
    judging_method: str = "ai"
    judge_model: Optional[str] = None
    judge_provider: Optional[str] = None
