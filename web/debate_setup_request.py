from typing import Dict, Optional, List
from pydantic import BaseModel, field_validator
from config.settings import ModelConfig


class DebateSetupRequest(BaseModel):
    """Request model for creating a new debate."""

    topic: str
    format: str = "oxford"
    word_limit: int = 200
    models: Dict[str, ModelConfig]
    judging_method: str = "ai"
    judge_models: Optional[List[str]] = None
    judge_provider: Optional[str] = None

    @field_validator('judge_models')
    @classmethod
    def validate_judge_models(cls, v):
        """Validate judge_models field."""
        if v is None:
            return v
        if not all(isinstance(model, str) for model in v):
            raise ValueError("All judge models must be strings")
        return v
