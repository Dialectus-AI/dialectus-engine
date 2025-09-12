from pydantic import BaseModel


class MessageResponse(BaseModel):
    """Response model for debate messages."""

    speaker_id: str
    position: str
    phase: str
    round_number: int
    content: str
    timestamp: str
    word_count: int
