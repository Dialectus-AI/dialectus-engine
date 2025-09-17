"""Data models for the debate engine."""

from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

from config.settings import ModelConfig
from .types import DebatePhase, Position


@dataclass
class DebateMessage:
    """A single message in the debate."""

    speaker_id: str
    position: Position
    phase: DebatePhase
    round_number: int
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateContext:
    """Full context of an ongoing debate."""

    topic: str
    participants: dict[str, ModelConfig]
    messages: list[DebateMessage] = field(default_factory=list)
    current_phase: DebatePhase = DebatePhase.SETUP
    current_round: int = 1
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
