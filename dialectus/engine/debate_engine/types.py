"""Shared types and enums for the debate engine."""

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Protocol

from pydantic import BaseModel


class PhaseEventType(str, Enum):
    """Event types for phase callbacks."""

    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"


class MessageEventType(str, Enum):
    """Event types for message callbacks."""

    MESSAGE_START = "message_start"
    MESSAGE_COMPLETE = "message_complete"


class PhaseStartedEventData(BaseModel):
    """Data structure for phase_started event callbacks."""

    phase: str
    instruction: str
    current_phase: int
    total_phases: int
    progress_percentage: int


class MessageStartEventData(BaseModel):
    """Data structure for message_start event callbacks."""

    message_id: str
    speaker_id: str
    position: str
    phase: str
    round_number: int


class MessageCompleteEventData(BaseModel):
    """Data structure for message_complete event callbacks."""

    message_id: str
    speaker_id: str
    position: str
    content: str
    phase: str
    round_number: int
    timestamp: str
    word_count: int
    metadata: dict[str, str]
    cost: float | None = None
    generation_id: str | None = None


# Callback protocols with proper type narrowing
class PhaseEventCallback(Protocol):
    """Protocol for phase event callbacks with type-safe event data."""

    async def __call__(
        self, event: PhaseEventType, data: PhaseStartedEventData
    ) -> None: ...


class MessageEventCallback(Protocol):
    """Protocol for message event callbacks with type-safe event data."""

    async def __call__(
        self,
        event: MessageEventType,
        data: MessageStartEventData | MessageCompleteEventData,
    ) -> None: ...


type ChunkCallback = Callable[[str, bool], Awaitable[None]]


class DebatePhase(Enum):
    """Phases of a debate."""

    SETUP = "setup"
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CROSS_EXAM = "cross_examination"
    CLOSING = "closing"
    EVALUATION = "evaluation"
    COMPLETED = "completed"


class Position(Enum):
    """Debate positions."""

    PRO = "pro"
    CON = "con"
    NEUTRAL = "neutral"
