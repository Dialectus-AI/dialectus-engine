"""Shared types and enums for the debate engine."""

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypedDict


class PhaseStartedEventData(TypedDict):
    """Data structure for phase_started event callbacks."""

    phase: str
    instruction: str
    current_phase: int
    total_phases: int
    progress_percentage: int


class MessageStartEventData(TypedDict):
    """Data structure for message_start event callbacks."""

    message_id: str
    speaker_id: str
    position: str
    phase: str
    round_number: int


class MessageCompleteEventData(TypedDict):
    """Data structure for message_complete event callbacks."""

    message_id: str
    speaker_id: str
    position: str
    content: str
    phase: str
    round_number: int
    generation_time_ms: int


# Callback type aliases for debate engine events
type PhaseEventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
type MessageEventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
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
