"""Shared types and enums for the debate engine."""

from enum import Enum


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
