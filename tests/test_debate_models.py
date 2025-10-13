"""Tests for debate data models."""

from __future__ import annotations

from datetime import datetime

import pytest

from dialectus.engine.config.settings import ModelConfig
from dialectus.engine.debate_engine.models import DebateContext, DebateMessage
from dialectus.engine.debate_engine.types import DebatePhase, Position


def test_debate_message_defaults() -> None:
    """DebateMessage assigns default timestamp and metadata/cost fields."""
    message = DebateMessage(
        speaker_id="model_a",
        position=Position.PRO,
        phase=DebatePhase.OPENING,
        round_number=1,
        content="Opening argument",
    )

    assert message.timestamp <= datetime.now()
    assert message.metadata == {}
    assert message.message_id is None
    assert message.cost is None
    assert message.generation_id is None
    assert message.cost_queried_at is None


def test_debate_message_metadata_is_isolated_between_instances() -> None:
    """Default metadata dict should not be shared across instances."""
    first = DebateMessage(
        speaker_id="model_a",
        position=Position.PRO,
        phase=DebatePhase.OPENING,
        round_number=1,
        content="Message one",
    )
    second = DebateMessage(
        speaker_id="model_b",
        position=Position.CON,
        phase=DebatePhase.OPENING,
        round_number=1,
        content="Message two",
    )

    first.metadata["foo"] = "bar"
    assert second.metadata == {}
    assert first.metadata == {"foo": "bar"}


def test_debate_context_defaults(two_participants: list[str]) -> None:
    """DebateContext sets default phase, round, and empty containers."""
    participants = {
        pid: ModelConfig(name=f"{pid}-model", provider="ollama")
        for pid in two_participants
    }

    context = DebateContext(topic="Test topic", participants=participants)

    assert context.current_phase == DebatePhase.SETUP
    assert context.current_round == 1
    assert context.messages == []
    assert context.scores == {}
    assert context.metadata == {}


def test_debate_context_records_messages(two_participants: list[str]) -> None:
    """Adding messages updates the context message list without sharing defaults."""
    participants = {
        pid: ModelConfig(name=f"{pid}-model", provider="ollama")
        for pid in two_participants
    }
    context = DebateContext(topic="Topic", participants=participants)

    message = DebateMessage(
        speaker_id="model_a",
        position=Position.PRO,
        phase=DebatePhase.OPENING,
        round_number=1,
        content="Argument",
    )

    context.messages.append(message)

    assert len(context.messages) == 1
    assert context.messages[0].content == "Argument"
    assert context.messages[0] is message
