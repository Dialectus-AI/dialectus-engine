"""Tests for debate format base classes and implementations."""

from __future__ import annotations

import pytest

from dialectus.engine.debate_engine.types import DebatePhase, Position
from dialectus.engine.formats.base import DebateFormat, FormatPhase
from dialectus.engine.formats.oxford import OxfordFormat
from dialectus.engine.formats.parliamentary import ParliamentaryFormat
from dialectus.engine.formats.public_forum import PublicForumFormat
from dialectus.engine.formats.socratic import SocraticFormat
from dialectus.engine.formats.topic_themes import (
    THEME_DESCRIPTIONS,
    TONE_DESCRIPTIONS,
    TopicTheme,
    TopicTone,
)

FORMAT_CASES: list[tuple[type[DebateFormat], tuple[str, str], tuple[str, str]]] = [
    (OxfordFormat, ("Proposition", "Opposition"), ("support", "against")),
    (ParliamentaryFormat, ("Government", "Opposition"), ("defend", "challenge")),
    (PublicForumFormat, ("Advocate", "Opponent"), ("advocate", "oppose")),
    (SocraticFormat, ("Questioner", "Responder"), ("probe", "respond")),
]


def test_format_phase_default_time_multiplier() -> None:
    """FormatPhase defaults to time multiplier of 1.0 when unspecified."""
    phase = FormatPhase(
        phase=DebatePhase.OPENING,
        name="Opening",
        instruction="State your case.",
        speaking_order=["model_a"],
    )

    assert phase.time_multiplier == 1.0


@pytest.mark.parametrize("format_cls", [case[0] for case in FORMAT_CASES])
def test_formats_require_minimum_participants(format_cls: type[DebateFormat]) -> None:
    """All formats require at least two participants to build phases."""
    fmt = format_cls()
    with pytest.raises(ValueError):
        fmt.get_phases(["only_one"])


@pytest.mark.parametrize("format_cls", [case[0] for case in FORMAT_CASES])
def test_format_phases_use_participants(
    format_cls: type[DebateFormat],
    two_participants: list[str],
) -> None:
    """get_phases returns structured FormatPhase items using provided participants."""
    fmt = format_cls()
    phases = fmt.get_phases(two_participants)

    assert len(phases) > 0
    for phase in phases:
        assert isinstance(phase, FormatPhase)
        assert phase.name
        assert phase.instruction
        # Speaking order should be limited to the provided participants
        assert set(phase.speaking_order).issubset(set(two_participants))


@pytest.mark.parametrize(
    "format_cls,labels_expected,description_keywords", FORMAT_CASES
)
def test_format_assignments_labels_and_descriptions(
    format_cls: type[DebateFormat],
    labels_expected: tuple[str, str],
    description_keywords: tuple[str, str],
    two_participants: list[str],
) -> None:
    """Verify side mapping, labels, and description semantics for each format."""
    fmt = format_cls()
    participants = two_participants

    assignments = fmt.get_position_assignments(participants)
    assert assignments[participants[0]] is Position.PRO
    assert assignments[participants[1]] is Position.CON

    labels = fmt.get_side_labels(participants)
    assert labels[participants[0]] == labels_expected[0]
    assert labels[participants[1]] == labels_expected[1]

    descriptions = fmt.get_side_descriptions(participants)
    assert description_keywords[0] in descriptions[participants[0]].lower()
    assert description_keywords[1] in descriptions[participants[1]].lower()

    instructions = fmt.get_format_instructions()
    assert isinstance(instructions, str) and instructions


@pytest.mark.parametrize("format_cls", [case[0] for case in FORMAT_CASES])
def test_topic_generation_messages_include_theme_and_tone(
    format_cls: type[DebateFormat],
) -> None:
    """Theme/tone requirements should appear in user prompt for every format."""
    fmt = format_cls()
    messages = fmt.get_topic_generation_messages(
        theme=TopicTheme.TECHNOLOGY,
        tone=TopicTone.INNOVATION,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_prompt = messages[1]["content"]
    theme_text = THEME_DESCRIPTIONS[TopicTheme.TECHNOLOGY]
    tone_text = TONE_DESCRIPTIONS[TopicTone.INNOVATION]

    assert "modern technological developments" in theme_text
    assert "modern technological developments" in user_prompt
    assert "emerging innovations" in user_prompt

    assert "breakthroughs" in tone_text
    assert "breakthroughs" in user_prompt
