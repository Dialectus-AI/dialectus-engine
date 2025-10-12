"""Debate format definitions and implementations."""

from .base import DebateFormat, FormatPhase
from .oxford import OxfordFormat
from .parliamentary import ParliamentaryFormat
from .public_forum import PublicForumFormat
from .registry import format_registry
from .socratic import SocraticFormat
from .topic_themes import (
    THEME_DESCRIPTIONS,
    THEME_TONES,
    TONE_DESCRIPTIONS,
    TopicTheme,
    TopicTone,
    get_theme_display_name,
    get_tone_display_name,
)

__all__ = [
    "DebateFormat",
    "FormatPhase",
    "OxfordFormat",
    "ParliamentaryFormat",
    "SocraticFormat",
    "PublicForumFormat",
    "format_registry",
    "TopicTheme",
    "TopicTone",
    "THEME_DESCRIPTIONS",
    "TONE_DESCRIPTIONS",
    "THEME_TONES",
    "get_theme_display_name",
    "get_tone_display_name",
]
