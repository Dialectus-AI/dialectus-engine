"""Debate format definitions and implementations."""

from .base import DebateFormat, FormatPhase
from .oxford import OxfordFormat
from .parliamentary import ParliamentaryFormat
from .socratic import SocraticFormat
from .registry import format_registry

__all__ = [
    'DebateFormat',
    'FormatPhase', 
    'OxfordFormat',
    'ParliamentaryFormat',
    'SocraticFormat',
    'format_registry'
]