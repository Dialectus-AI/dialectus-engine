"""Base classes and interfaces for debate formats."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from dialectus.engine.debate_engine.types import DebatePhase, Position
from dialectus.engine.formats.topic_themes import (
    THEME_DESCRIPTIONS,
    TONE_DESCRIPTIONS,
    TopicTheme,
    TopicTone,
)


@dataclass
class FormatPhase:
    """A phase within a specific debate format."""

    phase: DebatePhase
    name: str
    instruction: str
    speaking_order: list[str]  # Order of speaker IDs
    time_multiplier: float = 1.0  # Multiply base time limit


class DebateFormat(ABC):
    """Abstract base class for debate formats."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Format name."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable format name for display in UI."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Format description."""
        pass

    @abstractmethod
    def get_phases(self, participants: list[str]) -> list[FormatPhase]:
        """Get the phases for this format given participants."""
        pass

    @abstractmethod
    def get_position_assignments(self, participants: list[str]) -> dict[str, Position]:
        """Assign positions to participants."""
        pass

    @abstractmethod
    def get_format_instructions(self) -> str:
        """Get format-specific instructions for system prompts."""
        pass

    @abstractmethod
    def get_side_labels(self, participants: list[str]) -> dict[str, str]:
        """Get format-specific labels for each participant's side/role."""
        pass

    @abstractmethod
    def get_side_descriptions(self, participants: list[str]) -> dict[str, str]:
        """Get format-specific descriptions explaining each participant's role."""
        pass

    def get_max_participants(self) -> int:
        """Maximum number of participants supported."""
        return 2

    def get_min_participants(self) -> int:
        """Minimum number of participants required."""
        return 2

    def _build_theme_tone_requirements(
        self,
        theme: TopicTheme | None,
        tone: TopicTone | None,
    ) -> str:
        """Build theme/tone requirement string for prompts.

        Args:
            theme: Optional high-level theme
            tone: Optional specific tone

        Returns:
            String with theme/tone requirements (may be empty)
        """
        requirements = ""

        # Add theme guidance if provided
        if theme is not None:
            theme_desc = THEME_DESCRIPTIONS.get(theme, "")
            if theme_desc:
                requirements += f" The topic should be {theme_desc}"

        # Add tone guidance if provided (more specific than theme)
        if tone is not None:
            tone_desc = TONE_DESCRIPTIONS.get(tone, "")
            if tone_desc:
                requirements += f" Specifically, make it {tone_desc}."
        elif theme is not None:
            # Close the sentence if we only have theme
            requirements += "."

        return requirements

    def get_topic_generation_messages(
        self,
        theme: TopicTheme | None = None,
        tone: TopicTone | None = None,
    ) -> list[dict[str, str]]:
        """Get format-specific messages for AI topic generation.

        Args:
            theme: Optional high-level theme to guide topic generation
            tone: Optional specific tone/sub-theme for more granular control

        Returns:
            List of message dictionaries for the AI model
        """
        # Build the system prompt
        system_content = (
            "You are an expert debate topic generator. Create engaging, balanced, and"
            " thought-provoking debate topics that have clear pro and con sides."
        )

        # Build theme/tone requirements
        theme_tone_requirements = self._build_theme_tone_requirements(theme, tone)

        # Build the user prompt
        user_content = (
            "Generate a single debate topic that would make for an interesting and"
            " balanced debate. The topic should be phrased as a clear statement that"
            f" can be argued for or against.{theme_tone_requirements} Make it"
            " thought-provoking and suitable for intellectual discourse. Respond with"
            " just the topic statement, no additional text or explanation."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
