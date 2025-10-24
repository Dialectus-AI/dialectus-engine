"""Public Forum Debate format implementation."""

from dialectus.engine.debate_engine.types import DebatePhase
from dialectus.engine.models.providers.base_model_provider import ChatMessage

from .base import DebateFormat, FormatPhase, Position
from .topic_themes import TopicTheme, TopicTone


class PublicForumFormat(DebateFormat):
    """Public Forum Debate format for discussing current events and real-world
    issues."""

    @property
    def name(self) -> str:
        return "public_forum"

    @property
    def display_name(self) -> str:
        return "Public Forum"

    @property
    def description(self) -> str:
        return (
            "Public Forum Debate format for discussing current events and real-world"
            " issues with accessible arguments"
        )

    def get_phases(self, participants: list[str]) -> list[FormatPhase]:
        """Public Forum format: Constructive Speeches -> Crossfire ->
        Rebuttals -> Final Focus"""
        if len(participants) < 2:
            raise ValueError("Public Forum format requires at least 2 participants")

        advocate = participants[0]  # Supports the position/resolution
        opponent = participants[1]  # Opposes the position/resolution

        return [
            FormatPhase(
                phase=DebatePhase.OPENING,
                name="Advocate Constructive Speech",
                instruction=(
                    "Deliver your constructive speech supporting the resolution. Focus"
                    " on clear, accessible arguments that the general public can"
                    " understand. Provide concrete evidence and real-world examples."
                ),
                speaking_order=[advocate],
                time_multiplier=1.0,
            ),
            FormatPhase(
                phase=DebatePhase.OPENING,
                name="Opponent Constructive Speech",
                instruction=(
                    "Deliver your constructive speech opposing the resolution."
                    " Challenge the Advocate's framework and provide counter-arguments"
                    " with evidence. Focus on practical concerns and alternative"
                    " perspectives."
                ),
                speaking_order=[opponent],
                time_multiplier=1.0,
            ),
            FormatPhase(
                phase=DebatePhase.CROSS_EXAM,
                name="Crossfire Round One",
                instruction=(
                    "Engage in direct questioning and response. Ask strategic questions"
                    " to clarify positions, expose weaknesses, or advance your"
                    " argument. Keep exchanges dynamic and accessible."
                ),
                speaking_order=[advocate, opponent, advocate, opponent],
                time_multiplier=0.6,
            ),
            FormatPhase(
                phase=DebatePhase.REBUTTAL,
                name="Advocate Rebuttal",
                instruction=(
                    "Respond to the Opponent's constructive case. Attack their"
                    " strongest arguments while defending your own position. Begin to"
                    " frame the key voting issues."
                ),
                speaking_order=[advocate],
                time_multiplier=0.8,
            ),
            FormatPhase(
                phase=DebatePhase.REBUTTAL,
                name="Opponent Rebuttal",
                instruction=(
                    "Respond to the Advocate's constructive case. Challenge their"
                    " arguments and evidence while reinforcing your opposition to the"
                    " resolution."
                ),
                speaking_order=[opponent],
                time_multiplier=0.8,
            ),
            FormatPhase(
                phase=DebatePhase.CROSS_EXAM,
                name="Crossfire Round Two",
                instruction=(
                    "Second round of direct questioning. Focus on the clash that has"
                    " developed and key disagreements. Push for clarity on the most"
                    " important issues."
                ),
                speaking_order=[opponent, advocate, opponent, advocate],
                time_multiplier=0.6,
            ),
            FormatPhase(
                phase=DebatePhase.CLOSING,
                name="Final Focus Speeches",
                instruction=(
                    "Make your final appeal to the judges and audience. Focus on the"
                    " key voting issues, explain why your side wins the most important"
                    " arguments, and provide a clear decision framework."
                ),
                speaking_order=[advocate, opponent],
                time_multiplier=0.9,
            ),
        ]

    def get_position_assignments(self, participants: list[str]) -> dict[str, Position]:
        """First participant is Advocate (PRO), second is Opponent (CON)."""
        assignments: dict[str, Position] = {}
        for i, participant in enumerate(participants[:2]):
            assignments[participant] = Position.PRO if i == 0 else Position.CON
        return assignments

    def get_side_labels(self, participants: list[str]) -> dict[str, str]:
        """Return Public Forum labels: Advocate and Opponent."""
        labels: dict[str, str] = {}
        for i, participant in enumerate(participants[:2]):
            labels[participant] = "Advocate" if i == 0 else "Opponent"
        return labels

    def get_side_descriptions(self, participants: list[str]) -> dict[str, str]:
        """Return descriptions for each side in Public Forum debate."""
        descriptions: dict[str, str] = {}
        for i, participant in enumerate(participants[:2]):
            if i == 0:
                descriptions[participant] = (
                    "This AI will advocate in support of the resolution."
                )
            else:
                descriptions[participant] = "This AI will oppose the resolution."
        return descriptions

    def get_format_instructions(self) -> str:
        """Public Forum format-specific instructions."""
        return (
            "PUBLIC FORUM DEBATE FORMAT:"
            "- Focus on current events and real-world policy issues"
            "- Make arguments accessible to a general audience, not just debate experts"
            "- Use concrete evidence from credible sources (news, studies, expert "
            "opinions)"
            "- Emphasize practical impacts and real-world consequences"
            "- Engage in respectful but direct clash with your opponent"
            "- Crossfire should be conversational but strategic"
            "- Final Focus should clearly explain why your side wins the debate"
            "ADVOCATE ROLE:"
            "- Support the resolution with clear, compelling arguments"
            "- Provide evidence that the proposed policy/position is beneficial"
            "- Address practical concerns and implementation challenges"
            "OPPONENT ROLE:"
            "- Challenge the resolution with substantive counter-arguments"
            "- Highlight negative consequences and practical problems"
            "- Offer alternative solutions or perspectives when appropriate"
            "Remember: This is a public forum - your arguments should be clear and "
            "persuasive to ordinary citizens, not just debate judges."
        )

    def get_max_participants(self) -> int:
        """Public Forum format supports up to 4 participants (2 per side)."""
        return 4

    def get_topic_generation_messages(
        self,
        theme: TopicTheme | None = None,
        tone: TopicTone | None = None,
    ) -> list[ChatMessage]:
        """Get Public Forum specific messages for AI topic generation.

        Args:
            theme: Optional high-level theme to guide topic generation
            tone: Optional specific tone/sub-theme for more granular control

        Returns:
            List of message dictionaries for the AI model
        """
        # Customize system prompt for Public Forum format
        pf_system = (
            "You are an expert at generating Public Forum debate topics. Public Forum"
            " debates focus on current events, policy issues, and real-world problems"
            " that affect society. Topics should be accessible to general audiences,"
            " contemporary, and have clear practical implications."
        )

        # Build theme/tone requirements using base class helper
        theme_tone_requirements = self._build_theme_tone_requirements(theme, tone)

        # Build Public Forum-specific user prompt
        pf_user = (
            "Generate a single Public Forum debate topic suitable for high school or"
            " college students discussing current events and policy issues. The topic"
            " should be phrased as a clear statement that ordinary citizens could"
            f" understand and have opinions about.{theme_tone_requirements} Focus on"
            " real-world policy questions, not abstract philosophical concepts or"
            " highly technical subjects. Make it current and relevant to today's"
            " society. Respond with just the topic statement, no additional text or"
            " explanation."
        )

        return [
            {"role": "system", "content": pf_system},
            {"role": "user", "content": pf_user},
        ]
