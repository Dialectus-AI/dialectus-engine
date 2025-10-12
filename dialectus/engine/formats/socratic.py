"""Socratic dialogue format implementation."""

from dialectus.engine.debate_engine.types import DebatePhase

from .base import DebateFormat, FormatPhase, Position
from .topic_themes import TopicTheme, TopicTone


class SocraticFormat(DebateFormat):
    """Socratic dialogue format focused on inquiry and discovery."""

    @property
    def name(self) -> str:
        return "socratic"

    @property
    def display_name(self) -> str:
        return "Socratic"

    @property
    def description(self) -> str:
        return (
            "Socratic dialogue using questions to explore ideas, challenge assumptions,"
            " and discover truth through inquiry"
        )

    def get_phases(self, participants: list[str]) -> list[FormatPhase]:
        """Socratic format: Initial Position -> Probing Questions -> Deep Inquiry -> 
        Reflection"""
        if len(participants) < 2:
            raise ValueError("Socratic format requires at least 2 participants")

        questioner = participants[0]  # Takes the "teacher/examiner" role
        responder = participants[1]  # Takes the "student/examined" role

        return [
            FormatPhase(
                phase=DebatePhase.OPENING,
                name="Initial Position Statement",
                instruction=(
                    "State your initial understanding or belief about the topic from"
                    " the responder perspective. Be open about what you know and what"
                    " you're uncertain about."
                ),
                speaking_order=[responder],
                time_multiplier=0.8,
            ),
            FormatPhase(
                phase=DebatePhase.OPENING,
                name="Opening Inquiry",
                instruction=(
                    "Begin probing the responder's initial position from the questioner"
                    " role. Ask clarifying questions about definitions and basic"
                    " assumptions."
                ),
                speaking_order=[questioner],
                time_multiplier=0.8,
            ),
            FormatPhase(
                phase=DebatePhase.CROSS_EXAM,
                name="First Round of Questioning",
                instruction=(
                    "Engage in the first round of Socratic inquiry. Questioner: probe"
                    " assumptions and explore implications. Responder: answer"
                    " thoughtfully and honestly."
                ),
                speaking_order=[questioner, responder, questioner, responder],
                time_multiplier=0.9,
            ),
            FormatPhase(
                phase=DebatePhase.CROSS_EXAM,
                name="Deep Inquiry Round",
                instruction=(
                    "Continue deeper exploration. Questioner: build on previous answers"
                    " to reveal contradictions or new insights. Responder: examine your"
                    " own thinking carefully."
                ),
                speaking_order=[questioner, responder, questioner, responder],
                time_multiplier=0.9,
            ),
            FormatPhase(
                phase=DebatePhase.REBUTTAL,
                name="Role Reversal",
                instruction=(
                    "Now the responder questions the questioner's approach or"
                    " assumptions. Explore what the questioning itself has revealed."
                ),
                speaking_order=[responder, questioner],
                time_multiplier=0.7,
            ),
            FormatPhase(
                phase=DebatePhase.CLOSING,
                name="Collaborative Reflection",
                instruction=(
                    "Reflect together on what has been discovered through this"
                    " dialogue. What new understanding or questions have emerged? What"
                    " remains to be explored?"
                ),
                speaking_order=[responder, questioner],
                time_multiplier=1.2,
            ),
        ]

    def get_position_assignments(self, participants: list[str]) -> dict[str, Position]:
        """In Socratic dialogue, assign questioner and responder roles rather than 
        pro/con."""
        assignments: dict[str, Position] = {}
        for i, participant in enumerate(participants[:2]):
            # Use PRO for questioner (teacher), CON for responder (student)
            assignments[participant] = Position.PRO if i == 0 else Position.CON
        return assignments

    def get_side_labels(self, participants: list[str]) -> dict[str, str]:
        """Return Socratic labels: Questioner and Responder."""
        labels: dict[str, str] = {}
        for i, participant in enumerate(participants[:2]):
            labels[participant] = "Questioner" if i == 0 else "Responder"
        return labels

    def get_side_descriptions(self, participants: list[str]) -> dict[str, str]:
        """Return descriptions for each role in Socratic dialogue."""
        descriptions: dict[str, str] = {}
        for i, participant in enumerate(participants[:2]):
            if i == 0:
                descriptions[participant] = (
                    "This AI will probe and question assumptions through inquiry."
                )
            else:
                descriptions[participant] = (
                    "This AI will respond to questions and explore ideas through"
                    " dialogue."
                )
        return descriptions

    def get_format_instructions(self) -> str:
        """Socratic format-specific instructions."""
        return ("SOCRATIC DIALOGUE FORMAT:"
            "- Focus on collaborative inquiry and discovery, not adversarial debate"
            "- Questions should probe assumptions, definitions, and underlying beliefs"
            "- Answers should be honest and reflective, acknowledging uncertainty"
            "- Build on previous exchanges to explore deeper implications"
            "- Use the method of elenchus (cross-examination) to examine ideas"
            "- Embrace intellectual humility and the 'wisdom of knowing that you don't "
            "know'"

            "QUESTIONER ROLE (Teacher/Examiner):"
            "- Ask open-ended questions that lead to insight, not trap the responder"
            "- Follow up on answers to explore contradictions and hidden assumptions  "
            "- Guide the dialogue toward deeper understanding"

            "RESPONDER ROLE (Student/Examined):"
            "- Answer honestly and examine your own thinking process"
            "- Admit when you're uncertain or when questions reveal flaws in your "
            "reasoning"
            "- Be willing to revise your understanding based on the inquiry"

            "Remember: The goal is mutual learning and truth-seeking, not winning or "
            "losing.")

    def get_min_participants(self) -> int:
        """Socratic works best with exactly 2 participants."""
        return 2

    def get_max_participants(self) -> int:
        """Socratic works best with exactly 2 participants."""
        return 2

    def get_topic_generation_messages(
        self,
        theme: TopicTheme | None = None,
        tone: TopicTone | None = None,
    ) -> list[dict[str, str]]:
        """Get Socratic specific messages for AI topic generation.

        Args:
            theme: Optional high-level theme to guide topic generation
            tone: Optional specific tone/sub-theme for more granular control

        Returns:
            List of message dictionaries for the AI model
        """
        # Customize system prompt for Socratic format
        socratic_system = (
            "You are an expert at generating Socratic dialogue topics. Socratic"
            " dialogues explore fundamental questions about knowledge, ethics, truth,"
            " and human nature through inquiry and questioning."
        )

        # Build theme/tone requirements using base class helper
        theme_tone_requirements = self._build_theme_tone_requirements(theme, tone)

        # Build Socratic-specific user prompt
        socratic_user = (
            "Generate a single topic suitable for Socratic dialogue that explores"
            " fundamental philosophical questions. The topic should invite deep"
            " inquiry about concepts like truth, justice, knowledge, virtue, or human"
            f" nature.{theme_tone_requirements} Make it thought-provoking for"
            " philosophical exploration. Respond with just the topic statement, no"
            " additional text or explanation."
        )

        return [
            {"role": "system", "content": socratic_system},
            {"role": "user", "content": socratic_user},
        ]
