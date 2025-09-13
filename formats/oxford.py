"""Oxford-style debate format implementation."""

from typing import List, Dict
from .base import DebateFormat, FormatPhase, Position
from debate_engine.types import DebatePhase


class OxfordFormat(DebateFormat):
    """Oxford Union-style debate with formal structure and balanced argument exchange."""

    @property
    def name(self) -> str:
        return "oxford"

    @property
    def display_name(self) -> str:
        return "Oxford"

    @property
    def description(self) -> str:
        return "Oxford Union-style debate with equal time allocation, formal procedure, and structured argument exchange"

    def get_phases(self, participants: List[str]) -> List[FormatPhase]:
        """Oxford format: Proposition Opening -> Opposition Opening -> Rebuttals -> Cross-Examination -> Closing"""
        if len(participants) < 2:
            raise ValueError("Oxford format requires at least 2 participants")

        proposition = participants[0]  # Supports the motion
        opposition = participants[1]  # Opposes the motion

        return [
            FormatPhase(
                phase=DebatePhase.OPENING,
                name="Proposition Opening Statement",
                instruction="As the Proposition, present your case supporting the motion. Establish key arguments, provide evidence, and set the framework for debate.",
                speaking_order=[proposition],
                time_multiplier=1.0,
            ),
            FormatPhase(
                phase=DebatePhase.OPENING,
                name="Opposition Opening Statement",
                instruction="As the Opposition, challenge the motion and the Proposition's framework. Present your counter-case with strong evidence and reasoning.",
                speaking_order=[opposition],
                time_multiplier=1.0,
            ),
            FormatPhase(
                phase=DebatePhase.REBUTTAL,
                name="Opposition Rebuttal",
                instruction="Address the Proposition's arguments directly. Identify weaknesses and reinforce your opposition to the motion.",
                speaking_order=[opposition],
                time_multiplier=0.9,
            ),
            FormatPhase(
                phase=DebatePhase.REBUTTAL,
                name="Proposition Rebuttal",
                instruction="Defend your opening case against Opposition attacks. Counter their objections and strengthen your support for the motion.",
                speaking_order=[proposition],
                time_multiplier=0.9,
            ),
            FormatPhase(
                phase=DebatePhase.CROSS_EXAM,
                name="Cross-Examination Round",
                instruction="Engage in strategic questioning and answering. Ask pointed questions to expose flaws or seek clarification on your opponent's position.",
                speaking_order=[
                    proposition,
                    opposition,
                    proposition,
                    opposition,
                ],  # Alternating Q&A
                time_multiplier=0.7,
            ),
            FormatPhase(
                phase=DebatePhase.CLOSING,
                name="Final Statements",
                instruction="Make your final appeal to the audience. Summarize your strongest points, address your opponent's best arguments, and conclude persuasively.",
                speaking_order=[proposition, opposition],
                time_multiplier=1.1,
            ),
        ]

    def get_position_assignments(self, participants: List[str]) -> Dict[str, Position]:
        """First participant is PRO, second is CON."""
        assignments = {}
        for i, participant in enumerate(participants[:2]):
            assignments[participant] = Position.PRO if i == 0 else Position.CON
        return assignments

    def get_side_labels(self, participants: List[str]) -> Dict[str, str]:
        """Return Oxford-style labels: Proposition and Opposition."""
        labels = {}
        for i, participant in enumerate(participants[:2]):
            labels[participant] = "Proposition" if i == 0 else "Opposition"
        return labels

    def get_format_instructions(self) -> str:
        """Oxford format-specific instructions."""
        return """OXFORD DEBATE FORMAT:
            - Maintain formal academic discourse and courtesy throughout
            - Proposition supports the motion, Opposition challenges it
            - Structure arguments clearly: premise, evidence, reasoning, conclusion
            - Address the Chair and audience with appropriate formality
            - Engage substantively with opponent's strongest arguments
            - Use evidence-based reasoning and logical analysis
            - Cross-examination should be strategic and respectful

            Remember: You are participating in a prestigious Oxford Union-style debate."""

    def get_max_participants(self) -> int:
        """Oxford format supports up to 4 participants (2 per side)."""
        return 4

    def get_topic_generation_messages(self) -> List[Dict[str, str]]:
        """Get Oxford specific messages for AI topic generation."""
        return [
            {
                "role": "system",
                "content": "You are an expert at generating Oxford Union-style debate topics. Oxford debates are formal, academic discussions that require sophisticated argumentation and evidence-based reasoning. Topics should be intellectually rigorous and suitable for scholarly debate.",
            },
            {
                "role": "user",
                "content": "Generate a single Oxford Union-style debate topic suitable for formal academic debate. The topic should be phrased as a clear motion that can be argued for or against with scholarly rigor. Make it intellectually challenging and suitable for academic discourse. Respond with just the topic statement, no additional text.",
            },
        ]
