"""Prompt building and generation for debate participants."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.settings import AppConfig
from .types import Position, DebatePhase
from .models import DebateContext

if TYPE_CHECKING:
    from formats import DebateFormat

logger = logging.getLogger(__name__)

# Available personality types for debate participants
PERSONALITY_TYPES = [
    "analytical",
    "passionate",
    "scholarly",
    "practical",
    "neutral",
]


class PromptBuilder:
    """Handles generation of system prompts and instructions for debate participants."""

    def __init__(self, config: AppConfig, debate_format: DebateFormat):
        """Initialize the prompt builder.

        Args:
            config: Application configuration
            debate_format: The debate format being used
        """
        self.config = config
        self.format = debate_format
        self.system_prompts: dict[str, str] = {}

    async def generate_system_prompts(self, context: DebateContext) -> dict[str, str]:
        """Generate system prompts for each participant.

        Args:
            context: The debate context

        Returns:
            Dictionary mapping model_id to system prompt
        """
        base_prompt = self._get_base_system_prompt(context)

        # Use format-specific position assignments
        model_ids = list(context.participants.keys())
        position_assignments = self.format.get_position_assignments(model_ids)

        for model_id in model_ids:
            position = position_assignments.get(model_id, Position.NEUTRAL)
            personality = context.participants[model_id].personality

            prompt = self._create_participant_prompt(
                base_prompt, position, personality, model_id, context
            )
            self.system_prompts[model_id] = prompt

            logger.debug(f"Generated system prompt for {model_id} ({position.value})")

        return self.system_prompts

    def _get_base_system_prompt(self, context: DebateContext) -> str:
        """Get the base system prompt template.

        Args:
            context: The debate context

        Returns:
            Base system prompt text
        """
        format_instructions = self.format.get_format_instructions()

        return f"""You are participating in a formal debate about: "{context.topic}"

DEBATE FORMAT: {self.format.name.upper()}
WORD LIMIT: {self.config.debate.word_limit} words or less per response

{format_instructions}

GENERAL RULES:
1. Stay focused on the topic
2. Keep your response under {self.config.debate.word_limit} words
3. Provide evidence and reasoning
4. Address opponent's arguments
5. Maintain a respectful tone
6. Follow the debate format structure

RESPONSE FORMAT:
- Speak directly as your assigned role without any labels, prefixes, or announcements
- Use plain text without markdown formatting (avoid **bold**, *italics*, # headers, bullet points)
- Write in natural conversational style as if speaking live to an audience
- Do not add labels like "COUNTER:", "STATEMENT:", "OPENING:", etc. - just speak your argument
- Focus on content, not formatting or structure

You will be told your specific role and speaking context for each response."""

    def _create_participant_prompt(
        self,
        base_prompt: str,
        position: Position,
        personality: str,
        model_id: str,
        context: DebateContext,
    ) -> str:
        """Create a participant-specific system prompt using role-based identity.

        Args:
            base_prompt: The base system prompt
            position: The participant's position (PRO/CON/NEUTRAL)
            personality: The participant's personality trait
            model_id: The model identifier
            context: The debate context

        Returns:
            Participant-specific system prompt
        """
        # Get the proper role name for this format (Opposition, Proposition, etc.)
        model_ids = list(context.participants.keys())
        side_labels = self.format.get_side_labels(model_ids)
        role_name = side_labels.get(model_id, position.value.title())

        position_guidance = {
            Position.PRO: f"You ARE the {role_name} speaker. You support the motion and believe it is correct.",
            Position.CON: f"You ARE the {role_name} speaker. You oppose the motion and believe it is wrong.",
        }

        personality_guidance = {
            "analytical": "Your speaking style is data-driven, logical, and systematic.",
            "passionate": "Your speaking style is persuasive and emotionally engaging while factual.",
            "scholarly": "Your speaking style references academic sources and theoretical frameworks.",
            "practical": "Your speaking style emphasizes real-world applications and practical implications.",
            "neutral": "Your speaking style is balanced with measured reasoning.",
        }

        role_instruction = position_guidance.get(
            position, f"You ARE the {role_name} speaker."
        )
        style_instruction = personality_guidance.get(
            personality, "Speak in your natural style."
        )

        return f"""{base_prompt}

YOUR ROLE: {role_instruction}
You are not describing what {role_name} would say - you ARE {role_name} speaking directly.
Speak as if you are standing at the podium addressing the audience.
Do not announce your role, add labels, or use prefixes like "COUNTER:" or "STATEMENT:" - simply speak your argument.

YOUR SPEAKING STYLE: {style_instruction}

Remember: You are embodying the {role_name} position throughout this debate. Speak naturally and directly as that person would speak."""

    @staticmethod
    def get_phase_instruction(phase: DebatePhase) -> str:
        """Get instruction text for the current phase.

        Args:
            phase: The current debate phase

        Returns:
            Phase-specific instruction
        """
        instructions = {
            DebatePhase.OPENING: "Present your opening arguments clearly and persuasively.",
            DebatePhase.REBUTTAL: "Address the opponent's arguments and strengthen your position.",
            DebatePhase.CROSS_EXAM: "Ask probing questions or provide direct answers.",
            DebatePhase.CLOSING: "Summarize your case and make your final persuasive appeal.",
        }
        return instructions.get(phase, "Participate according to the debate format.")
