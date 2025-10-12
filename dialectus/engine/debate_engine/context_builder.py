"""Conversation context building for debate participants."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialectus.engine.config.settings import AppConfig

from .models import DebateContext
from .prompt_builder import PromptBuilder
from .types import DebatePhase

if TYPE_CHECKING:
    from dialectus.engine.formats import DebateFormat, FormatPhase


class ContextBuilder:
    """Builds conversation context for model prompts during debates."""

    def __init__(
        self,
        config: AppConfig,
        debate_format: DebateFormat,
        prompt_builder: PromptBuilder,
    ):
        """Initialize the context builder.

        Args:
            config: Application configuration
            debate_format: The debate format being used
            prompt_builder: PromptBuilder instance for accessing system prompts
        """
        self.config = config
        self.format = debate_format
        self.prompt_builder = prompt_builder

    def build_conversation_context(
        self,
        speaker_id: str,
        phase: DebatePhase,
        context: DebateContext,
    ) -> list[dict[str, str]]:
        """Build conversation context for the model.

        Args:
            speaker_id: ID of the speaking model
            phase: Current debate phase
            context: The debate context

        Returns:
            List of message dictionaries for the model API
        """
        messages: list[dict[str, str]] = []

        # Add system prompt
        if speaker_id in self.prompt_builder.system_prompts:
            messages.append(
                {
                    "role": "system",
                    "content": self.prompt_builder.system_prompts[speaker_id],
                }
            )

        # Add role-focused phase instruction
        model_ids = list(context.participants.keys())
        side_labels = self.format.get_side_labels(model_ids)
        role_name = side_labels.get(speaker_id, "Speaker")

        phase_instruction = PromptBuilder.get_phase_instruction(phase)
        messages.append(
            {
                "role": "system",
                "content": f"You are now speaking as {role_name}. {phase_instruction}",
            }
        )

        # Add relevant debate history
        for msg in context.messages[-6:]:  # Last 6 messages for context
            role = "assistant" if msg.speaker_id == speaker_id else "user"

            # Use a cleaner format that's less likely to be repeated by small models
            if role == "assistant":
                # When showing the model its own previous messages, just show the content
                messages.append({"role": role, "content": msg.content})
            else:
                # When showing opponent messages, use a simple format
                messages.append(
                    {
                        "role": role,
                        "content": f"Opponent ({msg.position.value}): {msg.content}",
                    }
                )

        # Add current turn prompt with role focus
        turn_prompt = f"Now speak as {role_name}. Stay under {self.config.debate.word_limit} words."
        messages.append({"role": "user", "content": turn_prompt})

        return messages

    def build_format_conversation_context(
        self,
        speaker_id: str,
        format_phase: FormatPhase,
        context: DebateContext,
    ) -> list[dict[str, str]]:
        """Build conversation context for the model using format phase.

        Args:
            speaker_id: ID of the speaking model
            format_phase: The format-specific phase
            context: The debate context

        Returns:
            List of message dictionaries for the model API
        """
        messages: list[dict[str, str]] = []

        # Add system prompt
        if speaker_id in self.prompt_builder.system_prompts:
            messages.append(
                {
                    "role": "system",
                    "content": self.prompt_builder.system_prompts[speaker_id],
                }
            )

        # Add role-focused phase instruction instead of procedural description
        model_ids = list(context.participants.keys())
        side_labels = self.format.get_side_labels(model_ids)
        role_name = side_labels.get(speaker_id, "Speaker")

        # Simplify the instruction to focus on role, not procedure
        simplified_instruction = (
            format_phase.instruction.replace("As the Proposition,", "")
            .replace("As the Opposition,", "")
            .strip()
        )

        messages.append(
            {
                "role": "system",
                "content": f"You are now speaking as {role_name}. {simplified_instruction}",
            }
        )

        # Add relevant debate history
        for msg in context.messages[-6:]:  # Last 6 messages for context
            role = "assistant" if msg.speaker_id == speaker_id else "user"

            # Use a cleaner format that's less likely to be repeated by small models
            if role == "assistant":
                # When showing the model its own previous messages, just show the content
                messages.append({"role": role, "content": msg.content})
            else:
                # When showing opponent messages, use a simple format
                messages.append(
                    {
                        "role": role,
                        "content": f"Opponent ({msg.position.value}): {msg.content}",
                    }
                )

        # Add current turn prompt with role focus
        word_limit = int(self.config.debate.word_limit * format_phase.time_multiplier)
        turn_prompt = f"Now speak as {role_name}. Stay under {word_limit} words."
        messages.append({"role": "user", "content": turn_prompt})

        return messages
