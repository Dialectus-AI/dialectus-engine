"""Round execution and management for debates."""

from __future__ import annotations

import logging
import uuid
from typing import Callable, Awaitable, Any, TYPE_CHECKING

from .types import DebatePhase, Position
from .models import DebateContext, DebateMessage
from .response_handler import ResponseHandler

if TYPE_CHECKING:
    from formats import DebateFormat, FormatPhase

logger = logging.getLogger(__name__)


class RoundManager:
    """Manages execution of debate rounds and speaking order."""

    def __init__(self, debate_format: DebateFormat, response_handler: ResponseHandler):
        """Initialize the round manager.

        Args:
            debate_format: The debate format being used
            response_handler: ResponseHandler instance
        """
        self.format = debate_format
        self.response_handler = response_handler

    async def conduct_round(
        self,
        phase: DebatePhase,
        context: DebateContext,
    ) -> list[DebateMessage]:
        """Conduct a single round of the debate.

        Args:
            phase: Current debate phase
            context: The debate context

        Returns:
            List of messages from this round
        """
        context.current_phase = phase
        round_messages = []

        # Determine speaking order based on phase
        speakers = await self._get_speaking_order(phase, context)

        for speaker_id in speakers:
            message = await self.response_handler.get_speaker_response(
                speaker_id, phase, context
            )
            round_messages.append(message)
            context.messages.append(message)

            logger.info(
                f"Round {context.current_round}, {phase.value}: {speaker_id}"
            )

        return round_messages

    async def conduct_format_round(
        self,
        format_phase: FormatPhase,
        context: DebateContext,
    ) -> list[DebateMessage]:
        """Conduct a round using a specific format phase.

        Args:
            format_phase: The format-specific phase
            context: The debate context

        Returns:
            List of messages from this round
        """
        context.current_phase = format_phase.phase
        round_messages = []

        # Use format-defined speaking order
        for speaker_id in format_phase.speaking_order:
            message = await self.response_handler.get_format_speaker_response(
                speaker_id, format_phase, context
            )
            round_messages.append(message)
            context.messages.append(message)

            logger.info(
                f"Round {context.current_round}, {format_phase.name}: {speaker_id}"
            )

        return round_messages

    async def conduct_format_round_stream(
        self,
        format_phase: FormatPhase,
        context: DebateContext,
        message_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        chunk_callback: Callable[[str, bool], Awaitable[None]] | None = None,
    ) -> list[DebateMessage]:
        """Conduct a round with streaming callbacks.

        Args:
            format_phase: The format-specific phase
            context: The debate context
            message_callback: Called when messages start/complete
            chunk_callback: Called for streaming chunks

        Returns:
            List of messages from this round
        """
        context.current_phase = format_phase.phase
        round_messages = []

        # Use format-defined speaking order
        for speaker_id in format_phase.speaking_order:
            message_id = str(uuid.uuid4())

            # Determine position
            model_ids = list(context.participants.keys())
            position_assignments = self.format.get_position_assignments(model_ids)
            speaker_position = position_assignments.get(speaker_id, Position.NEUTRAL)

            # Call message_start callback
            if message_callback:
                await message_callback("message_start", {
                    "message_id": message_id,
                    "speaker_id": speaker_id,
                    "position": speaker_position.value,
                    "phase": format_phase.phase.value,
                    "round_number": context.current_round,
                })

            # Create streaming wrapper that calls chunk_callback
            async def chunk_wrapper(chunk: str, is_complete: bool):
                if chunk_callback:
                    await chunk_callback(chunk, is_complete)

            # Get streaming response
            message = await self.response_handler.get_format_speaker_response_stream(
                speaker_id, format_phase, context, chunk_wrapper, message_id
            )
            round_messages.append(message)
            context.messages.append(message)

            # Call message_complete callback
            if message_callback:
                await message_callback("message_complete", {
                    "message_id": message_id,
                    "speaker_id": message.speaker_id,
                    "position": message.position.value,
                    "phase": message.phase.value,
                    "round_number": message.round_number,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "word_count": len(message.content.split()),
                    "metadata": message.metadata,
                    "cost": message.cost,
                    "generation_id": message.generation_id,
                })

            logger.info(
                f"Round {context.current_round}, {format_phase.name}: {speaker_id}"
            )

        return round_messages

    async def _get_speaking_order(
        self,
        phase: DebatePhase,
        context: DebateContext,
    ) -> list[str]:
        """Determine speaking order for the current phase.

        Args:
            phase: Current debate phase
            context: The debate context

        Returns:
            List of speaker IDs in order
        """
        model_ids = list(context.participants.keys())

        if phase in [DebatePhase.OPENING, DebatePhase.CLOSING]:
            return model_ids  # Pro speaks first, then con
        elif phase == DebatePhase.REBUTTAL:
            return list(reversed(model_ids))  # Con speaks first in rebuttals
        elif phase == DebatePhase.CROSS_EXAM:
            # Alternating questions
            return model_ids * 2  # Each gets to ask and answer

        return model_ids
