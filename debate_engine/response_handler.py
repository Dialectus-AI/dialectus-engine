"""Response generation and processing for debate participants."""

import asyncio
import logging
import re
import time
from typing import Callable, Awaitable

from config.settings import AppConfig
from models.manager import ModelManager
from formats import DebateFormat, FormatPhase
from .types import DebatePhase, Position
from .models import DebateContext, DebateMessage
from .context_builder import ContextBuilder
from .utils import calculate_max_tokens, query_and_update_cost

logger = logging.getLogger(__name__)


class ResponseHandler:
    """Handles generation and processing of model responses during debates."""

    def __init__(
        self,
        config: AppConfig,
        model_manager: ModelManager,
        debate_format: DebateFormat,
        context_builder: ContextBuilder,
    ):
        """Initialize the response handler.

        Args:
            config: Application configuration
            model_manager: ModelManager instance
            debate_format: The debate format being used
            context_builder: ContextBuilder instance
        """
        self.config = config
        self.model_manager = model_manager
        self.format = debate_format
        self.context_builder = context_builder

    async def get_speaker_response(
        self,
        speaker_id: str,
        phase: DebatePhase,
        context: DebateContext,
    ) -> DebateMessage:
        """Get a response from a specific speaker.

        Args:
            speaker_id: ID of the speaking model
            phase: Current debate phase
            context: The debate context

        Returns:
            DebateMessage with the speaker's response
        """
        # Determine position
        positions = [Position.PRO, Position.CON]
        model_ids = list(context.participants.keys())
        speaker_position = (
            positions[model_ids.index(speaker_id)]
            if speaker_id in model_ids[:2]
            else Position.NEUTRAL
        )

        # Build conversation context
        messages = self.context_builder.build_conversation_context(speaker_id, phase, context)

        # Generate response with timing
        start_time = time.time()
        async with self.model_manager.model_session(speaker_id):
            response_content = await self.model_manager.generate_response(
                speaker_id,
                messages,
                max_tokens=min(
                    self.config.models[speaker_id].max_tokens,
                    calculate_max_tokens(self.config.debate.word_limit),
                ),
            )
        generation_time = time.time() - start_time

        # Clean the response to remove any echoed prefixes
        cleaned_response = self.clean_model_response(response_content, speaker_id, context)

        return DebateMessage(
            speaker_id=speaker_id,
            position=speaker_position,
            phase=phase,
            round_number=context.current_round,
            content=cleaned_response,
            metadata={"generation_time_ms": int(generation_time * 1000)},
        )

    async def get_format_speaker_response(
        self,
        speaker_id: str,
        format_phase: FormatPhase,
        context: DebateContext,
    ) -> DebateMessage:
        """Get a response from a specific speaker using format phase.

        Args:
            speaker_id: ID of the speaking model
            format_phase: The format-specific phase
            context: The debate context

        Returns:
            DebateMessage with the speaker's response
        """
        # Determine position
        model_ids = list(context.participants.keys())
        position_assignments = self.format.get_position_assignments(model_ids)
        speaker_position = position_assignments.get(speaker_id, Position.NEUTRAL)

        # Build conversation context with format-specific instruction
        messages = self.context_builder.build_format_conversation_context(
            speaker_id, format_phase, context
        )

        # Calculate max tokens with format time multiplier
        base_max_tokens = min(
            self.config.models[speaker_id].max_tokens,
            calculate_max_tokens(self.config.debate.word_limit),
        )
        adjusted_max_tokens = int(base_max_tokens * format_phase.time_multiplier)

        # Generate response with timing and detailed error handling
        start_time = time.time()
        try:
            logger.info(
                f"RESPONSE HANDLER: Starting response generation for {speaker_id} "
                f"({self.config.models[speaker_id].name}) via {self.config.models[speaker_id].provider}"
            )
            async with self.model_manager.model_session(speaker_id):
                response_content = await self.model_manager.generate_response(
                    speaker_id, messages, max_tokens=adjusted_max_tokens
                )
            generation_time = time.time() - start_time
            logger.info(
                f"RESPONSE HANDLER: Successfully generated {len(response_content)} chars "
                f"for {speaker_id} in {generation_time:.2f}s"
            )
        except Exception as e:
            generation_time = time.time() - start_time
            model_config = self.config.models[speaker_id]
            error_msg = (
                f"RESPONSE HANDLER FAILURE: {speaker_id} ({model_config.name}) "
                f"via {model_config.provider} failed after {generation_time:.2f}s: "
                f"{type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Clean the response to remove any echoed prefixes
        cleaned_response = self.clean_model_response(response_content, speaker_id, context)

        return DebateMessage(
            speaker_id=speaker_id,
            position=speaker_position,
            phase=format_phase.phase,
            round_number=context.current_round,
            content=cleaned_response,
            metadata={"generation_time_ms": int(generation_time * 1000)},
        )

    async def get_format_speaker_response_stream(
        self,
        speaker_id: str,
        format_phase: FormatPhase,
        context: DebateContext,
        chunk_callback: Callable[[str, bool], Awaitable[None]],
        message_id: str | None = None,
    ) -> DebateMessage:
        """Get a streaming response from a specific speaker using format phase.

        Args:
            speaker_id: ID of the speaking model
            format_phase: The format-specific phase
            context: The debate context
            chunk_callback: Callback for streaming chunks
            message_id: Optional message ID for tracking

        Returns:
            DebateMessage with the speaker's response
        """
        # Determine position
        model_ids = list(context.participants.keys())
        position_assignments = self.format.get_position_assignments(model_ids)
        speaker_position = position_assignments.get(speaker_id, Position.NEUTRAL)

        # Build conversation context with format-specific instruction
        messages = self.context_builder.build_format_conversation_context(
            speaker_id, format_phase, context
        )

        # Calculate max tokens with format time multiplier
        base_max_tokens = min(
            self.config.models[speaker_id].max_tokens,
            calculate_max_tokens(self.config.debate.word_limit),
        )
        adjusted_max_tokens = int(base_max_tokens * format_phase.time_multiplier)

        # Generate streaming response with metadata for cost tracking
        try:
            logger.info(
                f"RESPONSE HANDLER: Starting streaming response generation for {speaker_id} "
                f"({self.config.models[speaker_id].name}) via {self.config.models[speaker_id].provider}"
            )
            async with self.model_manager.model_session(speaker_id):
                generation_metadata = await self.model_manager.generate_response_stream_with_metadata(
                    speaker_id, messages, chunk_callback, max_tokens=adjusted_max_tokens
                )
            logger.info(
                f"RESPONSE HANDLER: Successfully generated {len(generation_metadata.content)} chars "
                f"via streaming for {speaker_id}, generation_id: {generation_metadata.generation_id}"
            )
        except Exception as e:
            model_config = self.config.models[speaker_id]
            error_msg = (
                f"RESPONSE HANDLER STREAMING FAILURE: {speaker_id} ({model_config.name}) "
                f"via {model_config.provider} failed: {type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Clean the response to remove any echoed prefixes
        cleaned_response = self.clean_model_response(generation_metadata.content, speaker_id, context)

        # Create message with cost tracking fields
        debate_message = DebateMessage(
            speaker_id=speaker_id,
            position=speaker_position,
            phase=format_phase.phase,
            round_number=context.current_round,
            content=cleaned_response,
            metadata={"generation_time_ms": generation_metadata.generation_time_ms or 0},
            message_id=message_id,
            cost=generation_metadata.cost,
            generation_id=generation_metadata.generation_id,
        )

        # Schedule background cost query for OpenRouter models
        if generation_metadata.generation_id and generation_metadata.provider == "openrouter":
            asyncio.create_task(
                query_and_update_cost(debate_message, speaker_id, self.model_manager)
            )

        return debate_message

    def clean_model_response(self, response: str, speaker_id: str, context: DebateContext) -> str:
        """Clean model response by removing any echoed prefixes or formatting.

        Args:
            response: Raw response from the model
            speaker_id: ID of the speaking model
            context: The debate context

        Returns:
            Cleaned response text
        """
        # Get the model name for this speaker
        display_name = speaker_id
        if speaker_id in context.participants:
            display_name = context.participants[speaker_id].name

        # Debug logging for empty responses
        if not response.strip():
            logger.warning(
                f"Model {display_name} ({speaker_id}) returned empty response before cleaning"
            )

        cleaned = response.strip()

        # Remove position statement prefixes like "Proposition Opening Statement", etc.
        position_prefixes = [
            r"^(?:proposition|opposition|pro|con)\s+(?:opening|rebuttal|closing)\s+(?:statement|argument)[\:\-\s]*",
            r"^(?:opening|rebuttal|closing)\s+(?:statement|argument)[\:\-\s]*",
            r"^(?:proposition|opposition|pro|con)[\:\-\s]+",
            r"^\*\*(?:proposition|opposition|pro|con)\s+(?:opening|rebuttal|closing)\s+(?:statement|argument)\*\*[\:\-\s]*",  # **Position Statement**
            r"^(?:Phase\s+\d+[\:\-\s]*)?(?:proposition|opposition|pro|con)\s+(?:opening|rebuttal|closing)[\:\-\s]*",  # Phase 1: Proposition Opening
        ]

        for pattern in position_prefixes:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Remove any echoed conversation prefixes that the model might repeat
        conversation_patterns = [
            rf"\[{re.escape(speaker_id)}\s*-\s*\w+\]:\s*",  # [model_a - pro]:
            rf"\[{re.escape(display_name)}\s*-\s*\w+\]:\s*",  # [llama3.2:latest - pro]:
            r"\[.*?\s*-\s*\w+\]:\s*",  # Any [something - position]: pattern
        ]

        for pattern in conversation_patterns:
            # Keep removing the pattern until no more matches (handles multiple prefixes)
            while re.match(pattern, cleaned, re.IGNORECASE):
                cleaned = re.sub(
                    pattern, "", cleaned, count=1, flags=re.IGNORECASE
                ).strip()

        # Remove markdown formatting
        # Remove headers (# ## ###)
        cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)

        # Remove bold and italic markdown (**text**, *text*)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)  # **bold** -> bold
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)  # *italic* -> italic

        # Remove any remaining asterisks that might be formatting attempts
        cleaned = re.sub(
            r"^\*+\s*", "", cleaned, flags=re.MULTILINE
        )  # Remove lines starting with *

        # Debug logging if cleaning resulted in empty response
        final_cleaned = cleaned.strip()
        if response.strip() and not final_cleaned:
            logger.warning(
                f"Model {display_name} ({speaker_id}) response was cleaned to empty. "
                f"Original: {repr(response)}, Cleaned: {repr(final_cleaned)}"
            )

        return final_cleaned
