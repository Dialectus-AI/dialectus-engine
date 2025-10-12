"""Utility functions for the debate engine."""

import asyncio
import logging
from datetime import datetime

from dialectus.engine.models.manager import ModelManager

from .models import DebateMessage

logger = logging.getLogger(__name__)


def calculate_max_tokens(word_limit: int) -> int:
    """Calculate maximum tokens based on word limit with buffer for incomplete
    sentences.

    Args:
        word_limit: Maximum number of words allowed

    Returns:
        Maximum number of tokens with ~100 word buffer (1 token ≈ 0.75 words)
    """
    # Base conversion: word_limit * 1.33 (since 1 token ≈ 0.75 words)
    # Add buffer: ~100 words = 133 tokens
    base_tokens = int(word_limit * 1.33)
    buffer_tokens = 133  # ~100 words of buffer for sentence completion
    return base_tokens + buffer_tokens


def trim_incomplete_sentence(text: str) -> str:
    """Remove trailing incomplete sentence if response doesn't end with punctuation.

    Args:
        text: The response text to check

    Returns:
        Text with incomplete sentence removed, or original text if already complete
    """
    text = text.rstrip()

    if not text:
        return text

    # Check if ends with sentence-ending punctuation
    if text[-1] not in ".!?":
        # Find last complete sentence
        last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))

        if last_period > 0:
            # Keep everything up to and including the last sentence
            trimmed = text[: last_period + 1].rstrip()
            logger.info(
                f"Trimmed incomplete sentence: removed {len(text) - len(trimmed)}"
                " characters"
            )
            return trimmed
        else:
            # No sentence-ending punctuation found at all
            logger.warning(
                "Response has no sentence-ending punctuation - returning as-is"
            )
            return text

    return text


async def query_and_update_cost(
    message: DebateMessage,
    speaker_id: str,
    model_manager: ModelManager,
) -> None:
    """Background task to query and update cost for a message with generation_id.

    Args:
        message: The debate message to update
        speaker_id: ID of the speaker/model
        model_manager: ModelManager instance for querying cost
    """
    if not message.generation_id:
        return

    try:
        # Wait a bit to ensure the generation is finalized on OpenRouter's end
        await asyncio.sleep(2.0)

        cost = await model_manager.query_generation_cost(
            speaker_id, message.generation_id
        )
        if cost is not None:
            # Update the message object in memory
            message.cost = cost
            message.cost_queried_at = datetime.now()
            logger.info(f"Updated cost for message {message.generation_id}: ${cost}")
        else:
            logger.warning(
                f"Failed to retrieve cost for generation {message.generation_id}"
            )

    except Exception as e:
        logger.error(
            f"Failed to query cost for generation {message.generation_id}: {e}"
        )
