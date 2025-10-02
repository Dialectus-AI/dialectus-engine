"""Utility functions for the debate engine."""

import asyncio
import logging
from datetime import datetime

from models.manager import ModelManager
from .models import DebateMessage

logger = logging.getLogger(__name__)


def calculate_max_tokens(word_limit: int) -> int:
    """Calculate maximum tokens based on word limit.

    Args:
        word_limit: Maximum number of words allowed

    Returns:
        Maximum number of tokens (rough estimate: 1 token ≈ 0.75 words)
    """
    return int(word_limit * 1.33)


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

        cost = await model_manager.query_generation_cost(speaker_id, message.generation_id)
        if cost is not None:
            # Update the message object in memory
            # The web API will save this when it saves the transcript
            message.cost = cost
            message.cost_queried_at = datetime.now()
            logger.info(f"Updated cost for message {message.generation_id}: ${cost}")
        else:
            logger.warning(f"Failed to retrieve cost for generation {message.generation_id}")

    except Exception as e:
        logger.error(f"Failed to query cost for generation {message.generation_id}: {e}")
