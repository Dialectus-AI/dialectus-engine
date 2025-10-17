"""Base classes for content moderation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ModerationResult(BaseModel):
    """Result from a moderation check.

    Attributes:
        is_safe: Whether the content passed moderation
        categories: List of detected policy violation categories (empty if safe)
        confidence: Confidence score from 0.0 to 1.0
        raw_response: Optional raw response from the moderation model
    """

    is_safe: bool = Field(..., description="Whether the content is safe")
    categories: list[str] = Field(
        default_factory=list, description="Policy violation categories detected"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    raw_response: str | None = Field(
        default=None, description="Raw response from moderation model"
    )


class BaseModerator(ABC):
    """Abstract base class for content moderators.

    Moderators analyze text content and determine if it violates
    content policies or safety guidelines.
    """

    @abstractmethod
    async def moderate(self, text: str) -> ModerationResult:
        """Moderate the given text content.

        Args:
            text: The text content to moderate

        Returns:
            ModerationResult with safety determination

        Raises:
            ModerationProviderError: If the moderation request fails
        """
        pass
