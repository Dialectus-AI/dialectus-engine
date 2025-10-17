"""Custom exceptions for the moderation system."""

from __future__ import annotations


class TopicRejectedError(ValueError):
    """Raised when a user-provided topic fails moderation checks.

    This exception should be caught by the API/CLI layer to provide
    user-friendly error messages.
    """

    def __init__(
        self,
        topic: str,
        reason: str,
        categories: list[str] | None = None,
    ):
        """Initialize the exception.

        Args:
            topic: The topic that was rejected
            reason: Human-readable reason for rejection
            categories: List of policy violation categories detected
        """
        self.topic = topic
        self.reason = reason
        self.categories = categories or []

        categories_str = ", ".join(self.categories) if self.categories else "unsafe content"
        message = f"Topic rejected by moderation: {reason} (detected: {categories_str})"
        super().__init__(message)


class ModerationProviderError(RuntimeError):
    """Raised when a moderation provider fails to process a request.

    This indicates an infrastructure failure (Ollama down, timeout, etc.)
    rather than a content policy violation.
    """

    def __init__(self, provider: str, detail: str):
        """Initialize the exception.

        Args:
            provider: The moderation provider that failed
            detail: Details about the failure
        """
        self.provider = provider
        self.detail = detail

        message = f"Moderation provider '{provider}' failed: {detail}"
        super().__init__(message)
