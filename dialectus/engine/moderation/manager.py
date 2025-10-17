"""Moderation manager for orchestrating content safety checks."""

from __future__ import annotations

import logging
import os

from dialectus.engine.config.settings import ModerationConfig, SystemConfig

from .base_moderator import BaseModerator, ModerationResult
from .exceptions import ModerationProviderError, TopicRejectedError
from .llm_moderator import LLMModerator
from .openai_moderator import OpenAIModerator

logger = logging.getLogger(__name__)


class ModerationManager:
    """Manages content moderation for user-provided topics.

    The manager coordinates moderation checks using configured providers
    and enforces content safety policies.
    """

    def __init__(
        self, moderation_config: ModerationConfig, system_config: SystemConfig
    ):
        """Initialize the moderation manager.

        Args:
            moderation_config: Moderation configuration
            system_config: System configuration for provider settings
        """
        self.config = moderation_config
        self.system_config = system_config
        self._moderator: BaseModerator | None = None

        if self.config.enabled:
            self._initialize_moderator()

    def _initialize_moderator(self) -> None:
        """Initialize the moderator based on configuration.

        Raises:
            ModerationProviderError: If provider initialization fails
        """
        provider = self.config.provider.lower()

        if provider == "ollama":
            base_url = (
                self.config.base_url or f"{self.system_config.ollama_base_url}/v1"
            )
            api_key = None

            logger.info(
                "Initializing moderation: %s at %s (timeout: %ss)",
                self.config.model,
                base_url,
                self.config.timeout,
            )

            self._moderator = LLMModerator(
                base_url=base_url,
                model=self.config.model,
                api_key=api_key,
                timeout=self.config.timeout,
            )
            return

        if provider == "openrouter":
            base_url = self.config.base_url or self.system_config.openrouter.base_url
            api_key = self.config.api_key or self.system_config.openrouter.api_key

            if not api_key:
                raise ModerationProviderError(
                    provider,
                    "OpenRouter requires an API key in config or OPENROUTER_API_KEY env var",
                )

            logger.info(
                "Initializing moderation: %s at %s (timeout: %ss)",
                self.config.model,
                base_url,
                self.config.timeout,
            )

            self._moderator = LLMModerator(
                base_url=base_url,
                model=self.config.model,
                api_key=api_key,
                timeout=self.config.timeout,
            )
            return

        if provider == "openai":
            base_url = self.config.base_url or "https://api.openai.com/v1"
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ModerationProviderError(
                    provider,
                    "OpenAI moderation requires an API key via config or OPENAI_API_KEY env var",
                )

            logger.info(
                "Initializing moderation: %s at %s (timeout: %ss)",
                self.config.model,
                base_url,
                self.config.timeout,
            )

            self._moderator = OpenAIModerator(
                api_key=api_key,
                model=self.config.model,
                timeout=self.config.timeout,
                base_url=base_url,
            )
            return

        # Custom provider - user must supply base_url
        if not self.config.base_url:
            raise ModerationProviderError(
                provider,
                f"Custom provider '{provider}' requires base_url in moderation config",
            )

        base_url = self.config.base_url
        api_key = self.config.api_key

        logger.info(
            "Initializing moderation: %s at %s (timeout: %ss)",
            self.config.model,
            base_url,
            self.config.timeout,
        )

        self._moderator = LLMModerator(
            base_url=base_url,
            model=self.config.model,
            api_key=api_key,
            timeout=self.config.timeout,
        )

    @property
    def enabled(self) -> bool:
        """Check if moderation is enabled."""
        return self.config.enabled

    async def moderate_topic(self, topic: str) -> ModerationResult:
        """Moderate a user-provided topic.

        Args:
            topic: The topic text to moderate

        Returns:
            ModerationResult with safety determination

        Raises:
            TopicRejectedError: If the topic fails moderation
            ModerationProviderError: If the moderation request fails
        """
        if not self.enabled:
            logger.debug("Moderation disabled, skipping check for topic")
            return ModerationResult(is_safe=True, categories=[])

        if not self._moderator:
            raise ModerationProviderError(
                self.config.provider, "Moderator not initialized"
            )

        logger.info(
            "Moderating topic with %s/%s",
            self.config.provider,
            self.config.model,
        )

        # Perform moderation check
        result = await self._moderator.moderate(topic)

        if result.is_safe:
            logger.info("Topic passed moderation")
            return result
        else:
            # Topic failed moderation - raise exception for caller to handle
            categories_str = ", ".join(result.categories)
            reason = f"Topic violates content policy: {categories_str}"

            logger.info("Topic rejected by moderation: %s", reason)

            raise TopicRejectedError(
                topic=topic,
                reason=reason,
                categories=result.categories,
            )
