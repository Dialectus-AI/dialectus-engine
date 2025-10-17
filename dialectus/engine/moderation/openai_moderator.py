"""OpenAI native moderation provider implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from openai import AsyncOpenAI, RateLimitError

from .base_moderator import BaseModerator, ModerationResult
from .exceptions import ModerationProviderError

logger = logging.getLogger(__name__)


_CATEGORY_MAPPING = {
    "harassment": "harassment",
    "harassment/threatening": "harassment",
    "hate": "hate_speech",
    "hate/threatening": "hate_speech",
    "violence": "violence",
    "violence/graphic": "violence",
    "sexual": "sexual_content",
    "sexual/minors": "sexual_content",
    "illicit": "dangerous_content",
    "illicit/violent": "dangerous_content",
    "self-harm": "dangerous_content",
    "self-harm/instructions": "dangerous_content",
    "self-harm/intent": "dangerous_content",
}


def _map_categories(category_names: Iterable[str]) -> list[str]:
    """Map OpenAI moderation categories into the engine taxonomy."""
    mapped = {
        _CATEGORY_MAPPING.get(name, "policy_violation") for name in category_names
    }
    return sorted(mapped)


class OpenAIModerator(BaseModerator):
    """Moderator that uses OpenAI's dedicated moderation endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 10.0,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI moderator requires an API key")

        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_base_delay = max(0.1, retry_base_delay)

        self.client = client or AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            base_url=base_url.rstrip("/") if base_url else None,
        )

    async def _call_moderation_endpoint(self, text: str):
        """Invoke the OpenAI moderation endpoint."""
        return await self.client.moderations.create(
            model=self.model,
            input=text,
        )

    async def moderate(self, text: str) -> ModerationResult:
        """Call OpenAI's moderation endpoint and normalise the result."""
        attempt = 0

        while True:
            try:
                response = await self._call_moderation_endpoint(text)
                break
            except RateLimitError as exc:
                attempt += 1
                if attempt > self.max_retries:
                    message = f"OpenAI moderation request failed: RateLimitError: {exc}"
                    logger.error(message)
                    raise ModerationProviderError("openai", message) from exc

                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI moderation rate limited (attempt %s/%s). Retrying in %.2fs",
                    attempt,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
            except (
                Exception
            ) as exc:  # pragma: no cover - network errors mocked in tests
                message = (
                    f"OpenAI moderation request failed: {type(exc).__name__}: {exc}"
                )
                logger.error(message)
                raise ModerationProviderError("openai", message) from exc

        results = getattr(response, "results", None)
        if not results:
            raise ModerationProviderError(
                "openai",
                "Moderation response contained no results",
            )

        result = results[0]
        flagged = bool(getattr(result, "flagged", False))
        categories_dict = getattr(result, "categories", {}) or {}
        category_scores = getattr(result, "category_scores", {}) or {}

        flagged_categories = [
            name for name, is_flagged in categories_dict.items() if is_flagged
        ]
        mapped_categories = _map_categories(flagged_categories)

        if flagged and not mapped_categories:
            mapped_categories = ["policy_violation"]

        max_score = max(category_scores.values(), default=0.0)
        confidence = max_score if flagged else max(0.0, 1.0 - max_score)

        raw_response: str
        if hasattr(response, "model_dump_json"):
            try:
                raw_response = response.model_dump_json(indent=2)
            except TypeError:
                raw_response = response.model_dump_json()
        else:
            raw_response = str(response)

        if flagged:
            logger.info(
                "OpenAI moderation flagged content with categories: %s",
                ", ".join(mapped_categories),
            )

        return ModerationResult(
            is_safe=not flagged,
            categories=mapped_categories,
            confidence=confidence,
            raw_response=raw_response,
        )
