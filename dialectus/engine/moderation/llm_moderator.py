"""OpenAI-compatible LLM moderator for content safety."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from .base_moderator import BaseModerator, ModerationResult
from .exceptions import ModerationProviderError

logger = logging.getLogger(__name__)


class LLMModerator(BaseModerator):
    """Content moderator using any OpenAI-compatible LLM endpoint.

    Works with:
    - Ollama (http://localhost:11434/v1)
    - OpenRouter (https://openrouter.ai/api/v1)
    - Any other OpenAI-compatible API

    Uses instruction-following to classify content safety.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 10.0,
    ):
        """Initialize the LLM moderator.

        Args:
            base_url: OpenAI-compatible API base URL (must end in /v1)
            model: Model name to use for moderation
            api_key: API key (optional, e.g., Ollama doesn't need one)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key or "not-needed",  # Some providers don't require keys
            timeout=timeout,
        )

        # Standard safety categories
        self.safety_categories = [
            "harassment",
            "hate_speech",
            "violence",
            "sexual_content",
            "dangerous_content",
        ]

    async def moderate(self, text: str) -> ModerationResult:
        """Moderate text using an instruction-following LLM.

        Args:
            text: The text to moderate

        Returns:
            ModerationResult with safety determination

        Raises:
            ModerationProviderError: If the API request fails
        """
        # Construct moderation prompt (ShieldGemma-inspired format)
        # This format works well with any instruction-following LLM
        system_prompt = """You are a safety policy expert. Your job is to identify if content violates safety policies, even if the violation is implicit or indirect."""

        user_prompt = f"""<start_of_turn>user
{text}
<end_of_turn>

**Safety Policies:**
- Harassment: Content that harasses, threatens, or bullies individuals
- Hate Speech: Content promoting hatred toward groups based on identity
- Violence: Content promoting, glorifying, or requesting information about harming others
- Sexual Content: Explicit sexual or adult content
- Dangerous Content: Content promoting dangerous, illegal, or discriminatory activities

**Question:** Does the above user input violate any of these safety policies?

**Important:** Content that asks "how to" harm, discriminate, or violate others' rights DOES violate policies, even if phrased as a question.

**Instructions:** Answer ONLY 'Yes' or 'No' first. Answer 'Yes' if the content violates ANY policy above."""

        try:
            # Make API call using OpenAI client
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=50,  # Short response expected
            )

            # Extract response content
            raw_response = response.choices[0].message.content or ""
            raw_response = raw_response.strip()

            logger.debug(
                "Moderation response for topic '%s': %s",
                text[:50],
                raw_response,
            )

            # Parse the response
            is_safe, categories = self._parse_response(raw_response)

            if not is_safe:
                logger.info(
                    "Topic moderation failed - categories: %s",
                    ", ".join(categories),
                )

            return ModerationResult(
                is_safe=is_safe,
                categories=categories,
                confidence=1.0,
                raw_response=raw_response,
            )

        except Exception as exc:
            error_msg = f"Moderation request failed: {type(exc).__name__}: {exc}"
            logger.error(error_msg)
            raise ModerationProviderError("llm", error_msg) from exc

    def _parse_response(self, response: str) -> tuple[bool, list[str]]:
        """Parse LLM response into safety determination.

        Args:
            response: Raw response from the model

        Returns:
            Tuple of (is_safe, categories)
        """
        response_clean = response.strip()
        response_lower = response_clean.lower()

        # Check for Yes/No format (ShieldGemma-style)
        # "Yes" = violates policy = unsafe
        # "No" = doesn't violate = safe
        if response_clean.lower().startswith("yes"):
            # Content violates policy - extract which categories
            detected = [cat for cat in self.safety_categories if cat.replace("_", " ") in response_lower]
            if detected:
                return False, detected
            else:
                return False, ["policy_violation"]

        if response_clean.lower().startswith("no"):
            # Content doesn't violate policy - it's safe
            return True, []

        # Fuzzy matching for models that don't follow format exactly
        safe_indicators = ["safe", "appropriate", "acceptable", "allowed", "does not violate"]
        unsafe_indicators = [
            "unsafe",
            "inappropriate",
            "unacceptable",
            "violation",
            "violates",
            "not allowed",
        ]

        has_safe = any(indicator in response_lower for indicator in safe_indicators)
        has_unsafe = any(indicator in response_lower for indicator in unsafe_indicators)

        # Prioritize unsafe indicators
        if has_unsafe:
            detected_categories = [
                cat for cat in self.safety_categories if cat.replace("_", " ") in response_lower
            ]
            if detected_categories:
                return False, detected_categories
            else:
                return False, ["policy_violation"]

        # Only safe indicators present
        if has_safe and not has_unsafe:
            return True, []

        # Can't parse clearly - fail closed (treat as unsafe)
        logger.warning(
            "Could not parse moderation response: '%s' - treating as unsafe (fail-closed)",
            response[:100],
        )
        return False, ["unknown"]
