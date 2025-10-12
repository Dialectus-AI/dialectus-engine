"""Custom exceptions for model providers."""

from __future__ import annotations


class ProviderRateLimitError(RuntimeError):
    """Raised when an upstream provider rejects a request with a rate limit response."""

    def __init__(
        self,
        provider: str,
        model: str | None,
        status_code: int,
        detail: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.detail = detail

        base_message = (
            f"{provider.capitalize()} returned HTTP {status_code} (rate limited)."
        )
        if detail:
            base_message = f"{base_message} {detail}"

        super().__init__(base_message)

    @property
    def is_free_tier_model(self) -> bool:
        """Indicate whether the model name hints at a free-tier OpenRouter route."""
        return bool(self.model and ":free" in self.model)
