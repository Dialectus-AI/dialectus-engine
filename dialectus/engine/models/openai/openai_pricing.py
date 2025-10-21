"""OpenAI pricing model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OpenAIPricing(BaseModel):
    """Pricing information for an OpenAI model."""

    input_cost_per_mtok: float = Field(
        description="Cost per million input tokens in USD"
    )
    output_cost_per_mtok: float = Field(
        description="Cost per million output tokens in USD"
    )

    @property
    def prompt_cost_per_1k(self) -> float:
        """Cost per 1K input tokens for compatibility with base model pricing."""
        return self.input_cost_per_mtok / 1000

    @property
    def completion_cost_per_1k(self) -> float:
        """Cost per 1K output tokens for compatibility with base model pricing."""
        return self.output_cost_per_mtok / 1000

    @property
    def is_free(self) -> bool:
        """Whether this model is free to use."""
        return self.input_cost_per_mtok == 0 and self.output_cost_per_mtok == 0
