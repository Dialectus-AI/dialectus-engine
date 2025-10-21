"""Anthropic pricing model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnthropicPricing(BaseModel):
    """Pricing information for an Anthropic model."""

    input_cost_per_mtok: float = Field(
        description="Cost per million input tokens in USD"
    )
    output_cost_per_mtok: float = Field(
        description="Cost per million output tokens in USD"
    )

    @property
    def prompt_cost_per_1k(self) -> float:
        """Convert to cost per 1K tokens for consistency with base types."""
        return self.input_cost_per_mtok / 1000

    @property
    def completion_cost_per_1k(self) -> float:
        """Convert to cost per 1K tokens for consistency with base types."""
        return self.output_cost_per_mtok / 1000

    @property
    def is_free(self) -> bool:
        """Whether this model is free to use."""
        return self.input_cost_per_mtok == 0 and self.output_cost_per_mtok == 0
