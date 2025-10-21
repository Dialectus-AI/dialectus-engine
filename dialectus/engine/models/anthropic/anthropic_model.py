"""Anthropic model definition."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .anthropic_pricing import AnthropicPricing


class AnthropicModel(BaseModel):
    """Anthropic model information (curated/hardcoded for now)."""

    id: str = Field(description="Model identifier")
    name: str = Field(description="Human-readable model name")
    description: str = Field(description="Model description")
    context_length: int = Field(description="Maximum context window in tokens")
    max_output_tokens: int = Field(description="Maximum output tokens per request")
    pricing: AnthropicPricing = Field(description="Pricing information")
    release_date: str = Field(description="Release date (YYYY-MM-DD format)")
    is_latest: bool = Field(
        default=False, description="Whether this is the latest version"
    )
    supports_vision: bool = Field(
        default=False, description="Whether model supports vision input"
    )


# Curated list of Anthropic models with current pricing (as of 2025)
# Source: https://www.anthropic.com/pricing
ANTHROPIC_MODELS: list[AnthropicModel] = [
    AnthropicModel(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet (2024-10-22)",
        description="Most intelligent model - best for complex reasoning, coding, and nuanced analysis",
        context_length=200000,
        max_output_tokens=8192,
        pricing=AnthropicPricing(
            input_cost_per_mtok=3.00,
            output_cost_per_mtok=15.00,
        ),
        release_date="2024-10-22",
        is_latest=True,
        supports_vision=True,
    ),
    AnthropicModel(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku (2024-10-22)",
        description="Fastest model - best for quick responses and high-throughput tasks",
        context_length=200000,
        max_output_tokens=8192,
        pricing=AnthropicPricing(
            input_cost_per_mtok=1.00,
            output_cost_per_mtok=5.00,
        ),
        release_date="2024-10-22",
        is_latest=True,
        supports_vision=True,
    ),
    AnthropicModel(
        id="claude-3-opus-20240229",
        name="Claude 3 Opus (2024-02-29)",
        description="Most capable Claude 3 model - powerful intelligence for complex tasks",
        context_length=200000,
        max_output_tokens=4096,
        pricing=AnthropicPricing(
            input_cost_per_mtok=15.00,
            output_cost_per_mtok=75.00,
        ),
        release_date="2024-02-29",
        is_latest=False,
        supports_vision=True,
    ),
    AnthropicModel(
        id="claude-3-sonnet-20240229",
        name="Claude 3 Sonnet (2024-02-29)",
        description="Balanced Claude 3 model - good performance and speed",
        context_length=200000,
        max_output_tokens=4096,
        pricing=AnthropicPricing(
            input_cost_per_mtok=3.00,
            output_cost_per_mtok=15.00,
        ),
        release_date="2024-02-29",
        is_latest=False,
        supports_vision=True,
    ),
    AnthropicModel(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku (2024-03-07)",
        description="Fastest Claude 3 model - quick and efficient",
        context_length=200000,
        max_output_tokens=4096,
        pricing=AnthropicPricing(
            input_cost_per_mtok=0.25,
            output_cost_per_mtok=1.25,
        ),
        release_date="2024-03-07",
        is_latest=False,
        supports_vision=True,
    ),
]
