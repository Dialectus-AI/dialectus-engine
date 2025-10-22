"""OpenAI model definition."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .openai_pricing import OpenAIPricing


class OpenAIModel(BaseModel):
    """Describes an OpenAI chat/completions model."""

    id: str = Field(description="Model identifier")
    name: str = Field(description="Human-readable model name")
    description: str = Field(description="Model description for UI display")
    context_length: int = Field(description="Maximum context window in tokens")
    max_output_tokens: int = Field(description="Maximum output tokens per request")
    pricing: OpenAIPricing = Field(description="Pricing information for the model")
    release_date: str = Field(description="Release date (YYYY-MM-DD)")
    is_latest: bool = Field(
        default=False, description="Whether the model is part of the latest lineup"
    )
    supports_vision: bool = Field(
        default=True,
        description="Whether multimodal (vision/audio) inputs are supported",
    )


# Curated selection of OpenAI production models (pricing as of 2024-11)
# Source: https://openai.com/pricing
OPENAI_MODELS: list[OpenAIModel] = [
    OpenAIModel(
        id="gpt-4.1",
        name="GPT-4.1",
        description=(
            "Flagship reasoning model with enhanced multimodal capabilities and strong tool use."
        ),
        context_length=128000,
        max_output_tokens=4096,
        pricing=OpenAIPricing(
            input_cost_per_mtok=10.0,
            output_cost_per_mtok=30.0,
        ),
        release_date="2024-12-18",
        is_latest=True,
        supports_vision=True,
    ),
    OpenAIModel(
        id="gpt-4.1-mini",
        name="GPT-4.1 Mini",
        description=(
            "Cost-efficient GPT-4.1 variant optimised for rapid iterations and long context."
        ),
        context_length=128000,
        max_output_tokens=16384,
        pricing=OpenAIPricing(
            input_cost_per_mtok=0.75,
            output_cost_per_mtok=4.0,
        ),
        release_date="2024-12-18",
        is_latest=True,
        supports_vision=True,
    ),
    OpenAIModel(
        id="gpt-4o",
        name="GPT-4o",
        description=(
            "Balanced flagship model delivering high quality multimodal reasoning and generation."
        ),
        context_length=128000,
        max_output_tokens=16384,
        pricing=OpenAIPricing(
            input_cost_per_mtok=5.0,
            output_cost_per_mtok=15.0,
        ),
        release_date="2024-05-13",
        is_latest=False,
        supports_vision=True,
    ),
    OpenAIModel(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        description=(
            "High-speed, low-cost model for lightweight assistants and realtime applications."
        ),
        context_length=128000,
        max_output_tokens=16384,
        pricing=OpenAIPricing(
            input_cost_per_mtok=0.15,
            output_cost_per_mtok=0.60,
        ),
        release_date="2024-09-12",
        is_latest=True,
        supports_vision=True,
    ),
]
