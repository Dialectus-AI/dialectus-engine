"""
Base types for model providers.
Shared types and enums that work across all providers (Ollama, OpenRouter, future direct APIs).
"""

from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel, Field

SourceInfo = Mapping[str, object]


class ModelWeightClass(Enum):
    """Tournament-style weight classes for models based on capability and cost."""

    LIGHTWEIGHT = "lightweight"  # Small, fast models (< 10B params equivalent)
    MIDDLEWEIGHT = "middleweight"  # Mid-range models (10B-70B params equivalent)
    HEAVYWEIGHT = "heavyweight"  # Large models (70B+ params equivalent)
    ULTRAWEIGHT = "ultraweight"  # Frontier models (GPT-4, Claude-3 Opus, etc.)
    PREVIEW = "preview"  # Anonymous/preview models (unstable)


class ModelTier(Enum):
    """Quality tiers for models."""

    BUDGET = "budget"  # Cheapest options
    BALANCED = "balanced"  # Best value proposition
    PREMIUM = "premium"  # High quality, reasonable cost
    FLAGSHIP = "flagship"  # Top tier, cost no object


class ModelPricing(BaseModel):
    """Generic pricing information for any model."""

    prompt_cost_per_1k: float = 0.0
    completion_cost_per_1k: float = 0.0
    is_free: bool = True
    currency: str = "USD"

    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1K tokens (assuming 50/50 prompt/completion)."""
        return (self.prompt_cost_per_1k + self.completion_cost_per_1k) / 2


class BaseEnhancedModelInfo(BaseModel):
    """Base enhanced model information that works across all providers."""

    id: str
    name: str
    provider: str  # "ollama", "openrouter", "openai", etc.
    description: str
    weight_class: ModelWeightClass
    tier: ModelTier
    context_length: int
    max_completion_tokens: int
    pricing: ModelPricing
    value_score: float  # Higher is better value
    is_preview: bool
    is_text_only: bool
    estimated_params: str | None = None  # e.g., "7B", "70B", "Unknown"
    source_info: SourceInfo = Field(default_factory=dict)  # Provider-specific metadata

    @property
    def display_name(self) -> str:
        """User-friendly display name - just the model name without technical details."""
        return self.name

    @property
    def sort_key(self) -> tuple[int, int, float, float]:
        """Sorting key: provider (ollama first), tier, value_score desc, cost asc."""
        provider_order = {"ollama": 0, "openrouter": 1}  # Local models first
        tier_order = {
            ModelTier.FLAGSHIP: 0,
            ModelTier.PREMIUM: 1,
            ModelTier.BALANCED: 2,
            ModelTier.BUDGET: 3,
        }
        return (
            provider_order.get(self.provider, 99),
            tier_order.get(self.tier, 99),
            -self.value_score,
            self.pricing.avg_cost_per_1k,
        )
