"""Filter and enhance Anthropic models for debate use."""

from __future__ import annotations

from dialectus.engine.models.base_types import ModelTier, ModelWeightClass

from .anthropic_enhanced_model_info import AnthropicEnhancedModelInfo
from .anthropic_model import ANTHROPIC_MODELS, AnthropicModel


class AnthropicModelFilter:
    """Filter and classify Anthropic models."""

    @staticmethod
    def _classify_weight_class(model: AnthropicModel) -> ModelWeightClass:
        """Classify model into weight classes based on capability and cost."""
        # All Claude 3+ models are heavyweight or ultraweight
        if "opus" in model.id.lower():
            return ModelWeightClass.ULTRAWEIGHT
        if "sonnet" in model.id.lower() and "3-5" in model.id:
            return ModelWeightClass.ULTRAWEIGHT  # Claude 3.5 Sonnet is flagship
        if "sonnet" in model.id.lower():
            return ModelWeightClass.HEAVYWEIGHT
        if "haiku" in model.id.lower():
            return ModelWeightClass.MIDDLEWEIGHT

        return ModelWeightClass.HEAVYWEIGHT

    @staticmethod
    def _classify_tier(model: AnthropicModel) -> ModelTier:
        """Classify model into quality/price tiers."""
        avg_cost_per_1k = (
            model.pricing.prompt_cost_per_1k + model.pricing.completion_cost_per_1k
        ) / 2

        # Anthropic tiers based on pricing
        if avg_cost_per_1k > 30:  # Opus: $45/1K avg
            return ModelTier.FLAGSHIP
        if avg_cost_per_1k > 5:  # Sonnet: $9/1K avg
            return ModelTier.PREMIUM
        if avg_cost_per_1k > 1:  # Haiku 3.5: $3/1K avg
            return ModelTier.BALANCED

        return ModelTier.BUDGET  # Haiku 3: $0.75/1K avg

    @staticmethod
    def _calculate_value_score(model: AnthropicModel, tier: ModelTier) -> float:
        """Calculate value score (higher is better value).

        Based on quality/cost ratio with tier-based adjustments.
        """
        # Base quality scores for Anthropic models
        quality_map = {
            "claude-3-5-sonnet": 95,
            "claude-3-5-haiku": 85,
            "claude-3-opus": 98,
            "claude-3-sonnet": 88,
            "claude-3-haiku": 75,
        }

        # Find base quality
        quality = 75  # default
        for key, value in quality_map.items():
            if key in model.id:
                quality = value
                break

        avg_cost_per_1k = (
            model.pricing.prompt_cost_per_1k + model.pricing.completion_cost_per_1k
        ) / 2

        # Prevent division by zero
        if avg_cost_per_1k == 0:
            return quality

        # Value = quality / cost (scaled)
        # Multiply by 10 to get reasonable scale
        return (quality / avg_cost_per_1k) * 10

    @staticmethod
    def filter_and_enhance_models() -> list[AnthropicEnhancedModelInfo]:
        """Convert hardcoded models to enhanced info.

        All Anthropic models are included since we have a curated list.
        """
        enhanced_models: list[AnthropicEnhancedModelInfo] = []

        for model in ANTHROPIC_MODELS:
            weight_class = AnthropicModelFilter._classify_weight_class(model)
            tier = AnthropicModelFilter._classify_tier(model)
            value_score = AnthropicModelFilter._calculate_value_score(model, tier)

            # Estimate parameters based on model tier
            estimated_params = "Unknown"
            if "opus" in model.id.lower():
                estimated_params = "175B+"
            elif "sonnet" in model.id.lower():
                estimated_params = "~100B"
            elif "haiku" in model.id.lower():
                estimated_params = "~50B"

            enhanced = AnthropicEnhancedModelInfo(
                id=model.id,
                name=model.name,
                provider="anthropic",
                description=model.description,
                weight_class=weight_class,
                tier=tier,
                context_length=model.context_length,
                max_completion_tokens=model.max_output_tokens,
                pricing=model.pricing,
                value_score=value_score,
                is_preview=not model.is_latest,  # Older models marked as preview
                is_text_only=not model.supports_vision,
                estimated_params=estimated_params,
                raw_model=model,
            )
            enhanced_models.append(enhanced)

        # Sort by tier and value score
        enhanced_models.sort(
            key=lambda m: (
                {
                    ModelTier.FLAGSHIP: 0,
                    ModelTier.PREMIUM: 1,
                    ModelTier.BALANCED: 2,
                    ModelTier.BUDGET: 3,
                }.get(m.tier, 99),
                -m.value_score,
            )
        )

        return enhanced_models
