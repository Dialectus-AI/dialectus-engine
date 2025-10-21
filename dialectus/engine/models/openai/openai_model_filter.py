"""Filter and enhance OpenAI models for debate use."""

from __future__ import annotations

from dialectus.engine.models.base_types import ModelTier, ModelWeightClass

from .openai_enhanced_model_info import OpenAIEnhancedModelInfo
from .openai_model import OPENAI_MODELS, OpenAIModel


class OpenAIModelFilter:
    """Utility routines that classify OpenAI models into debate-friendly metadata."""

    @staticmethod
    def _classify_weight_class(model: OpenAIModel) -> ModelWeightClass:
        """Determine weight class based on model family."""
        model_id = model.id.lower()
        if model_id.startswith("gpt-4.1") and "mini" not in model_id:
            return ModelWeightClass.ULTRAWEIGHT
        if model_id.startswith("gpt-4.1"):
            return ModelWeightClass.HEAVYWEIGHT
        if model_id.startswith("gpt-4o") and "mini" not in model_id:
            return ModelWeightClass.HEAVYWEIGHT
        if "mini" in model_id:
            return ModelWeightClass.MIDDLEWEIGHT
        return ModelWeightClass.MIDDLEWEIGHT

    @staticmethod
    def _classify_tier(model: OpenAIModel) -> ModelTier:
        """Assign tier based on pricing."""
        avg_cost_per_1k = (
            model.pricing.prompt_cost_per_1k + model.pricing.completion_cost_per_1k
        ) / 2

        if avg_cost_per_1k >= 0.02:
            return ModelTier.FLAGSHIP
        if avg_cost_per_1k >= 0.01:
            return ModelTier.PREMIUM
        if avg_cost_per_1k >= 0.001:
            return ModelTier.BALANCED
        return ModelTier.BUDGET

    @staticmethod
    def _calculate_value_score(model: OpenAIModel) -> float:
        """Compute a simple quality/cost value score."""
        quality_map = {
            "gpt-4.1": 97,
            "gpt-4.1-mini": 88,
            "gpt-4o": 94,
            "gpt-4o-mini": 82,
        }

        quality = 80.0
        for key, score in quality_map.items():
            if key in model.id:
                quality = float(score)
                break

        avg_cost_per_1k = (
            model.pricing.prompt_cost_per_1k + model.pricing.completion_cost_per_1k
        ) / 2
        if avg_cost_per_1k == 0:
            return quality

        return (quality / avg_cost_per_1k) * 10

    @staticmethod
    def filter_and_enhance_models() -> list[OpenAIEnhancedModelInfo]:
        """Convert curated OpenAI models into enhanced metadata objects."""
        enhanced_models: list[OpenAIEnhancedModelInfo] = []

        for model in OPENAI_MODELS:
            weight_class = OpenAIModelFilter._classify_weight_class(model)
            tier = OpenAIModelFilter._classify_tier(model)
            value_score = OpenAIModelFilter._calculate_value_score(model)

            enhanced_models.append(
                OpenAIEnhancedModelInfo(
                    id=model.id,
                    name=model.name,
                    provider="openai",
                    description=model.description,
                    weight_class=weight_class,
                    tier=tier,
                    context_length=model.context_length,
                    max_completion_tokens=model.max_output_tokens,
                    pricing=model.pricing,
                    value_score=value_score,
                    is_preview=not model.is_latest,
                    is_text_only=not model.supports_vision,
                    estimated_params="Unknown",
                    raw_model=model,
                )
            )

        enhanced_models.sort(
            key=lambda info: (
                {
                    ModelTier.FLAGSHIP: 0,
                    ModelTier.PREMIUM: 1,
                    ModelTier.BALANCED: 2,
                    ModelTier.BUDGET: 3,
                }.get(info.tier, 99),
                -info.value_score,
            )
        )

        return enhanced_models
