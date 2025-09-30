"""
OpenRouter API types and model filtering logic.
Handles model selection, pricing analysis, and weight classification.
"""

import re
import logging

from models.openrouter.openrouter_capability_extractor import OpenRouterCapabilityExtractor
from models.openrouter.openrouter_enhanced_model_info import OpenRouterEnhancedModelInfo
from models.openrouter.openrouter_filter_config import OpenRouterFilterConfig
from models.openrouter.openrouter_model import OpenRouterModel
from ..base_types import ModelWeightClass, ModelTier, ModelPricing

logger = logging.getLogger(__name__)


class OpenRouterModelFilter:
    """Intelligent filtering for OpenRouter models with configurable patterns."""

    def __init__(self, filter_config: OpenRouterFilterConfig | None = None):
        """Initialize with optional custom filter configuration."""
        self.filter_config = filter_config or OpenRouterFilterConfig()

    @classmethod
    def is_suitable_for_debate(cls, model: OpenRouterModel) -> bool:
        """Check if model is suitable for text-based conversation and debate."""
        # Must have text input and output
        has_text_input = "text" in model.architecture.input_modalities
        has_text_output = "text" in model.architecture.output_modalities

        if not (has_text_input and has_text_output):
            return False

        # Exclude models that output non-text (images, audio, video)
        # These are fundamentally different from multimodal models that can INPUT multiple types
        non_text_outputs = [
            m for m in model.architecture.output_modalities if m != "text"
        ]
        if len(non_text_outputs) > 0:
            return False

        # Allow multimodal INPUT models - they're often better at reasoning and text generation
        # Examples: GPT-4V, Claude 3, Gemini Pro Vision - excellent for debates even with image capability

        # Additional checks based on model name/description for debate suitability
        model_name_lower = model.name.lower()
        description_lower = model.description.lower()

        # Exclude models with names suggesting non-conversational focus
        non_conversational_keywords = [
            "instruct-only",
            "completion-only",
            "base-model",
            "fine-tuned-only",
            "retrieval",
            "embedding",
            "classification",
        ]

        for keyword in non_conversational_keywords:
            if keyword in model_name_lower or keyword in description_lower:
                return False

        return True

    def is_preview_model(self, model: OpenRouterModel) -> bool:
        """Detect preview/anonymous models using configurable patterns."""
        model_id_lower = model.id.lower()
        model_name_lower = model.name.lower()
        description_lower = model.description.lower()

        # Check configurable preview patterns
        preview_patterns = self.filter_config.get_preview_patterns()
        for pattern in preview_patterns:
            if re.search(pattern, model_id_lower) or re.search(
                pattern, model_name_lower
            ):
                return True

        # Check description for preview keywords
        preview_keywords = ["preview", "anonymous", "beta", "experimental", "temporary"]
        if any(keyword in description_lower for keyword in preview_keywords):
            return True

        # Free models with suspicious names (don't match known providers)
        if model.pricing.is_free:
            known_free_providers = ["meta-llama", "microsoft", "google", "huggingface"]
            if not any(
                provider in model.id.lower() for provider in known_free_providers
            ):
                # Could be a preview model
                return True

        return False

    def should_exclude_model(self, model: OpenRouterModel) -> bool:
        """Check if model should be excluded entirely using configurable patterns."""
        model_id_lower = model.id.lower()
        model_name_lower = model.name.lower()

        # Check configurable exclusion patterns
        exclude_patterns = self.filter_config.get_exclude_patterns()
        for pattern in exclude_patterns:
            if re.search(pattern, model_id_lower) or re.search(
                pattern, model_name_lower
            ):
                logger.debug(f"Excluding model {model.id} due to pattern: {pattern}")
                return True

        return False

    @classmethod
    def classify_model(
        cls, model: OpenRouterModel
    ) -> tuple[ModelWeightClass, str | None]:
        """Classify model into weight class and estimate parameters using dynamic extraction."""

        # Use the new capability extractor for automatic classification
        estimated_params = OpenRouterCapabilityExtractor.extract_parameter_count(model)
        weight_class = OpenRouterCapabilityExtractor.determine_weight_class(
            model, estimated_params
        )

        return weight_class, estimated_params

    @classmethod
    def calculate_value_score(cls, model: OpenRouterModel) -> float:
        """Calculate value score (higher = better value) using debate-focused scoring."""

        # Use the new debate-focused scoring system
        weight_class = OpenRouterCapabilityExtractor.determine_weight_class(model)
        return OpenRouterCapabilityExtractor.calculate_debate_score(model, weight_class)

    @classmethod
    def determine_tier(
        cls,
        model: OpenRouterModel,
        weight_class: ModelWeightClass,
        model_filter: "OpenRouterModelFilter",
    ) -> ModelTier:
        """Determine model tier based on debate performance, cost efficiency, and capabilities."""

        # Check if model is preview using provided filter instance or create minimal check
        if model_filter and model_filter.is_preview_model(model):
            return ModelTier.BUDGET  # Preview models are always budget tier
        elif not model_filter and "preview" in model.name.lower():
            return ModelTier.BUDGET  # Fallback preview check without filter

        avg_cost = model.pricing.avg_cost_per_1k
        context_length = model.context_length

        # Tier determination based on debate suitability
        # Consider: context length (debate flow), cost efficiency, and model size

        # Free models are automatically good value
        if model.pricing.is_free:
            if weight_class == ModelWeightClass.ULTRAWEIGHT:
                return ModelTier.FLAGSHIP
            elif weight_class == ModelWeightClass.HEAVYWEIGHT:
                return ModelTier.PREMIUM
            else:
                return ModelTier.BALANCED

        # Cost-based tiers with context length consideration
        if weight_class == ModelWeightClass.ULTRAWEIGHT:
            # Large models: Premium if reasonably priced, Flagship if expensive but capable
            if avg_cost <= 0.01:
                return ModelTier.PREMIUM  # Great large model at good price
            elif avg_cost <= 0.02:
                return (
                    ModelTier.FLAGSHIP if context_length >= 64000 else ModelTier.PREMIUM
                )
            else:
                return ModelTier.FLAGSHIP  # Expensive but most capable

        elif weight_class == ModelWeightClass.HEAVYWEIGHT:
            # Medium-large models: Good balance for debates
            if avg_cost <= 0.003:
                return ModelTier.PREMIUM  # Excellent value
            elif avg_cost <= 0.008:
                return ModelTier.BALANCED  # Good value
            else:
                return ModelTier.PREMIUM  # Higher cost but good capabilities

        elif weight_class == ModelWeightClass.MIDDLEWEIGHT:
            # Medium models: Depend heavily on cost
            if avg_cost <= 0.002:
                return ModelTier.BALANCED  # Good value for medium model
            elif avg_cost <= 0.006:
                return (
                    ModelTier.PREMIUM if context_length >= 32000 else ModelTier.BALANCED
                )
            else:
                return ModelTier.PREMIUM  # Expensive medium model, likely high quality

        else:  # LIGHTWEIGHT
            # Small models: Mainly for budget tier unless very cheap
            if avg_cost <= 0.001:
                return (
                    ModelTier.BALANCED if context_length >= 16000 else ModelTier.BUDGET
                )
            else:
                return ModelTier.BUDGET

    @classmethod
    def filter_and_enhance_models(
        cls,
        models: list[OpenRouterModel],
        include_preview: bool | None = None,
        max_cost_per_1k: float | None = None,
        min_context_length: int | None = None,
        max_models_per_tier: int | None = None,
        filter_config: OpenRouterFilterConfig | None = None,
    ) -> list[OpenRouterEnhancedModelInfo]:
        """Filter and enhance models with intelligent selection using configurable filters."""

        # Initialize filter config and get settings
        config = filter_config or OpenRouterFilterConfig()

        # Use config defaults if parameters not specified
        if include_preview is None:
            include_preview = bool(config.get_setting("allow_preview_models", False) or False)
        if max_cost_per_1k is None:
            max_cost_per_1k = float(config.get_setting("max_cost_per_1k_tokens", 0.02) or 0.02)
        if min_context_length is None:
            min_context_length = int(config.get_setting("min_context_length", 4096) or 4096)
        if max_models_per_tier is None:
            max_models_per_tier = int(config.get_setting("max_models_per_tier", 5) or 5)

        # Get free tier exclusion setting
        exclude_free_tier = bool(config.get_setting("exclude_free_tier_models", False) or False)

        # Create filter instance
        model_filter = cls(config)
        enhanced_models = []

        for model in models:
            # Skip excluded models (including meta models)
            if model_filter.should_exclude_model(model):
                continue

            # Skip models not suitable for debate
            if not cls.is_suitable_for_debate(model):
                continue

            # Skip preview models if not requested
            is_preview = model_filter.is_preview_model(model)
            if is_preview and not include_preview:
                continue

            # Skip :free tier models if configured to exclude them (production setting)
            if exclude_free_tier and model.id.endswith(":free"):
                logger.debug(f"Excluding free-tier model {model.id} (exclude_free_tier_models=true)")
                continue

            # Skip models that are too expensive
            if (
                max_cost_per_1k is not None
                and model.pricing.avg_cost_per_1k > max_cost_per_1k
                and not model.pricing.is_free
            ):
                continue

            # Skip models with insufficient context
            if (
                min_context_length is not None
                and model.context_length < min_context_length
            ):
                continue

            # Classify and score the model
            weight_class, estimated_params = cls.classify_model(model)
            value_score = cls.calculate_value_score(model)
            tier = cls.determine_tier(model, weight_class, model_filter)

            # Convert OpenRouter pricing to generic pricing
            generic_pricing = ModelPricing(
                prompt_cost_per_1k=model.pricing.prompt_cost_per_1k,
                completion_cost_per_1k=model.pricing.completion_cost_per_1k,
                is_free=model.pricing.is_free,
                currency="USD",
            )

            enhanced_model = OpenRouterEnhancedModelInfo(
                id=model.id,
                name=model.name,
                provider="openrouter",
                description=model.description,
                weight_class=weight_class,
                tier=tier,
                context_length=model.context_length,
                max_completion_tokens=model.top_provider.max_completion_tokens or 4096,
                pricing=generic_pricing,
                value_score=value_score,
                is_preview=is_preview,
                is_text_only=(
                    len([m for m in model.architecture.input_modalities if m != "text"])
                    == 0
                ),
                estimated_params=estimated_params,
                source_info={"openrouter_raw": model.model_dump()},
            )

            enhanced_models.append(enhanced_model)

        # Sort models by tier and value
        enhanced_models.sort(key=lambda m: m.sort_key)

        # Limit models per tier to avoid overwhelming users
        if max_models_per_tier is not None and max_models_per_tier > 0:
            tier_counts = {}
            filtered_models = []

            for model in enhanced_models:
                tier_count = tier_counts.get(model.tier, 0)
                if tier_count < max_models_per_tier:
                    filtered_models.append(model)
                    tier_counts[model.tier] = tier_count + 1

            enhanced_models = filtered_models

        logger.info(
            f"Filtered {len(models)} OpenRouter models down to {len(enhanced_models)} suitable for debate"
        )
        return enhanced_models
