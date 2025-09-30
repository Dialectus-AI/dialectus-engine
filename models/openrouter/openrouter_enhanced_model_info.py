from ..base_types import ModelPricing
from models.base_types import BaseEnhancedModelInfo
from models.openrouter.openrouter_model import OpenRouterPricing


class OpenRouterEnhancedModelInfo(BaseEnhancedModelInfo):
    """Enhanced OpenRouter model information extending base class."""

    def __init__(self, **data):
        # Handle pricing field - could be OpenRouterPricing (fresh) or dict (cached)
        if "pricing" in data:
            if isinstance(data["pricing"], OpenRouterPricing):
                # Fresh data: convert OpenRouter pricing to generic pricing
                openrouter_pricing = data["pricing"]
                data["pricing"] = ModelPricing(
                    prompt_cost_per_1k=openrouter_pricing.prompt_cost_per_1k,
                    completion_cost_per_1k=openrouter_pricing.completion_cost_per_1k,
                    is_free=openrouter_pricing.is_free,
                    currency="USD",
                )
            elif isinstance(data["pricing"], dict):
                # Cached data: create ModelPricing from dict
                data["pricing"] = ModelPricing(**data["pricing"])

        # Handle enum conversions from cached string values
        if "weight_class" in data and isinstance(data["weight_class"], str):
            from ..base_types import ModelWeightClass

            data["weight_class"] = ModelWeightClass(data["weight_class"])

        if "tier" in data and isinstance(data["tier"], str):
            from ..base_types import ModelTier

            data["tier"] = ModelTier(data["tier"])

        # Store OpenRouter-specific info
        if "openrouter_raw" not in data.get("source_info", {}):
            data.setdefault("source_info", {})["openrouter_raw"] = data.get(
                "raw_model", {}
            )

        super().__init__(**data)
