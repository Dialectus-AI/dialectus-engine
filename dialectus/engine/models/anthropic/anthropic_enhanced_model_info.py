"""Enhanced Anthropic model information."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dialectus.engine.models.base_types import (
    BaseEnhancedModelInfo,
    ModelPricing,
    ModelTier,
    ModelWeightClass,
)

from .anthropic_model import AnthropicModel
from .anthropic_pricing import AnthropicPricing


class AnthropicEnhancedModelInfo(BaseEnhancedModelInfo):
    """Enhanced Anthropic model information extending base class."""

    def __init__(self, **data: object) -> None:
        """Initialize enhanced model info, normalizing Anthropic-specific types."""
        # Copy into a mutable dict for normalization
        mutable_data = dict(data)
        raw_model = mutable_data.pop("raw_model", None)

        # Convert AnthropicPricing to ModelPricing
        pricing_obj = mutable_data.get("pricing")
        if isinstance(pricing_obj, AnthropicPricing):
            mutable_data["pricing"] = ModelPricing(
                prompt_cost_per_1k=pricing_obj.prompt_cost_per_1k,
                completion_cost_per_1k=pricing_obj.completion_cost_per_1k,
                is_free=pricing_obj.is_free,
                currency="USD",
            )
        elif isinstance(pricing_obj, Mapping):
            # From cache - validate into canonical model
            mutable_data["pricing"] = ModelPricing.model_validate(pricing_obj)

        # Convert string enums to proper enum types
        weight_class = mutable_data.get("weight_class")
        if isinstance(weight_class, str):
            mutable_data["weight_class"] = ModelWeightClass(weight_class)

        tier = mutable_data.get("tier")
        if isinstance(tier, str):
            mutable_data["tier"] = ModelTier(tier)

        # Ensure source_info exists
        source_info_obj = mutable_data.get("source_info")
        if isinstance(source_info_obj, Mapping):
            # Cast to suppress type checker - we know this is a Mapping[str, object] at runtime
            source_info = dict(cast(Mapping[str, object], source_info_obj))
        else:
            source_info = {}
        mutable_data["source_info"] = source_info

        # Store raw model data if provided
        if raw_model is not None and "anthropic_raw" not in source_info:
            if isinstance(raw_model, AnthropicModel):
                source_info["anthropic_raw"] = raw_model.model_dump()
            else:
                source_info["anthropic_raw"] = raw_model

        # Call parent with normalized data
        # Type checker can't verify runtime normalization, so we use type: ignore
        super().__init__(**mutable_data)  # type: ignore[arg-type]
