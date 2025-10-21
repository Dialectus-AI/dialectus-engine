"""Enhanced OpenAI model information."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dialectus.engine.models.base_types import (
    BaseEnhancedModelInfo,
    ModelPricing,
    ModelTier,
    ModelWeightClass,
)

from .openai_model import OpenAIModel
from .openai_pricing import OpenAIPricing


class OpenAIEnhancedModelInfo(BaseEnhancedModelInfo):
    """Enhanced metadata for OpenAI models."""

    def __init__(self, **data: object) -> None:
        """Normalise OpenAI-specific fields before BaseEnhancedModelInfo init."""
        mutable_data = dict(data)
        raw_model = mutable_data.pop("raw_model", None)

        pricing_obj = mutable_data.get("pricing")
        if isinstance(pricing_obj, OpenAIPricing):
            mutable_data["pricing"] = ModelPricing(
                prompt_cost_per_1k=pricing_obj.prompt_cost_per_1k,
                completion_cost_per_1k=pricing_obj.completion_cost_per_1k,
                is_free=pricing_obj.is_free,
                currency="USD",
            )
        elif isinstance(pricing_obj, Mapping):
            mutable_data["pricing"] = ModelPricing.model_validate(pricing_obj)

        weight_class = mutable_data.get("weight_class")
        if isinstance(weight_class, str):
            mutable_data["weight_class"] = ModelWeightClass(weight_class)

        tier = mutable_data.get("tier")
        if isinstance(tier, str):
            mutable_data["tier"] = ModelTier(tier)

        source_info_obj = mutable_data.get("source_info")
        if isinstance(source_info_obj, Mapping):
            source_info = dict(cast(Mapping[str, object], source_info_obj))
        else:
            source_info = {}
        mutable_data["source_info"] = source_info

        if raw_model is not None and "openai_raw" not in source_info:
            if isinstance(raw_model, OpenAIModel):
                source_info["openai_raw"] = raw_model.model_dump()
            else:
                source_info["openai_raw"] = raw_model

        super().__init__(**mutable_data)  # type: ignore[arg-type]
