from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast

from dialectus.engine.models.base_types import BaseEnhancedModelInfo
from dialectus.engine.models.openrouter.openrouter_model import OpenRouterPricing

from ..base_types import ModelPricing, ModelTier, ModelWeightClass


class _EnhancedModelPayload(TypedDict, total=False):
    """Typed view of the payload accepted by the enhanced model initializer."""

    id: str
    name: str
    provider: str
    description: str
    weight_class: ModelWeightClass | str
    tier: ModelTier | str
    context_length: int
    max_completion_tokens: int
    pricing: ModelPricing | OpenRouterPricing | Mapping[str, object]
    value_score: float
    is_preview: bool
    is_text_only: bool
    estimated_params: str | None
    source_info: Mapping[str, object]


class _ValidatedPayload(TypedDict, total=False):
    """Typed view once provider-specific variants have been normalized."""

    id: str
    name: str
    provider: str
    description: str
    weight_class: ModelWeightClass
    tier: ModelTier
    context_length: int
    max_completion_tokens: int
    pricing: ModelPricing
    value_score: float
    is_preview: bool
    is_text_only: bool
    estimated_params: str | None
    source_info: Mapping[str, object]


class OpenRouterEnhancedModelInfo(BaseEnhancedModelInfo):
    """Enhanced OpenRouter model information extending base class."""

    def __init__(self, **data: object) -> None:
        # Copy into a mutable, typed payload so we can safely normalize inputs.
        mutable_data = dict(data)
        raw_model = mutable_data.pop("raw_model", {})
        payload = cast(_EnhancedModelPayload, mutable_data)

        pricing_obj = payload.get("pricing")
        if isinstance(pricing_obj, OpenRouterPricing):
            # Fresh API data ships with provider-specific pricing; convert eagerly so
            # downstream code never has to handle both shapes (keeps pyright and 
            # business logic aligned).
            payload["pricing"] = ModelPricing(
                prompt_cost_per_1k=pricing_obj.prompt_cost_per_1k,
                completion_cost_per_1k=pricing_obj.completion_cost_per_1k,
                is_free=pricing_obj.is_free,
                currency="USD",
            )
        elif isinstance(pricing_obj, Mapping):
            # Cached data (e.g., from disk) comes back as a plain mapping; validate into
            # the canonical model.
            payload["pricing"] = ModelPricing.model_validate(pricing_obj)

        weight_class = payload.get("weight_class")
        if isinstance(weight_class, str):
            payload["weight_class"] = ModelWeightClass(weight_class)

        tier = payload.get("tier")
        if isinstance(tier, str):
            payload["tier"] = ModelTier(tier)

        source_info_obj = payload.get("source_info")
        if isinstance(source_info_obj, Mapping):
            source_info = dict(source_info_obj)
        else:
            # Source info is optional in cached blobs; normalise to a dict for
            # consistent downstream typing.
            source_info = {}
        payload["source_info"] = source_info

        if "openrouter_raw" not in source_info:
            source_info["openrouter_raw"] = raw_model

        # Pydantic's runtime validation will confirm the payload contents; the cast
        # keeps pyright aware that every field now matches the stricter base-model
        # signature after our normalization above.
        validated_payload = cast(_ValidatedPayload, payload)
        super().__init__(**validated_payload)
