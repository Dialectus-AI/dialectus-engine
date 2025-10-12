from typing import Any

from pydantic import Field

from dialectus.engine.models.base_types import BaseModel
from dialectus.engine.models.openrouter.openrouter_architecture import (
    OpenRouterArchitecture,
)
from dialectus.engine.models.openrouter.openrouter_pricing import OpenRouterPricing
from dialectus.engine.models.openrouter.openrouter_top_provider import (
    OpenRouterTopProvider,
)


class OpenRouterModel(BaseModel):
    id: str
    name: str
    created: int
    description: str
    architecture: OpenRouterArchitecture
    top_provider: OpenRouterTopProvider
    pricing: OpenRouterPricing
    canonical_slug: str
    context_length: int
    hugging_face_id: str | None = None
    per_request_limits: dict[str, Any] | None = None
    supported_parameters: list[str] = Field(default_factory=list)
