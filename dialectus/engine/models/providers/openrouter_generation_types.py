"""Typed structures for OpenRouter generation API responses."""

from __future__ import annotations

from typing import TypedDict


class OpenRouterGenerationResponse(TypedDict):
    """Complete typed response from OpenRouter generation cost query API."""

    id: str
    upstream_id: str | None
    total_cost: float  # This is the key field we need for cost tracking
    cache_discount: float | None
    upstream_inference_cost: float | None
    created_at: str
    model: str
    app_id: int | None
    streamed: bool | None
    cancelled: bool | None
    provider_name: str | None
    latency: int | None
    moderation_latency: int | None
    generation_time: int | None
    finish_reason: str | None
    native_finish_reason: str | None
    tokens_prompt: int | None
    tokens_completion: int | None
    native_tokens_prompt: int | None
    native_tokens_completion: int | None
    native_tokens_reasoning: int | None
    num_media_prompt: int | None
    num_media_completion: int | None
    num_search_results: int | None
    origin: str
    usage: float
    is_byok: bool


class OpenRouterGenerationApiResponse(TypedDict):
    """Wrapper response from OpenRouter generation API endpoint."""

    data: OpenRouterGenerationResponse


class OpenRouterChatCompletionUsage(TypedDict):
    """Usage information from OpenRouter chat completion response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenRouterChatCompletionResponse(TypedDict):
    """Chat completion response from OpenRouter with generation ID."""

    id: str  # This is the generation ID we need to capture
    object: str
    created: int
    model: str
    choices: list[
        dict[str, object]
    ]  # We'll keep this generic since we only need content
    usage: OpenRouterChatCompletionUsage | None
