"""Tests for the OpenAI moderation provider."""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest

from httpx import Request, Response
from openai import AsyncOpenAI, RateLimitError
from dialectus.engine.config.settings import ModerationConfig, SystemConfig
from dialectus.engine.moderation.exceptions import ModerationProviderError
from dialectus.engine.moderation.manager import ModerationManager
from dialectus.engine.moderation.openai_moderator import OpenAIModerator
from dialectus.engine.moderation import manager as manager_module


def _normalise_payload(data: object) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    model_dump = getattr(data, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    try:
        return dict(data)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - defensive fallback
        pass
    raise TypeError(f"Unsupported payload type: {type(data)}")


class FakeModerationResponse:
    """Lightweight stand-in for the OpenAI moderation response."""

    def __init__(
        self,
        *,
        flagged: bool,
        categories: object,
        scores: object,
    ):
        payload = {
            "flagged": flagged,
            "categories": _normalise_payload(categories),
            "category_scores": _normalise_payload(scores),
        }
        self.results = [
            SimpleNamespace(
                flagged=flagged,
                categories=categories,
                category_scores=scores,
            )
        ]
        self._payload = payload

    def model_dump_json(self, **kwargs: Any) -> str:
        """Mirror the real response helper for raw logging."""
        return json.dumps({"results": [self._payload]}, **kwargs)


class FakeModerationsClient:
    """Simplified AsyncOpenAI client for testing."""

    def __init__(self, *responses: FakeModerationResponse | Exception) -> None:
        self._responses: list[FakeModerationResponse | Exception] = list(responses)
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> FakeModerationResponse:
        self.requests.append(dict(kwargs))
        if not self._responses:
            raise AssertionError("No fake responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_fake_openai_client(client: FakeModerationsClient) -> AsyncOpenAI:
    """Cast a fake moderations client into the AsyncOpenAI interface for tests."""
    return cast(AsyncOpenAI, SimpleNamespace(moderations=client))


def test_openai_moderator_marks_safe_content() -> None:
    """Non-flagged content should be treated as safe with high confidence."""
    response = FakeModerationResponse(
        flagged=False,
        categories={
            "harassment": False,
            "hate": False,
        },
        scores={
            "harassment": 0.12,
            "hate": 0.08,
        },
    )
    fake_client = FakeModerationsClient(response)
    moderator = OpenAIModerator(
        api_key="test-key",
        model="omni-moderation-latest",
        timeout=5.0,
        client=make_fake_openai_client(fake_client),
    )

    result = asyncio.run(
        moderator.moderate("Friendly discussion about renewable energy.")
    )

    assert result.is_safe is True
    assert result.categories == []
    assert math.isclose(result.confidence, 0.88, rel_tol=1e-3)
    assert "results" in (result.raw_response or "")


def test_openai_moderator_maps_flagged_categories() -> None:
    """Flagged responses should map OpenAI categories into engine taxonomy."""
    response = FakeModerationResponse(
        flagged=True,
        categories={
            "hate": True,
            "hate/threatening": True,
            "violence": True,
            "sexual": False,
        },
        scores={
            "hate": 0.92,
            "violence": 0.81,
        },
    )
    fake_client = FakeModerationsClient(response)
    moderator = OpenAIModerator(
        api_key="test-key",
        model="omni-moderation-latest",
        timeout=5.0,
        client=make_fake_openai_client(fake_client),
    )

    result = asyncio.run(
        moderator.moderate("Incite violence against a protected group.")
    )

    assert result.is_safe is False
    assert result.categories == ["hate_speech", "violence"]
    assert math.isclose(result.confidence, 0.92, rel_tol=1e-3)


def test_openai_moderator_retries_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate limit responses should trigger exponential backoff retries."""
    success_response = FakeModerationResponse(
        flagged=False,
        categories={"harassment": False},
        scores={"harassment": 0.05},
    )

    request = Request("POST", "https://api.openai.com/v1/moderations")
    response = Response(
        429,
        request=request,
        json={"error": {"message": "Too many requests"}},
    )
    rate_error = RateLimitError(
        "Too many requests", response=response, body=response.json()
    )

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "dialectus.engine.moderation.openai_moderator.asyncio.sleep",
        fake_sleep,
    )

    client = FakeModerationsClient(rate_error, success_response)
    moderator = OpenAIModerator(
        api_key="test-key",
        model="omni-moderation-latest",
        timeout=5.0,
        client=make_fake_openai_client(client),
        max_retries=2,
        retry_base_delay=0.25,
    )

    result = asyncio.run(moderator.moderate("Check moderation with rate limit"))

    assert result.is_safe is True
    assert sleeps == [0.25]
    assert len(client.requests) == 2


class _ModelLike:
    """Minimal stand-in for the Pydantic moderation types."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)

    def __getattr__(self, item: str) -> Any:
        try:
            return self._data[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


def test_openai_moderator_handles_model_objects() -> None:
    """Pydantic-style moderation objects should be normalised automatically."""
    categories_model = _ModelLike({"hate": True, "violence": False})
    scores_model = _ModelLike({"hate": 0.91, "violence": 0.05})
    response = FakeModerationResponse(
        flagged=True,
        categories=categories_model,
        scores=scores_model,
    )
    fake_client = FakeModerationsClient(response)
    moderator = OpenAIModerator(
        api_key="test-key",
        model="omni-moderation-latest",
        timeout=5.0,
        client=make_fake_openai_client(fake_client),
    )

    result = asyncio.run(
        moderator.moderate("Edge case flagged by typed moderation payloads.")
    )

    assert result.is_safe is False
    assert result.categories == ["hate_speech"]
    assert math.isclose(result.confidence, 0.91, rel_tol=1e-3)


def test_manager_initialises_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manager should build the specialised OpenAI moderator when requested."""
    captured: dict[str, object] = {}

    def fake_openai_moderator(**kwargs: object) -> OpenAIModerator:
        captured.update(dict(kwargs))
        return cast(OpenAIModerator, SimpleNamespace())

    monkeypatch.setenv("OPENAI_API_KEY", "live-test-key")
    monkeypatch.setattr(
        manager_module,
        "OpenAIModerator",
        fake_openai_moderator,
    )

    config = ModerationConfig(
        enabled=True,
        provider="openai",
        model="omni-moderation-latest",
        timeout=12.5,
    )
    system = SystemConfig()

    ModerationManager(config, system)

    assert captured == {
        "api_key": "live-test-key",
        "model": "omni-moderation-latest",
        "timeout": 12.5,
        "base_url": "https://api.openai.com/v1",
    }


def test_manager_requires_api_key_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuration without an API key should raise a provider error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = ModerationConfig(
        enabled=True,
        provider="openai",
        model="omni-moderation-latest",
    )
    system = SystemConfig()

    with pytest.raises(ModerationProviderError) as excinfo:
        ModerationManager(config, system)

    assert "requires an API key" in str(excinfo.value)
