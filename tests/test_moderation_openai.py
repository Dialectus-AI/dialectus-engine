"""Tests for the OpenAI moderation provider."""

from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

import pytest

from httpx import Request, Response
from openai import RateLimitError
from dialectus.engine.config.settings import ModerationConfig, SystemConfig
from dialectus.engine.moderation.exceptions import ModerationProviderError
from dialectus.engine.moderation.manager import ModerationManager
from dialectus.engine.moderation.openai_moderator import OpenAIModerator


class FakeModerationResponse:
    """Lightweight stand-in for the OpenAI moderation response."""

    def __init__(
        self,
        *,
        flagged: bool,
        categories: dict[str, bool],
        scores: dict[str, float],
    ):
        payload = {
            "flagged": flagged,
            "categories": categories,
            "category_scores": scores,
        }
        self.results = [
            SimpleNamespace(
                flagged=flagged,
                categories=categories,
                category_scores=scores,
            )
        ]
        self._payload = payload

    def model_dump_json(self, **kwargs) -> str:
        """Mirror the real response helper for raw logging."""
        return json.dumps({"results": [self._payload]}, **kwargs)


class FakeModerationsClient:
    """Simplified AsyncOpenAI client for testing."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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
    moderator = OpenAIModerator(
        api_key="test-key",
        model="omni-moderation-latest",
        timeout=5.0,
        client=SimpleNamespace(moderations=FakeModerationsClient(response)),
    )

    result = asyncio.run(
        moderator.moderate("Friendly discussion about renewable energy.")
    )

    assert result.is_safe is True
    assert result.categories == []
    assert result.confidence == pytest.approx(0.88, rel=1e-3)
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
    moderator = OpenAIModerator(
        api_key="test-key",
        model="omni-moderation-latest",
        timeout=5.0,
        client=SimpleNamespace(moderations=FakeModerationsClient(response)),
    )

    result = asyncio.run(
        moderator.moderate("Incite violence against a protected group.")
    )

    assert result.is_safe is False
    assert result.categories == ["hate_speech", "violence"]
    assert result.confidence == pytest.approx(0.92, rel=1e-3)


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
    rate_error = RateLimitError("Too many requests", response=response, body=response.json())

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
        client=SimpleNamespace(moderations=client),
        max_retries=2,
        retry_base_delay=0.25,
    )

    result = asyncio.run(moderator.moderate("Check moderation with rate limit"))

    assert result.is_safe is True
    assert sleeps == [0.25]
    assert len(client.requests) == 2


def test_manager_initialises_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manager should build the specialised OpenAI moderator when requested."""
    captured: dict[str, object] = {}

    def fake_openai_moderator(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setenv("OPENAI_API_KEY", "live-test-key")
    monkeypatch.setattr(
        "dialectus.engine.moderation.manager.OpenAIModerator",
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
