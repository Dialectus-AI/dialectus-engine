"""Tests for the model cache manager."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import Enum

import pytest

from dialectus.engine.models import cache_manager as cache_module
from dialectus.engine.models.cache_manager import CacheEntry, ModelCacheManager

RealDateTime = datetime


def freeze_time(monkeypatch: pytest.MonkeyPatch, frozen_at: datetime):
    """Freeze cache_manager.datetime at a specific moment and allow manual advancement."""

    class FrozenDateTime(RealDateTime):
        _now = frozen_at

        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return tz.fromutc(cls._now.replace(tzinfo=tz))
            return cls._now

        @classmethod
        def advance(cls, **delta_kwargs):
            cls._now = cls._now + timedelta(**delta_kwargs)
            return cls._now

    FrozenDateTime._now = frozen_at
    monkeypatch.setattr(cache_module, "datetime", FrozenDateTime)
    return FrozenDateTime


def test_set_get_and_invalidate_memory_cache(tmp_path):
    """Model responses are cached in memory with parameter-based keys."""
    cache_dir = tmp_path / "cache"
    manager = ModelCacheManager(cache_dir=cache_dir)

    payload = {"message": "cached"}
    params = {"size": 42}

    manager.set("provider", "endpoint", payload, params=params)
    # Specific params must be provided to obtain the cached entry.
    assert manager.get("provider", "endpoint", params=params) == payload
    assert manager.get("provider", "endpoint", params={"size": 41}) is None

    stats = manager.get_cache_stats()
    assert stats["memory_entries"] == 1
    assert stats["disk_entries"] == 0

    assert manager.invalidate("provider", "endpoint", params=params) is True
    assert manager.get("provider", "endpoint", params=params) is None
    assert manager.get_cache_stats()["memory_entries"] == 0


def test_entry_expires_after_ttl(monkeypatch, tmp_path):
    """Expired entries are evicted on access based on TTL."""
    frozen_at = datetime(2024, 1, 1, 12, 0, 0)
    frozen_datetime = freeze_time(monkeypatch, frozen_at)

    manager = ModelCacheManager(cache_dir=tmp_path / "cache")

    manager.set("provider", "endpoint", {"payload": True}, ttl_hours=1)
    assert manager.get("provider", "endpoint") == {"payload": True}

    frozen_datetime.advance(hours=2)
    assert manager.get("provider", "endpoint") is None
    assert manager.get_cache_stats()["memory_entries"] == 0


class Dumpable:
    """Object implementing model_dump for serialization testing."""

    def __init__(self, value):
        self.value = value

    def model_dump(self, *, mode=None):
        return {"value": self.value}


class Dictable:
    """Object implementing dict() and returning nested non-serializable objects."""

    def __init__(self, value):
        self.value = value

    def dict(self):
        return {"wrapped": Dumpable(self.value)}


class SampleEnum(Enum):
    OPTION = "option"


def test_save_and_load_round_trip_with_serialization(tmp_path):
    """Disk persistence serializes complex data and loads it back cleanly."""
    cache_dir = tmp_path / "cache"
    manager = ModelCacheManager(cache_dir=cache_dir)

    complex_data = {
        "dump": Dumpable("value"),
        "dict": Dictable(1),
        "enum": SampleEnum.OPTION,
        "tuple": (Dictable(2), 3),
        "list": [Dumpable("nested"), SampleEnum.OPTION],
    }

    entry = CacheEntry(
        data=complex_data,
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
        expires_at=datetime(2024, 1, 1, 16, 0, 0),
        provider="provider",
        endpoint="endpoint",
    )
    cache_key = manager._get_cache_key("provider", "endpoint")  # noqa: SLF001

    manager._save_to_disk(cache_key, entry)  # noqa: SLF001
    cache_file = cache_dir / f"{cache_key}.json"
    assert cache_file.exists()

    with cache_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    expected_serialized = {
        "dump": {"value": "value"},
        "dict": {"wrapped": {"value": 1}},
        "enum": "option",
        "tuple": [{"wrapped": {"value": 2}}, 3],
        "list": [{"value": "nested"}, "option"],
    }

    assert payload["data"] == expected_serialized
    assert payload["timestamp"] == entry.timestamp.isoformat()
    assert payload["expires_at"] == entry.expires_at.isoformat()

    loaded_entry = manager._load_from_disk(cache_key)  # noqa: SLF001
    assert loaded_entry is not None
    assert loaded_entry.data == expected_serialized
    assert loaded_entry.provider == "provider"
    assert loaded_entry.endpoint == "endpoint"


def test_cleanup_expired_removes_memory_and_disk(monkeypatch, tmp_path):
    """cleanup_expired removes expired entries from memory and disk."""
    frozen_at = datetime(2024, 1, 1, 12, 0, 0)
    freeze_time(monkeypatch, frozen_at)

    manager = ModelCacheManager(cache_dir=tmp_path / "cache")
    cache_key = manager._get_cache_key("provider", "endpoint")  # noqa: SLF001

    expired_entry = CacheEntry(
        data={"expired": True},
        timestamp=frozen_at - timedelta(hours=2),
        expires_at=frozen_at - timedelta(hours=1),
        provider="provider",
        endpoint="endpoint",
    )

    manager._memory_cache[cache_key] = expired_entry  # noqa: SLF001
    manager._save_to_disk(cache_key, expired_entry)  # noqa: SLF001
    assert any(manager.cache_dir.glob("*.json"))

    cleaned = manager.cleanup_expired()
    assert cleaned == 2
    assert cache_key not in manager._memory_cache  # noqa: SLF001
    assert not any(manager.cache_dir.glob("*.json"))
