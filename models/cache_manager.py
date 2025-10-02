from __future__ import annotations

"""Model response caching system with expiry."""

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Protocol, TypedDict, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
ParamsMapping = Mapping[str, object]


@runtime_checkable
class SupportsModelDump(Protocol):
    """Protocol for objects exposing Pydantic's model_dump."""

    def model_dump(self, *, mode: str | None = None) -> JsonValue:  # pragma: no cover
        ...


@runtime_checkable
class SupportsDict(Protocol):
    """Protocol for objects exposing a dict() method."""

    def dict(self) -> Mapping[str, object]:  # pragma: no cover
        ...


logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """Represents a cached API response."""

    data: object
    timestamp: datetime
    expires_at: datetime
    provider: str
    endpoint: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CacheStats(TypedDict):
    """Snapshot of cache usage."""

    memory_entries: int
    memory_expired: int
    disk_entries: int
    cache_dir: str
    default_ttl_hours: int


class ModelCacheManager:
    """Manages caching of model API responses with automatic expiry."""

    def __init__(self, cache_dir: Path | None = None, default_ttl_hours: int = 6):
        """Initialize cache manager."""
        self.cache_dir = cache_dir or Path("./cache")
        self.default_ttl_hours = default_ttl_hours
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: dict[str, CacheEntry] = {}

        logger.info(
            "Model cache manager initialized. Cache dir: %s, Default TTL: %sh",
            self.cache_dir,
            default_ttl_hours,
        )

    def _get_cache_key(
        self, provider: str, endpoint: str, params: ParamsMapping | None = None
    ) -> str:
        """Generate cache key from provider, endpoint, and parameters."""
        key_parts = [provider, endpoint]
        if params:
            param_dict = dict(params)
            param_str = json.dumps(param_dict, sort_keys=True, default=str)
            key_parts.append(param_str)

        return "_".join(key_parts).replace("/", "_")

    def _get_cache_file(self, cache_key: str) -> Path:
        """Return cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_disk(self, cache_key: str) -> CacheEntry | None:
        """Load cache entry from disk."""
        cache_file = self._get_cache_file(cache_key)
        if not cache_file.exists():
            return None

        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            return CacheEntry(
                data=data["data"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                expires_at=datetime.fromisoformat(data["expires_at"]),
                provider=data["provider"],
                endpoint=data["endpoint"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load cache file %s: %s", cache_file, exc)
            try:
                cache_file.unlink()
            except Exception:  # noqa: BLE001
                pass
            return None

    def _save_to_disk(self, cache_key: str, entry: CacheEntry) -> None:
        """Persist cache entry on disk."""
        cache_file = self._get_cache_file(cache_key)

        try:
            serializable_data = self._make_serializable(entry.data)
            payload: dict[str, JsonValue] = {
                "data": serializable_data,
                "timestamp": entry.timestamp.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "provider": entry.provider,
                "endpoint": entry.endpoint,
            }

            with cache_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

            logger.debug("Cached response saved to %s", cache_file)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save cache file %s: %s", cache_file, exc)

    def _make_serializable(self, data: object) -> JsonValue:
        """Convert data to a JSON-serializable structure."""
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data

        if isinstance(data, SupportsModelDump):
            try:
                return data.model_dump(mode="json")
            except Exception:  # noqa: BLE001
                return data.model_dump()

        if isinstance(data, SupportsDict):
            try:
                raw_mapping = data.dict()
            except Exception:  # noqa: BLE001
                raw_mapping = data.dict()
            return {
                str(key): self._make_serializable(value)
                for key, value in raw_mapping.items()
            }

        if isinstance(data, Enum):
            enum_value = data.value
            if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
                return enum_value
            return str(enum_value)

        if isinstance(data, list):
            list_data = cast(list[object], data)
            return [self._make_serializable(item) for item in list_data]

        if isinstance(data, tuple):
            tuple_data = cast(tuple[object, ...], data)
            return [self._make_serializable(item) for item in tuple_data]

        if isinstance(data, Mapping):
            mapping_data = cast(Mapping[object, object], data)
            return {
                str(key): self._make_serializable(value)
                for key, value in mapping_data.items()
            }

        try:
            json.dumps(data)
        except (TypeError, ValueError):
            return str(data)

        if isinstance(data, (str, int, float, bool)) or data is None:
            return data

        return str(data)

    def get(
        self,
        provider: str,
        endpoint: str,
        params: ParamsMapping | None = None,
    ) -> object | None:
        """Return cached response if available and not expired."""
        cache_key = self._get_cache_key(provider, endpoint, params)

        entry = self._memory_cache.get(cache_key)
        if entry and datetime.now() < entry.expires_at:
            logger.debug("Cache HIT (memory): %s", cache_key)
            return entry.data

        if entry:
            del self._memory_cache[cache_key]

        disk_entry = self._load_from_disk(cache_key)
        if disk_entry and datetime.now() < disk_entry.expires_at:
            self._memory_cache[cache_key] = disk_entry
            logger.debug("Cache HIT (disk): %s", cache_key)
            return disk_entry.data

        logger.debug("Cache MISS: %s", cache_key)
        return None

    def set(
        self,
        provider: str,
        endpoint: str,
        data: object,
        params: ParamsMapping | None = None,
        ttl_hours: int | None = None,
    ) -> None:
        """Store response in cache with expiry."""
        cache_key = self._get_cache_key(provider, endpoint, params)
        ttl = ttl_hours or self.default_ttl_hours
        now = datetime.now()
        expires_at = now + timedelta(hours=ttl)

        entry = CacheEntry(
            data=data,
            timestamp=now,
            expires_at=expires_at,
            provider=provider,
            endpoint=endpoint,
        )

        self._memory_cache[cache_key] = entry
        self._save_to_disk(cache_key, entry)

        logger.info(
            "Cached %s/%s response (expires: %s)",
            provider,
            endpoint,
            expires_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def invalidate(
        self,
        provider: str,
        endpoint: str,
        params: ParamsMapping | None = None,
    ) -> bool:
        """Invalidate cached response."""
        cache_key = self._get_cache_key(provider, endpoint, params)

        memory_removed = cache_key in self._memory_cache
        if memory_removed:
            del self._memory_cache[cache_key]

        cache_file = self._get_cache_file(cache_key)
        disk_removed = False
        if cache_file.exists():
            try:
                cache_file.unlink()
                disk_removed = True
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to remove cache file %s: %s", cache_file, exc)

        if memory_removed or disk_removed:
            logger.info("Invalidated cache: %s", cache_key)
            return True

        return False

    def cleanup_expired(self) -> int:
        """Remove expired cache entries from disk and memory."""
        now = datetime.now()
        cleaned_count = 0

        expired_keys = [
            key for key, entry in self._memory_cache.items() if now >= entry.expires_at
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            cleaned_count += 1

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                entry = self._load_from_disk(cache_file.stem)
                if entry and now >= entry.expires_at:
                    cache_file.unlink()
                    cleaned_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error checking cache file %s: %s", cache_file, exc)

        if cleaned_count > 0:
            logger.info("Cleaned up %s expired cache entries", cleaned_count)

        return cleaned_count

    def get_cache_stats(self) -> CacheStats:
        """Return cache statistics."""
        now = datetime.now()

        memory_entries = len(self._memory_cache)
        memory_expired = sum(
            1 for entry in self._memory_cache.values() if now >= entry.expires_at
        )
        disk_files = list(self.cache_dir.glob("*.json"))
        disk_entries = len(disk_files)

        return CacheStats(
            memory_entries=memory_entries,
            memory_expired=memory_expired,
            disk_entries=disk_entries,
            cache_dir=str(self.cache_dir),
            default_ttl_hours=self.default_ttl_hours,
        )


cache_manager = ModelCacheManager()
