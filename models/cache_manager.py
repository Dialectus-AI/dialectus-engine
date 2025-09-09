"""Model response caching system with expiry."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """Represents a cached API response."""
    data: Any
    timestamp: datetime
    expires_at: datetime
    provider: str
    endpoint: str
    
    class Config:
        arbitrary_types_allowed = True


class ModelCacheManager:
    """Manages caching of model API responses with automatic expiry."""
    
    def __init__(self, cache_dir: Path = None, default_ttl_hours: int = 6):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files (default: ./cache)
            default_ttl_hours: Default cache TTL in hours (default: 6 hours)
        """
        self.cache_dir = cache_dir or Path("./cache")
        self.default_ttl_hours = default_ttl_hours
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for faster access during runtime
        self._memory_cache: Dict[str, CacheEntry] = {}
        
        logger.info(f"Model cache manager initialized. Cache dir: {self.cache_dir}, Default TTL: {default_ttl_hours}h")
    
    def _get_cache_key(self, provider: str, endpoint: str, params: Dict = None) -> str:
        """Generate cache key from provider, endpoint and parameters."""
        key_parts = [provider, endpoint]
        if params:
            # Sort params for consistent key generation
            param_str = json.dumps(params, sort_keys=True)
            key_parts.append(param_str)
        
        return "_".join(key_parts).replace("/", "_")
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_disk(self, cache_key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk."""
        cache_file = self._get_cache_file(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Parse timestamps
            entry = CacheEntry(
                data=data['data'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                expires_at=datetime.fromisoformat(data['expires_at']),
                provider=data['provider'],
                endpoint=data['endpoint']
            )
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _save_to_disk(self, cache_key: str, entry: CacheEntry) -> None:
        """Save cache entry to disk."""
        cache_file = self._get_cache_file(cache_key)
        
        try:
            # Convert Pydantic models to dict if needed
            serializable_data = self._make_serializable(entry.data)
            
            data = {
                'data': serializable_data,
                'timestamp': entry.timestamp.isoformat(),
                'expires_at': entry.expires_at.isoformat(),
                'provider': entry.provider,
                'endpoint': entry.endpoint
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Cached response saved to {cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save cache file {cache_file}: {e}")
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format."""
        import json
        from enum import Enum
        
        if hasattr(data, 'model_dump'):  # Pydantic v2
            # Use mode='json' to properly serialize enums
            try:
                return data.model_dump(mode='json')
            except:
                return data.model_dump()
        elif hasattr(data, 'dict'):  # Pydantic v1
            try:
                return data.dict()
            except:
                pass
        
        if isinstance(data, Enum):  # Handle enum values
            return data.value
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._make_serializable(value) for key, value in data.items()}
        else:
            # Try to serialize to JSON to test if it's serializable
            try:
                json.dumps(data)
                return data
            except (TypeError, ValueError):
                # If it can't be serialized, convert to string
                return str(data)
    
    def get(self, provider: str, endpoint: str, params: Dict = None) -> Optional[Any]:
        """
        Get cached response if available and not expired.
        
        Args:
            provider: Provider name (e.g., 'openrouter')
            endpoint: API endpoint (e.g., 'models')
            params: Optional parameters used in the request
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        cache_key = self._get_cache_key(provider, endpoint, params)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if datetime.now() < entry.expires_at:
                logger.debug(f"Cache HIT (memory): {cache_key}")
                return entry.data
            else:
                # Expired, remove from memory
                del self._memory_cache[cache_key]
        
        # Check disk cache
        entry = self._load_from_disk(cache_key)
        if entry and datetime.now() < entry.expires_at:
            # Valid cache entry, load into memory for faster access
            self._memory_cache[cache_key] = entry
            logger.debug(f"Cache HIT (disk): {cache_key}")
            return entry.data
        
        # Cache miss or expired
        logger.debug(f"Cache MISS: {cache_key}")
        return None
    
    def set(self, provider: str, endpoint: str, data: Any, params: Dict = None, ttl_hours: int = None) -> None:
        """
        Store response in cache with expiry.
        
        Args:
            provider: Provider name (e.g., 'openrouter')
            endpoint: API endpoint (e.g., 'models')
            data: Response data to cache
            params: Optional parameters used in the request
            ttl_hours: Cache TTL in hours (uses default if None)
        """
        cache_key = self._get_cache_key(provider, endpoint, params)
        ttl = ttl_hours or self.default_ttl_hours
        
        now = datetime.now()
        expires_at = now + timedelta(hours=ttl)
        
        entry = CacheEntry(
            data=data,
            timestamp=now,
            expires_at=expires_at,
            provider=provider,
            endpoint=endpoint
        )
        
        # Store in memory and disk
        self._memory_cache[cache_key] = entry
        self._save_to_disk(cache_key, entry)
        
        logger.info(f"Cached {provider}/{endpoint} response (expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')})")
    
    def invalidate(self, provider: str, endpoint: str, params: Dict = None) -> bool:
        """
        Invalidate cached response.
        
        Args:
            provider: Provider name
            endpoint: API endpoint  
            params: Optional parameters
            
        Returns:
            True if cache entry was found and removed, False otherwise
        """
        cache_key = self._get_cache_key(provider, endpoint, params)
        
        # Remove from memory cache
        memory_removed = cache_key in self._memory_cache
        if memory_removed:
            del self._memory_cache[cache_key]
        
        # Remove from disk cache
        cache_file = self._get_cache_file(cache_key)
        disk_removed = False
        if cache_file.exists():
            try:
                cache_file.unlink()
                disk_removed = True
            except Exception as e:
                logger.error(f"Failed to remove cache file {cache_file}: {e}")
        
        if memory_removed or disk_removed:
            logger.info(f"Invalidated cache: {cache_key}")
            return True
        
        return False
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries from disk and memory.
        
        Returns:
            Number of entries cleaned up
        """
        now = datetime.now()
        cleaned_count = 0
        
        # Clean memory cache
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if now >= entry.expires_at
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
            cleaned_count += 1
        
        # Clean disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                entry = self._load_from_disk(cache_file.stem)
                if entry and now >= entry.expires_at:
                    cache_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        
        return cleaned_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        
        # Count memory cache entries
        memory_entries = len(self._memory_cache)
        memory_expired = sum(1 for entry in self._memory_cache.values() if now >= entry.expires_at)
        
        # Count disk cache entries
        disk_files = list(self.cache_dir.glob("*.json"))
        disk_entries = len(disk_files)
        
        return {
            'memory_entries': memory_entries,
            'memory_expired': memory_expired,
            'disk_entries': disk_entries,
            'cache_dir': str(self.cache_dir),
            'default_ttl_hours': self.default_ttl_hours
        }


# Global cache manager instance
cache_manager = ModelCacheManager()