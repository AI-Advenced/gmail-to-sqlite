"""
Advanced caching system for Gmail to SQLite.

Provides multiple caching backends (memory, Redis, file-based) with
TTL support, cache invalidation, and performance optimizations.
"""

import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self._cache:
            entry = self._cache[key]
            
            # Check TTL
            if entry.get('expires_at') and time.time() > entry['expires_at']:
                self.delete(key)
                self._misses += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            self._hits += 1
            return entry['value']
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            # Evict if at max size
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            expires_at = None
            if ttl:
                expires_at = time.time() + ttl
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            
            self._access_times[key] = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all entries from memory cache."""
        self._cache.clear()
        self._access_times.clear()
        self._hits = 0
        self._misses = 0
        return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if entry.get('expires_at') and time.time() > entry['expires_at']:
                self.delete(key)
                return False
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'backend': 'memory',
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self.delete(lru_key)


class RedisCache(CacheBackend):
    """Redis-based cache backend."""
    
    def __init__(self, redis_url: str, key_prefix: str = "gmail_sqlite"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._client = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = redis.from_url(self.redis_url, decode_responses=False)
            # Test connection
            self._client.ping()
            logger.info("Redis cache backend connected")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
    
    def _get_key(self, key: str) -> str:
        """Get prefixed cache key."""
        return f"{self.key_prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._client:
            return None
        
        try:
            cached_data = self._client.get(self._get_key(key))
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Redis cache get error for key {key}: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._client:
            return False
        
        try:
            serialized_value = pickle.dumps(value)
            cache_key = self._get_key(key)
            
            if ttl:
                return self._client.setex(cache_key, ttl, serialized_value)
            else:
                return self._client.set(cache_key, serialized_value)
                
        except Exception as e:
            logger.error(f"Redis cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self._client:
            return False
        
        try:
            return bool(self._client.delete(self._get_key(key)))
        except Exception as e:
            logger.error(f"Redis cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries with our prefix."""
        if not self._client:
            return False
        
        try:
            keys = self._client.keys(f"{self.key_prefix}:*")
            if keys:
                return bool(self._client.delete(*keys))
            return True
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._client:
            return False
        
        try:
            return bool(self._client.exists(self._get_key(key)))
        except Exception as e:
            logger.error(f"Redis cache exists error for key {key}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self._client:
            return {'backend': 'redis', 'connected': False}
        
        try:
            info = self._client.info()
            our_keys = len(self._client.keys(f"{self.key_prefix}:*"))
            
            return {
                'backend': 'redis',
                'connected': True,
                'our_keys': our_keys,
                'total_keys': info.get('db0', {}).get('keys', 0),
                'used_memory_mb': round(info.get('used_memory', 0) / (1024 * 1024), 2),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {'backend': 'redis', 'connected': False, 'error': str(e)}


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: str = "data/cache", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key to create safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check TTL from metadata
        if key in self.metadata:
            expires_at = self.metadata[key].get('expires_at')
            if expires_at and time.time() > expires_at:
                self.delete(key)
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"File cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache."""
        try:
            # Check cache size limit
            self._enforce_size_limit()
            
            cache_path = self._get_cache_path(key)
            
            # Serialize and save
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            expires_at = None
            if ttl:
                expires_at = time.time() + ttl
            
            self.metadata[key] = {
                'file': str(cache_path),
                'created_at': time.time(),
                'expires_at': expires_at,
                'size': cache_path.stat().st_size
            }
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"File cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
            
            return True
        except Exception as e:
            logger.error(f"File cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache files."""
        try:
            # Delete all cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            # Clear metadata
            self.metadata = {}
            self._save_metadata()
            
            return True
        except Exception as e:
            logger.error(f"File cache clear error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key in self.metadata:
            expires_at = self.metadata[key].get('expires_at')
            if expires_at and time.time() > expires_at:
                self.delete(key)
                return False
            
            cache_path = Path(self.metadata[key]['file'])
            return cache_path.exists()
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file cache statistics."""
        total_size = 0
        valid_keys = 0
        expired_keys = 0
        
        current_time = time.time()
        
        for key, meta in list(self.metadata.items()):
            if meta.get('expires_at') and current_time > meta['expires_at']:
                expired_keys += 1
            else:
                valid_keys += 1
                total_size += meta.get('size', 0)
        
        return {
            'backend': 'file',
            'cache_dir': str(self.cache_dir),
            'valid_keys': valid_keys,
            'expired_keys': expired_keys,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'max_size_mb': self.max_size_mb
        }
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit by removing oldest entries."""
        total_size = sum(meta.get('size', 0) for meta in self.metadata.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Sort by creation time (oldest first)
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k].get('created_at', 0)
        )
        
        # Remove oldest entries until under limit
        for key in sorted_keys:
            if total_size <= max_size_bytes:
                break
            
            size = self.metadata[key].get('size', 0)
            self.delete(key)
            total_size -= size


class CacheManager:
    """High-level cache manager with multiple backend support."""
    
    def __init__(self, config: Any):
        self.config = config
        self.backend = self._create_backend()
    
    def _create_backend(self) -> CacheBackend:
        """Create appropriate cache backend based on configuration."""
        if not self.config.cache.enabled:
            return MemoryCache(max_size=0)  # Disabled cache
        
        cache_type = self.config.cache.type.lower()
        
        if cache_type == "redis" and REDIS_AVAILABLE:
            if self.config.cache.redis_url:
                try:
                    return RedisCache(self.config.cache.redis_url)
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis cache, falling back to memory: {e}")
            else:
                logger.warning("Redis cache requested but no redis_url configured")
        
        elif cache_type == "file":
            return FileCache(
                cache_dir=self.config.cache.file_cache_dir,
                max_size_mb=self.config.cache.max_size
            )
        
        # Default to memory cache
        return MemoryCache(max_size=self.config.cache.max_size)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if ttl is None:
            ttl = self.config.cache.ttl_seconds
        return self.backend.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self.backend.delete(key)
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        return self.backend.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.backend.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.backend.get_stats()
    
    def cache_function_result(self, func_name: str, args_hash: str, 
                            result: Any, ttl: Optional[int] = None) -> bool:
        """Cache function result with standardized key format."""
        cache_key = f"func:{func_name}:{args_hash}"
        return self.set(cache_key, result, ttl)
    
    def get_cached_function_result(self, func_name: str, args_hash: str) -> Optional[Any]:
        """Get cached function result."""
        cache_key = f"func:{func_name}:{args_hash}"
        return self.get(cache_key)
    
    def invalidate_function_cache(self, func_name: str) -> int:
        """Invalidate all cached results for a function."""
        # This is a simplified implementation
        # Full implementation would need pattern-based key deletion
        prefix = f"func:{func_name}:"
        
        if hasattr(self.backend, '_cache') and isinstance(self.backend, MemoryCache):
            # Memory cache - can iterate keys
            keys_to_delete = [k for k in self.backend._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                self.delete(key)
            return len(keys_to_delete)
        
        # For other backends, would need specific implementation
        logger.warning(f"Function cache invalidation not fully supported for {type(self.backend)}")
        return 0


def cache_decorator(cache_manager: CacheManager, ttl: Optional[int] = None):
    """Decorator to cache function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            args_str = f"{args}:{sorted(kwargs.items())}"
            args_hash = hashlib.md5(args_str.encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_manager.get_cached_function_result(func.__name__, args_hash)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.cache_function_result(func.__name__, args_hash, result, ttl)
            
            return result
        
        return wrapper
    return decorator