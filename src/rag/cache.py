from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional, Any
from diskcache import Cache

T = TypeVar("T")

@dataclass(frozen=True)
class CacheKeys:
    namespace: str

class DiskCache:
    def __init__(self, directory: str):
        self._cache = Cache(directory)

    def get(self, key: str) -> Any | None:
        return self._cache.get(key, default=None)

    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        self._cache.set(key, value, expire=ttl_s)

    def close(self) -> None:
        self._cache.close()

def cached_call(
    cache: DiskCache,
    key: str,
    fn: Callable[[], T],
    ttl_s: int | None,
) -> T:
    hit = cache.get(key)
    if hit is not None:
        return hit
    val = fn()
    cache.set(key, val, ttl_s=ttl_s)
    return val
