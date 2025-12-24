from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenBucket:
    rate_per_sec: float
    capacity: float

class SyncRateLimiter:
    """Simple token-bucket limiter for sync code."""
    def __init__(self, bucket: TokenBucket):
        self._bucket = bucket
        self._tokens = bucket.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        if self._bucket.rate_per_sec <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self._bucket.capacity, self._tokens + elapsed * self._bucket.rate_per_sec)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_s = deficit / self._bucket.rate_per_sec
            time.sleep(max(0.0, wait_s))
