"""A tiny thread-safe TTL cache for outbound network calls.

Every ``/api/analyze`` request previously triggered a fresh NewsAPI request
plus a Google News RSS fetch, even for an identical query issued seconds
earlier. That wastes the NewsAPI free-tier quota, adds latency to every
request, and makes the service trivial to use as a traffic amplifier.

This is deliberately dependency-free and in-process. For a multi-worker
deployment, swap the backing store for Redis -- the public API here
(:meth:`TTLCache.get` / :meth:`TTLCache.set`) is intentionally small enough
that such a swap is a localised change.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from .logging_config import get_logger

logger = get_logger(__name__)

_MISS = object()


class TTLCache:
    """Fixed-capacity, least-recently-used cache with per-entry expiry."""

    def __init__(self, ttl_seconds: int = 600, max_entries: int = 256):
        """
        Args:
            ttl_seconds: How long an entry stays fresh.
            max_entries: Hard cap on stored entries; the least recently used
                entry is evicted once the cap is exceeded.
        """
        self._ttl = max(1, int(ttl_seconds))
        self._max_entries = max(1, int(max_entries))
        self._store: "OrderedDict[Any, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        """Return the cached value for ``key``, or ``None`` if absent/expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key, _MISS)
            if entry is _MISS:
                return None
            expires_at, value = entry
            if now >= expires_at:
                # Expired: drop it so the caller repopulates.
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)  # mark as recently used
            return value

    def set(self, key: Any, value: Any) -> None:
        """Store ``value`` under ``key`` with the configured TTL."""
        with self._lock:
            self._store[key] = (time.monotonic() + self._ttl, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                evicted, _ = self._store.popitem(last=False)
                logger.debug("Evicted cache key: %r", evicted)

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
