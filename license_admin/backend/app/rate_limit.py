from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        moment = time.monotonic()
        with self._lock:
            recent = [item for item in self._hits.get(key, []) if moment - item < self.window_seconds]
            if len(recent) >= self.limit:
                self._hits[key] = recent
                return False
            recent.append(moment)
            self._hits[key] = recent
            return True
