"""Bounded FIFO queue for LLM review items — drops oldest on overflow, never blocks."""

from __future__ import annotations
import collections
import threading


class ReviewQueue:
    """Thread-safe bounded deque; drops the oldest item silently when full."""

    def __init__(self, maxsize: int = 500) -> None:
        self._maxsize = maxsize
        self._deque: collections.deque[dict] = collections.deque()
        self._dropped: int = 0
        self._lock = threading.Lock()

    def put(self, item: dict) -> None:
        """Enqueue an item; if at capacity, evict the oldest and count the drop."""
        with self._lock:
            if len(self._deque) >= self._maxsize:
                self._deque.popleft()
                self._dropped += 1
            self._deque.append(item)

    def get(self) -> dict | None:
        """Dequeue and return the next item, or None if the queue is empty."""
        with self._lock:
            return self._deque.popleft() if self._deque else None

    def stats(self) -> dict:
        """Return current queue depth and cumulative drop count."""
        with self._lock:
            return {"queued": len(self._deque), "dropped": self._dropped}
