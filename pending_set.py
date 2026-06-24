"""Deduplication guard: prevents concurrent LLM calls for the same template string."""

from __future__ import annotations
import threading
import time


class PendingReviewSet:
    """Thread-safe set of in-flight template strings with TTL-based expiry."""

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self._ttl = ttl_seconds
        # Maps template_str -> expiry timestamp
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()

    def _expire(self) -> None:
        """Remove entries whose TTL has elapsed (must be called under lock)."""
        now = time.monotonic()
        expired = [k for k, exp in self._pending.items() if now >= exp]
        for k in expired:
            del self._pending[k]

    def should_review(self, template_str: str) -> bool:
        """Return True and mark as pending if not already in-flight; False otherwise."""
        with self._lock:
            self._expire()
            if template_str in self._pending:
                return False
            self._pending[template_str] = time.monotonic() + self._ttl
            return True

    def release(self, template_str: str) -> None:
        """Remove a template from the pending set after the LLM decision is applied."""
        with self._lock:
            self._pending.pop(template_str, None)
