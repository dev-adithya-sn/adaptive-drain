"""PendingReviewSet — dedup, TTL expiry, release, thread safety."""

from __future__ import annotations

import threading
import time

from adaptive_drain.pending_set import PendingReviewSet


def test_first_review_admitted_duplicate_blocked() -> None:
    p = PendingReviewSet(ttl_seconds=30)
    assert p.should_review("tmpl-A") is True
    assert p.should_review("tmpl-A") is False  # already in flight
    assert p.should_review("tmpl-B") is True   # distinct template


def test_release_allows_readmission() -> None:
    p = PendingReviewSet(ttl_seconds=30)
    assert p.should_review("tmpl") is True
    assert p.should_review("tmpl") is False
    p.release("tmpl")
    assert p.should_review("tmpl") is True


def test_release_unknown_is_noop() -> None:
    p = PendingReviewSet(ttl_seconds=30)
    p.release("never-added")  # must not raise


def test_ttl_expiry_readmits() -> None:
    p = PendingReviewSet(ttl_seconds=0.05)
    assert p.should_review("tmpl") is True
    assert p.should_review("tmpl") is False
    time.sleep(0.08)  # TTL elapses
    assert p.should_review("tmpl") is True


def test_concurrent_should_review_admits_exactly_one() -> None:
    """Under a race, only one caller may win the in-flight slot for a template."""
    p = PendingReviewSet(ttl_seconds=30)
    results: list[bool] = []
    results_lock = threading.Lock()
    start = threading.Barrier(20)

    def worker() -> None:
        start.wait()  # maximize contention
        r = p.should_review("hot-template")
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == 1, f"expected exactly one winner, got {sum(results)}"
