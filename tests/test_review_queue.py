"""ReviewQueue — FIFO, overflow (drop-oldest) semantics, drop counter, thread safety."""

from __future__ import annotations

import threading

from adaptive_drain.review_queue import ReviewQueue


def test_fifo_order() -> None:
    q = ReviewQueue(maxsize=10)
    for i in range(5):
        q.put({"i": i})
    out = [q.get()["i"] for _ in range(5)]
    assert out == [0, 1, 2, 3, 4]


def test_get_on_empty_returns_none() -> None:
    q = ReviewQueue(maxsize=3)
    assert q.get() is None


def test_overflow_drops_oldest_and_counts() -> None:
    q = ReviewQueue(maxsize=3)
    for i in range(5):  # 0,1 get evicted; 2,3,4 remain
        q.put({"i": i})
    assert q.stats() == {"queued": 3, "dropped": 2}
    remaining = [q.get()["i"] for _ in range(3)]
    assert remaining == [2, 3, 4]
    assert q.get() is None


def test_stats_snapshot() -> None:
    q = ReviewQueue(maxsize=2)
    q.put({"i": 0})
    assert q.stats() == {"queued": 1, "dropped": 0}


def test_thread_safety_no_lost_or_duplicated_items() -> None:
    """Concurrent puts well under capacity: every item lands exactly once,
    nothing is dropped, and no put corrupts the deque."""
    q = ReviewQueue(maxsize=10_000)
    n_threads, per_thread = 16, 500

    def worker(tid: int) -> None:
        for j in range(per_thread):
            q.put({"tid": tid, "j": j})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = n_threads * per_thread
    assert q.stats() == {"queued": total, "dropped": 0}

    seen = set()
    while (item := q.get()) is not None:
        seen.add((item["tid"], item["j"]))
    assert len(seen) == total  # no loss, no duplication


def test_concurrent_overflow_drop_count_is_accurate() -> None:
    """puts == maxsize + extra ⇒ exactly `extra` drops, regardless of interleaving."""
    maxsize = 100
    extra = 400
    q = ReviewQueue(maxsize=maxsize)
    total = maxsize + extra

    def worker(items: range) -> None:
        for i in items:
            q.put({"i": i})

    chunk = total // 4
    ranges = [range(k * chunk, (k + 1) * chunk) for k in range(4)]
    threads = [threading.Thread(target=worker, args=(r,)) for r in ranges]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = q.stats()
    assert s["queued"] == maxsize
    assert s["dropped"] == extra
