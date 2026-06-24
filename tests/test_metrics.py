"""MetricsCollector — thread-safe counters, snapshot isolation, emit, clean stop."""

from __future__ import annotations

import json
import threading
import time

from adaptive_drain.metrics import MetricsCollector


def test_increment_is_thread_safe() -> None:
    m = MetricsCollector()
    n_threads, per_thread = 50, 100
    start = threading.Barrier(n_threads)

    def worker() -> None:
        start.wait()
        for _ in range(per_thread):
            m.increment("logs_ingested")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert m.snapshot()["logs_ingested"] == n_threads * per_thread


def test_snapshot_returns_copy_not_reference() -> None:
    m = MetricsCollector()
    m.increment("llm_calls", 5)
    snap = m.snapshot()
    snap["llm_calls"] = 99999  # mutate the copy
    assert m.snapshot()["llm_calls"] == 5  # internal state untouched


def test_unknown_counter_increments_gracefully() -> None:
    m = MetricsCollector()
    m.increment("totally_made_up_counter", 3)  # must not raise
    assert m.snapshot()["totally_made_up_counter"] == 3


def test_emit_writes_one_json_line_per_call(tmp_path) -> None:
    log = tmp_path / "metrics.log"
    m = MetricsCollector(log_path=str(log))
    m.increment("templates_created", 2)
    m._emit()
    m.increment("templates_created", 1)
    m._emit()

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["templates_created"] == 2
    assert second["templates_created"] == 3
    assert "timestamp" in first


def test_stop_joins_thread_cleanly() -> None:
    m = MetricsCollector(emit_interval_seconds=0.05)
    m.start()
    time.sleep(0.12)  # let it emit at least once
    m.stop()
    # after stop, the emitter thread reference is cleared and not alive
    assert m._thread is None
