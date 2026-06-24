"""Runtime metrics: thread-safe counters with periodic JSON snapshots (stdlib only)."""

from __future__ import annotations

import json
import threading
import time

_COUNTERS = (
    "logs_ingested",
    "templates_created",
    "templates_updated",
    "templates_merged",
    "templates_split",
    "llm_calls",
    "llm_errors",
    "llm_fallback_keep",
    "queue_drops",
    "ocsf_matched",
    "ocsf_unmatched",
)


class MetricsCollector:
    """Increment-only counters with an optional background emitter thread."""

    def __init__(self, emit_interval_seconds: float = 60.0, log_path: str | None = None) -> None:
        self._interval = emit_interval_seconds
        self._log_path = log_path
        self._counters: dict[str, int] = {name: 0 for name in _COUNTERS}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def increment(self, counter: str, by: int = 1) -> None:
        """Thread-safe increment. Unknown counters are created on demand."""
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + by

    def snapshot(self) -> dict:
        """Return a copy of all counters plus a timestamp."""
        with self._lock:
            snap = dict(self._counters)
        snap["timestamp"] = time.time()
        return snap

    def start(self) -> None:
        """Start the background emitter thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="metrics-emitter")
        self._thread.start()

    def _run(self) -> None:
        # wait() returns True when stopped, False on timeout — emit on timeout.
        while not self._stop_event.wait(self._interval):
            self._emit()

    def _emit(self) -> None:
        """Print the snapshot as JSON to stdout and append to log_path if set."""
        snap = self.snapshot()
        line = json.dumps(snap)
        print(f"[metrics] {line}")
        if self._log_path:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                print(f"[metrics] emit to {self._log_path} failed: {exc}")

    def stop(self) -> None:
        """Signal the emitter to stop and join it (bounded wait)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
