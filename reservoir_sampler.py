"""Per-cluster reservoir sampler — keeps at most `maxsize` representative logs in memory."""

from __future__ import annotations
import random


class ReservoirSampler:
    """Implements Algorithm R reservoir sampling per cluster (O(1) memory per slot)."""

    def __init__(self, maxsize: int = 200) -> None:
        self._maxsize = maxsize
        self._reservoirs: dict[str, list[str]] = {}
        self._counts: dict[str, int] = {}

    def add(self, cluster_id: str, log: str) -> None:
        """Add a log line to the reservoir for the given cluster."""
        if cluster_id not in self._reservoirs:
            self._reservoirs[cluster_id] = []
            self._counts[cluster_id] = 0

        n = self._counts[cluster_id]
        self._counts[cluster_id] = n + 1

        reservoir = self._reservoirs[cluster_id]
        if len(reservoir) < self._maxsize:
            reservoir.append(log)
        else:
            # Replace a random element with probability maxsize / (n+1)
            j = random.randint(0, n)
            if j < self._maxsize:
                reservoir[j] = log

    def get(self, cluster_id: str) -> list[str]:
        """Return the current reservoir contents for a cluster (may be empty)."""
        return list(self._reservoirs.get(cluster_id, []))

    def clear(self, cluster_id: str) -> None:
        """Discard all reservoir samples for a cluster."""
        self._reservoirs.pop(cluster_id, None)
        self._counts.pop(cluster_id, None)
