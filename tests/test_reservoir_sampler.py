"""ReservoirSampler — boundedness, count correctness, Algorithm R uniformity."""

from __future__ import annotations

import random

from adaptive_drain.reservoir_sampler import ReservoirSampler


def test_below_capacity_keeps_everything() -> None:
    s = ReservoirSampler(maxsize=10)
    for i in range(7):
        s.add("c", f"log-{i}")
    assert s.get("c") == [f"log-{i}" for i in range(7)]


def test_reservoir_never_exceeds_maxsize() -> None:
    s = ReservoirSampler(maxsize=50)
    for i in range(10_000):
        s.add("c", f"log-{i}")
    assert len(s.get("c")) == 50


def test_per_cluster_isolation() -> None:
    s = ReservoirSampler(maxsize=5)
    for i in range(3):
        s.add("a", f"a-{i}")
    for i in range(100):
        s.add("b", f"b-{i}")
    assert len(s.get("a")) == 3
    assert len(s.get("b")) == 5
    assert s.get("missing") == []


def test_get_returns_copy() -> None:
    s = ReservoirSampler(maxsize=5)
    s.add("c", "x")
    out = s.get("c")
    out.append("mutated")
    assert s.get("c") == ["x"]  # internal state untouched


def test_clear_resets_count_so_refill_works() -> None:
    s = ReservoirSampler(maxsize=3)
    for i in range(100):
        s.add("c", f"log-{i}")
    s.clear("c")
    assert s.get("c") == []
    # after clear, count restarts: first maxsize adds are kept verbatim
    for i in range(2):
        s.add("c", f"new-{i}")
    assert s.get("c") == ["new-0", "new-1"]


def test_algorithm_r_is_approximately_uniform() -> None:
    """Each of N items should survive in a size-k reservoir with prob ~ k/N.

    Run many independent trials and assert the empirical survival rate of a
    fixed early item is within tolerance of k/N — this catches an off-by-one
    in the replacement probability (the classic Algorithm R bug).
    """
    random.seed(1234)
    k, n, trials = 10, 100, 4000
    target = "item-0"
    survived = 0
    for _ in range(trials):
        s = ReservoirSampler(maxsize=k)
        for i in range(n):
            s.add("c", f"item-{i}")
        if target in s.get("c"):
            survived += 1
    rate = survived / trials
    expected = k / n  # 0.10
    assert abs(rate - expected) < 0.03, f"survival {rate:.3f} vs expected {expected:.3f}"


def test_all_positions_uniform_at_capacity() -> None:
    """With n == maxsize+? the final reservoir should sample positions uniformly;
    sanity-check that no single original index dominates."""
    random.seed(99)
    k, n, trials = 5, 20, 3000
    counts = {i: 0 for i in range(n)}
    for _ in range(trials):
        s = ReservoirSampler(maxsize=k)
        for i in range(n):
            s.add("c", str(i))
        for v in s.get("c"):
            counts[int(v)] += 1
    # each index expected ~ trials * k / n occurrences
    expected = trials * k / n
    for i, c in counts.items():
        assert abs(c - expected) < expected * 0.35, f"index {i}: {c} vs ~{expected:.0f}"
