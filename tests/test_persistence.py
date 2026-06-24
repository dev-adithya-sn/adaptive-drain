"""StatePersistence — save/load round-trips, atomicity, failure tolerance."""

from __future__ import annotations

import json
import os

from adaptive_drain.persistence import StatePersistence
from adaptive_drain.reservoir_sampler import ReservoirSampler
from adaptive_drain.template_store import TemplateStatus, TemplateStore


def _populated_store() -> TemplateStore:
    store = TemplateStore(confirm_threshold=5)
    store.register("1", "User <*> logged in")
    store.register("2", "Order <*> shipped")
    store.stage_merge("2", "1")  # 2 -> PENDING_MERGE pointing at 1
    return store


def _populated_sampler() -> ReservoirSampler:
    s = ReservoirSampler(maxsize=10)
    for i in range(3):
        s.add("1", f"login-{i}")
    for i in range(7):
        s.add("2", f"order-{i}")
    return s


def test_save_creates_file(tmp_path) -> None:
    p = StatePersistence(str(tmp_path / "state.json"))
    assert p.save(_populated_store(), _populated_sampler()) is True
    assert p.exists() is True


def test_load_restores_templates_with_status(tmp_path) -> None:
    path = str(tmp_path / "state.json")
    StatePersistence(path).save(_populated_store(), _populated_sampler())

    store2 = TemplateStore()
    sampler2 = ReservoirSampler()
    assert StatePersistence(path).load(store2, sampler2) is True

    assert store2.get("1").pattern == "User <*> logged in"
    assert store2.get("1").status is TemplateStatus.ACTIVE
    t2 = store2.get("2")
    assert t2.status is TemplateStatus.PENDING_MERGE
    assert t2.merge_target_id == "1"
    # reverse index rebuilt so confirm_merge_hit still works after restore
    assert store2._pending_by_target == {"1": {"2"}}


def test_load_restores_reservoirs_and_counts(tmp_path) -> None:
    path = str(tmp_path / "state.json")
    StatePersistence(path).save(_populated_store(), _populated_sampler())

    store2 = TemplateStore()
    sampler2 = ReservoirSampler()
    StatePersistence(path).load(store2, sampler2)

    assert sampler2.get("1") == [f"login-{i}" for i in range(3)]
    assert len(sampler2.get("2")) == 7
    assert sampler2._counts == {"1": 3, "2": 7}


def test_missing_file_returns_false(tmp_path) -> None:
    p = StatePersistence(str(tmp_path / "nope.json"))
    assert p.exists() is False
    assert p.load(TemplateStore(), ReservoirSampler()) is False


def test_corrupt_json_returns_false(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{ this is not valid json ]]]")
    assert StatePersistence(str(path)).load(TemplateStore(), ReservoirSampler()) is False


def test_version_mismatch_returns_false(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 2, "templates": [], "reservoirs": {}, "reservoir_counts": {}}))
    assert StatePersistence(str(path)).load(TemplateStore(), ReservoirSampler()) is False


def test_atomic_write_leaves_no_tmp_file(tmp_path) -> None:
    path = str(tmp_path / "state.json")
    p = StatePersistence(path)
    assert p.save(_populated_store(), _populated_sampler()) is True
    assert not os.path.exists(path + ".tmp")
