"""TemplateSplitter — safe split execution semantics."""

from __future__ import annotations

import pytest

from adaptive_drain.reservoir_sampler import ReservoirSampler
from adaptive_drain.splitter import TemplateSplitter
from adaptive_drain.template_store import TemplateStatus, TemplateStore


class SpyDrain:
    """Records update_template calls so we can assert the slot-reuse rule."""

    def __init__(self) -> None:
        self.update_calls: list[tuple[str, list[str]]] = []

    def update_template(self, cluster_id: str, new_tokens: list[str]) -> bool:
        self.update_calls.append((cluster_id, new_tokens))
        return True


@pytest.fixture
def rig():
    drain = SpyDrain()
    store = TemplateStore(confirm_threshold=3)
    sampler = ReservoirSampler()
    store.register("5", "orig <*>")  # the cluster being split
    splitter = TemplateSplitter(drain, store, sampler)
    return splitter, drain, store, sampler


def test_valid_split_returns_correct_ids(rig) -> None:
    splitter, _, _, _ = rig
    new_ids = splitter.execute_split("5", ["GET <*>", "POST <*>"], [])
    assert new_ids == ["split_5_0", "split_5_1"]


def test_invalid_too_few_subtemplates_returns_empty(rig) -> None:
    splitter, drain, store, _ = rig
    assert splitter.execute_split("5", ["only one"], []) == []
    assert splitter.execute_split("5", [], []) == []
    # nothing should have been registered or injected
    assert drain.update_calls == []
    assert store.get("split_5_0") is None


def test_invalid_empty_string_subtemplate_returns_empty(rig) -> None:
    splitter, _, _, _ = rig
    assert splitter.execute_split("5", ["valid <*>", "   "], []) == []


def test_samples_distributed_round_robin(rig) -> None:
    splitter, _, _, sampler = rig
    samples = ["s0", "s1", "s2", "s3", "s4"]
    splitter.execute_split("5", ["A <*>", "B <*>"], samples)
    # j % 2: bucket0 gets indices 0,2,4 ; bucket1 gets 1,3
    assert sampler.get("split_5_0") == ["s0", "s2", "s4"]
    assert sampler.get("split_5_1") == ["s1", "s3"]


def test_original_staged_for_merge_into_first_subtemplate(rig) -> None:
    splitter, _, store, _ = rig
    splitter.execute_split("5", ["A <*>", "B <*>"], [])
    original = store.get("5")
    assert original.status is TemplateStatus.PENDING_MERGE
    assert original.merge_target_id == "split_5_0"
    # reverse index reflects the staged merge
    assert store._pending_by_target == {"split_5_0": {"5"}}


def test_index0_reuses_drain_slot_only_once(rig) -> None:
    splitter, drain, _, _ = rig
    splitter.execute_split("5", ["GET <*>", "POST <*>", "PUT <*>"], [])
    # exactly one drain injection, on the ORIGINAL cluster, with sub-template 0's tokens
    assert len(drain.update_calls) == 1
    assert drain.update_calls[0] == ("5", ["GET", "<*>"])


def test_store_count_after_split(rig) -> None:
    splitter, _, store, _ = rig
    splitter.execute_split("5", ["A <*>", "B <*>", "C <*>"], [])
    # 3 new ACTIVE sub-templates; original moved to PENDING_MERGE
    assert {t.cluster_id for t in store.all_active()} == {"split_5_0", "split_5_1", "split_5_2"}
    assert store.stats() == {"ACTIVE": 3, "PENDING_MERGE": 1, "MERGED": 0}
