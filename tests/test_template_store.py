"""TemplateStore lifecycle + reverse-index invariants."""

from __future__ import annotations

import pytest

from adaptive_drain.template_store import TemplateStore, TemplateStatus


@pytest.fixture
def store() -> TemplateStore:
    s = TemplateStore(confirm_threshold=3)
    for cid in ("1", "2", "3"):
        s.register(cid, f"pattern-{cid}")
    return s


def test_register_is_idempotent(store: TemplateStore) -> None:
    store.register("1", "different-pattern")
    assert store.get("1").pattern == "pattern-1"  # original kept
    assert store.stats()["ACTIVE"] == 3


def test_stage_confirm_to_threshold_hard_deletes(store: TemplateStore) -> None:
    store.stage_merge("2", "1")
    assert store.get("2").status is TemplateStatus.PENDING_MERGE
    # threshold is 3: two hits below, third triggers delete
    store.confirm_merge_hit("1")
    store.confirm_merge_hit("1")
    assert store.get("2") is not None
    assert store.get("2").confirmation_count == 2
    store.confirm_merge_hit("1")
    assert store.get("2") is None  # deleted at threshold
    assert store._pending_by_target == {}  # index cleaned


def test_confirm_only_touches_matching_target(store: TemplateStore) -> None:
    store.stage_merge("2", "1")
    store.stage_merge("3", "1")
    # hits against a non-target do nothing
    store.confirm_merge_hit("99")
    assert store.get("2").confirmation_count == 0
    assert store.get("3").confirmation_count == 0
    # hits against the shared target advance both
    store.confirm_merge_hit("1")
    assert store.get("2").confirmation_count == 1
    assert store.get("3").confirmation_count == 1


def test_rollback_restores_active_and_cleans_index(store: TemplateStore) -> None:
    store.stage_merge("2", "1")
    assert store._pending_by_target == {"1": {"2"}}
    store.rollback_merge("2")
    assert store.get("2").status is TemplateStatus.ACTIVE
    assert store.get("2").merge_target_id is None
    assert store._pending_by_target == {}
    # confirm hits after rollback no longer advance it
    store.confirm_merge_hit("1")
    assert store.get("2").confirmation_count == 0


def test_restage_to_new_target_drops_stale_link(store: TemplateStore) -> None:
    store.stage_merge("2", "1")
    store.stage_merge("2", "3")  # re-point
    assert store._pending_by_target == {"3": {"2"}}
    # hits against the old target are inert
    store.confirm_merge_hit("1")
    assert store.get("2").confirmation_count == 0
    store.confirm_merge_hit("3")
    assert store.get("2").confirmation_count == 1


def test_hard_delete_pending_cleans_reverse_index(store: TemplateStore) -> None:
    store.stage_merge("2", "1")
    store.hard_delete("2")
    assert store.get("2") is None
    assert store._pending_by_target == {}


def test_hard_delete_active_template(store: TemplateStore) -> None:
    store.hard_delete("3")
    assert store.get("3") is None
    assert store.stats()["ACTIVE"] == 2


def test_all_active_excludes_pending(store: TemplateStore) -> None:
    store.stage_merge("2", "1")
    active_ids = {t.cluster_id for t in store.all_active()}
    assert active_ids == {"1", "3"}


def test_stats_breakdown(store: TemplateStore) -> None:
    store.stage_merge("2", "1")
    assert store.stats() == {"ACTIVE": 2, "PENDING_MERGE": 1, "MERGED": 0}
