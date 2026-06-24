"""DrainAdapter — normalisation of drain3's dict return, int-key handling, token ops.

The real drain3.TemplateMiner is mocked: we model the 0.9.11 contract
(add_log_message -> dict with change_type/cluster_id/template_mined, and an
int-keyed `.drain.id_to_cluster` of clusters exposing `log_template_tokens`).
"""

from __future__ import annotations

from adaptive_drain.drain_adapter import DrainAdapter


class FakeCluster:
    def __init__(self, cluster_id: int, tokens):
        self.cluster_id = cluster_id
        self.log_template_tokens = tuple(tokens)


class FakeDrain:
    """Inner Drain: int-keyed cluster map, like drain3's real `.drain`."""

    def __init__(self):
        self.id_to_cluster: dict[int, FakeCluster] = {}


class FakeTemplateMiner:
    """Mimics drain3.TemplateMiner 0.9.11's dict-returning add_log_message."""

    def __init__(self):
        self.drain = FakeDrain()
        self._next_id = 0
        self._next_result: dict | None = None

    def queue_result(self, change_type: str, cluster_id: int, template: str) -> None:
        self._next_result = {
            "change_type": change_type,
            "cluster_id": cluster_id,
            "template_mined": template,
            "cluster_size": 1,
            "cluster_count": len(self.drain.id_to_cluster),
        }

    def add_log_message(self, raw_log: str) -> dict:
        assert self._next_result is not None, "queue_result first"
        res, self._next_result = self._next_result, None
        return res


def test_add_log_normalises_create() -> None:
    miner = FakeTemplateMiner()
    miner.drain.id_to_cluster[1] = FakeCluster(1, ["User", "<*>", "logged", "in"])
    adapter = DrainAdapter(miner)
    miner.queue_result("cluster_created", 1, "User <*> logged in")

    out = adapter.add_log("User 5 logged in")
    assert out["change_type"] == "CREATE"
    assert out["cluster_id"] == "1"          # normalised to str
    assert out["template"] == "User <*> logged in"
    assert out["tokens"] == ["User", "<*>", "logged", "in"]  # from int-keyed lookup


def test_change_type_mapping_and_unknown_falls_back_to_none() -> None:
    miner = FakeTemplateMiner()
    adapter = DrainAdapter(miner)

    miner.queue_result("cluster_template_changed", 2, "a <*>")
    assert adapter.add_log("x")["change_type"] == "UPDATE"

    miner.queue_result("none", 2, "a <*>")
    assert adapter.add_log("x")["change_type"] == "NONE"

    miner.queue_result("some_future_change_type", 2, "a <*>")
    assert adapter.add_log("x")["change_type"] == "NONE"  # safe fallback


def test_tokens_fallback_to_template_split_when_cluster_missing() -> None:
    miner = FakeTemplateMiner()  # id_to_cluster empty
    adapter = DrainAdapter(miner)
    miner.queue_result("cluster_created", 7, "foo bar baz")
    out = adapter.add_log("foo bar baz")
    assert out["tokens"] == ["foo", "bar", "baz"]


def test_update_template_uses_int_key_and_writes_tuple() -> None:
    miner = FakeTemplateMiner()
    miner.drain.id_to_cluster[3] = FakeCluster(3, ["old"])
    adapter = DrainAdapter(miner)

    assert adapter.update_template("3", ["new", "tokens"]) is True   # str id -> int key
    stored = miner.drain.id_to_cluster[3].log_template_tokens
    assert stored == ("new", "tokens")
    assert isinstance(stored, tuple)  # drain3 expects a tuple


def test_update_template_missing_returns_false() -> None:
    adapter = DrainAdapter(FakeTemplateMiner())
    assert adapter.update_template("999", ["x"]) is False


def test_get_template_joins_tokens() -> None:
    miner = FakeTemplateMiner()
    miner.drain.id_to_cluster[4] = FakeCluster(4, ["a", "<*>", "b"])
    adapter = DrainAdapter(miner)
    assert adapter.get_template("4") == "a <*> b"
    assert adapter.get_template("999") is None  # numeric-but-absent


def test_non_numeric_id_is_treated_as_not_found() -> None:
    """Bad ids must return the documented miss value, not raise ValueError."""
    adapter = DrainAdapter(FakeTemplateMiner())
    assert adapter.get_template("nope") is None
    assert adapter.update_template("nope", ["x"]) is False
    assert adapter.remove_cluster("nope") is False


def test_remove_cluster() -> None:
    miner = FakeTemplateMiner()
    miner.drain.id_to_cluster[5] = FakeCluster(5, ["a"])
    adapter = DrainAdapter(miner)
    assert adapter.remove_cluster("5") is True
    assert 5 not in miner.drain.id_to_cluster
    assert adapter.remove_cluster("5") is False  # already gone


def test_all_clusters_listing() -> None:
    miner = FakeTemplateMiner()
    miner.drain.id_to_cluster[1] = FakeCluster(1, ["a", "<*>"])
    miner.drain.id_to_cluster[2] = FakeCluster(2, ["b", "c", "d"])
    adapter = DrainAdapter(miner)
    out = {c["cluster_id"]: c for c in adapter.all_clusters()}
    assert out["1"]["template"] == "a <*>"
    assert out["1"]["token_count"] == 2
    assert out["2"]["token_count"] == 3
