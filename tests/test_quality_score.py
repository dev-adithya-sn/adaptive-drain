"""4 tests for quality scoring fields on ManagedTemplate + persistence + server."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template_store import ManagedTemplate, TemplateStore
from persistence import StatePersistence
from reservoir_sampler import ReservoirSampler


def _store_with_quality():
    store = TemplateStore()
    store.register("c1", "user <*> logged in")
    t = store.get("c1")
    t.quality_score      = 7
    t.quality_issues     = ["missing_field_labels"]
    t.quality_suggestion = "Add semantic labels to wildcards"
    return store


def test_quality_fields_default_to_none():
    t = ManagedTemplate(cluster_id="x", pattern="test <*>")
    assert t.quality_score is None
    assert t.quality_issues == []
    assert t.quality_suggestion is None


def test_quality_fields_saved_and_loaded():
    store = _store_with_quality()
    sampler = ReservoirSampler()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    sp = StatePersistence(path)
    assert sp.save(store, sampler)

    store2 = TemplateStore()
    sampler2 = ReservoirSampler()
    assert sp.load(store2, sampler2)
    t2 = store2.get("c1")
    assert t2.quality_score == 7
    assert "missing_field_labels" in t2.quality_issues
    assert t2.quality_suggestion == "Add semantic labels to wildcards"


def test_quality_score_range():
    t = ManagedTemplate(cluster_id="y", pattern="test")
    for score in [0, 5, 10]:
        t.quality_score = score
        assert t.quality_score == score


def test_quality_fields_in_saved_json():
    store = _store_with_quality()
    sampler = ReservoirSampler()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    StatePersistence(path).save(store, sampler)
    with open(path) as f:
        payload = json.load(f)
    tmpl = payload["templates"][0]
    assert tmpl["quality_score"] == 7
    assert tmpl["quality_issues"] == ["missing_field_labels"]
    assert tmpl["quality_suggestion"] == "Add semantic labels to wildcards"
