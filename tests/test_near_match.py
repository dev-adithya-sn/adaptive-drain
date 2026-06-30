"""Tests for NearMatchQueue status transitions and LLM decision types."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template_store import NearMatchItem, NearMatchQueue, TemplateStatus


# ---------------------------------------------------------------------------
# TemplateStatus enum
# ---------------------------------------------------------------------------

class TestTemplateStatusEnum:

    def test_near_match_review_value(self):
        assert TemplateStatus.NEAR_MATCH_REVIEW.value == "NEAR_MATCH_REVIEW"

    def test_near_match_review_in_all_statuses(self):
        values = {s.value for s in TemplateStatus}
        assert "NEAR_MATCH_REVIEW" in values

    def test_existing_statuses_unchanged(self):
        assert TemplateStatus.ACTIVE.value == "ACTIVE"
        assert TemplateStatus.PENDING_MERGE.value == "PENDING_MERGE"
        assert TemplateStatus.MERGED.value == "MERGED"


# ---------------------------------------------------------------------------
# NearMatchQueue — basic operations
# ---------------------------------------------------------------------------

class TestNearMatchQueueBasic:

    def test_add_returns_item_id(self):
        q = NearMatchQueue()
        iid = q.add("log line", "cluster-1", 0.92)
        assert isinstance(iid, str) and len(iid) > 0

    def test_get_pending_contains_new_item(self):
        q = NearMatchQueue()
        iid = q.add("log line", "cluster-1", 0.92)
        pending = q.get_pending()
        ids = [i.item_id for i in pending]
        assert iid in ids

    def test_get_returns_correct_item(self):
        q = NearMatchQueue()
        iid = q.add("my log", "c1", 0.95)
        item = q.get(iid)
        assert item is not None
        assert item.raw_log == "my log"
        assert item.candidate_cluster_id == "c1"
        assert item.similarity_score == 0.95

    def test_get_unknown_id_returns_none(self):
        q = NearMatchQueue()
        assert q.get("nonexistent") is None

    def test_len(self):
        q = NearMatchQueue()
        q.add("a", "c1", 0.9)
        q.add("b", "c2", 0.8)
        assert len(q) == 2

    def test_custom_item_id(self):
        q = NearMatchQueue()
        iid = q.add("log", "c1", 0.9, item_id="my-custom-id")
        assert iid == "my-custom-id"
        assert q.get("my-custom-id") is not None


# ---------------------------------------------------------------------------
# Status transitions: pending → approved / rejected
# ---------------------------------------------------------------------------

class TestNearMatchQueueTransitions:

    def test_new_item_starts_pending(self):
        q = NearMatchQueue()
        iid = q.add("log", "c1", 0.9)
        assert q.get(iid).status == "pending"

    def test_approve_same_template_fix_regex(self):
        q = NearMatchQueue()
        iid = q.add("log", "c1", 0.9)
        ok = q.approve(iid, llm_decision="same_template_fix_regex",
                       corrected_template="Accepted <auth_protocol> for <user.name> ssh2")
        assert ok is True
        item = q.get(iid)
        assert item.status == "approved"
        assert item.llm_decision == "same_template_fix_regex"
        assert item.corrected_template == "Accepted <auth_protocol> for <user.name> ssh2"

    def test_approve_new_template_send_to_drain(self):
        q = NearMatchQueue()
        iid = q.add("log", "c1", 0.9)
        ok = q.approve(iid, llm_decision="new_template_send_to_drain")
        assert ok is True
        item = q.get(iid)
        assert item.status == "approved"
        assert item.llm_decision == "new_template_send_to_drain"
        assert item.corrected_template is None

    def test_reject(self):
        q = NearMatchQueue()
        iid = q.add("log", "c1", 0.9)
        ok = q.reject(iid)
        assert ok is True
        assert q.get(iid).status == "rejected"

    def test_get_pending_excludes_approved(self):
        q = NearMatchQueue()
        iid1 = q.add("log1", "c1", 0.9)
        iid2 = q.add("log2", "c1", 0.8)
        q.approve(iid1, llm_decision="same_template_fix_regex")
        pending = [i.item_id for i in q.get_pending()]
        assert iid1 not in pending
        assert iid2 in pending

    def test_get_pending_excludes_rejected(self):
        q = NearMatchQueue()
        iid1 = q.add("log1", "c1", 0.9)
        iid2 = q.add("log2", "c1", 0.8)
        q.reject(iid1)
        pending = [i.item_id for i in q.get_pending()]
        assert iid1 not in pending
        assert iid2 in pending

    def test_remove(self):
        q = NearMatchQueue()
        iid = q.add("log", "c1", 0.9)
        ok = q.remove(iid)
        assert ok is True
        assert q.get(iid) is None
        assert len(q) == 0

    def test_remove_nonexistent_returns_false(self):
        q = NearMatchQueue()
        assert q.remove("no-such-id") is False

    def test_clear(self):
        q = NearMatchQueue()
        q.add("a", "c1", 0.9)
        q.add("b", "c2", 0.8)
        q.clear()
        assert len(q) == 0


# ---------------------------------------------------------------------------
# Pipeline integration: near-match queue + execute_near_match_decision
# ---------------------------------------------------------------------------

class TestPipelineNearMatchDecisionExecution:

    def _minimal_pipeline(self):
        """Build a TemplatePipeline with mocked drain and a primed SSH template."""
        try:
            from pipeline import TemplatePipeline
        except ImportError:
            import pytest; pytest.skip("pipeline not importable")
        from unittest.mock import MagicMock
        from template_store import TemplateStore

        drain_mock = MagicMock()
        drain_mock.add_log_message.return_value = MagicMock(
            cluster_id="1",
            get_template_str=lambda: "Accepted <auth_protocol> for <user.name> ssh2",
        )

        pipeline = TemplatePipeline(
            drain_instance    = drain_mock,
            openrouter_api_key = "test-key",
        )
        pipeline.store.register("1", "Accepted <auth_protocol> for <user.name> ssh2")
        t = pipeline.store.get("1")
        t.labeled_template = "Accepted <auth_protocol> for <user.name> from <src_endpoint.ip> port <dst_endpoint.port> ssh2"
        t.llm_decision     = "keep"
        pipeline._compiled_registry.update("1", t.labeled_template)
        return pipeline

    def test_same_template_fix_regex_recompiles_registry(self):
        pipeline = self._minimal_pipeline()
        iid = pipeline._near_match_queue.add(
            "Accepted password for alice from 10.0.0.1 extra port 22 ssh2", "1", 0.9
        )
        corrected = "Accepted <auth_protocol> for <user.name> from <src_endpoint.ip> <unknown> port <dst_endpoint.port> ssh2"
        pipeline._execute_near_match_decision({
            "item_id":            iid,
            "decision":           "same_template_fix_regex",
            "corrected_template": corrected,
        })
        # Template should have been updated in the store and registry
        updated_t = pipeline.store.get("1")
        assert updated_t.labeled_template == corrected
        # Registry should now use the corrected template
        ct = pipeline._compiled_registry.get("1")
        assert ct is not None

    def test_new_template_send_to_drain_routes_log(self):
        pipeline = self._minimal_pipeline()
        from unittest.mock import MagicMock

        # Set up drain_adapter.add_log to return a new-cluster result
        new_drain_result = {
            "cluster_id": "99", "template": "some <*> new template", "change_type": "CREATE"
        }
        pipeline.drain_adapter.add_log = MagicMock(return_value=new_drain_result)

        iid = pipeline._near_match_queue.add("some brand new log entry", "1", 0.88)
        pipeline._execute_near_match_decision({
            "item_id":  iid,
            "decision": "new_template_send_to_drain",
        })

        # drain_adapter.add_log must have been called with the preprocessed log
        pipeline.drain_adapter.add_log.assert_called_once()
        # Item should be removed from the queue after execution
        assert pipeline._near_match_queue.get(iid) is None
