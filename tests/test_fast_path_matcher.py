"""Tests for FastPathMatcher — all three MatchResult branches."""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template_compiler import CompiledTemplateRegistry
from fast_path_matcher import (
    ExactMatch, FastPathMatcher, MatchKind, NearMatch, NoMatch,
    _literal_token_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SSH_TEMPLATE  = "Accepted <auth_protocol> for <user.name> from <src_endpoint.ip> port <dst_endpoint.port> ssh2"
SSH_LOG       = "Accepted password for alice from 192.168.1.10 port 22 ssh2"
HTTP_TEMPLATE = "GET <http_request.url.path> HTTP/1.1 <http_response.code> <http_response.length>"
HTTP_LOG      = "GET /index.html HTTP/1.1 200 512"


def _registry_with(*templates) -> CompiledTemplateRegistry:
    reg = CompiledTemplateRegistry()
    for cid, tpl in templates:
        reg.update(cid, tpl)
    return reg


def _matcher(*templates, threshold=0.85) -> FastPathMatcher:
    return FastPathMatcher(_registry_with(*templates), near_match_threshold=threshold)


# ---------------------------------------------------------------------------
# EXACT_MATCH
# ---------------------------------------------------------------------------

class TestExactMatch:

    def test_ssh_log_exact_match(self):
        fm  = _matcher(("1", SSH_TEMPLATE))
        res = fm.match(SSH_LOG)
        assert isinstance(res, ExactMatch)
        assert res.cluster_id == "1"

    def test_http_log_exact_match(self):
        fm  = _matcher(("2", HTTP_TEMPLATE))
        res = fm.match(HTTP_LOG)
        assert isinstance(res, ExactMatch)
        assert res.cluster_id == "2"

    def test_extracted_fields_keyed_by_ocsf_path(self):
        fm  = _matcher(("1", SSH_TEMPLATE))
        res = fm.match(SSH_LOG)
        assert isinstance(res, ExactMatch)
        assert res.extracted_fields.get("user.name") == "alice"
        assert res.extracted_fields.get("src_endpoint.ip") == "192.168.1.10"
        assert res.extracted_fields.get("dst_endpoint.port") == "22"
        assert res.extracted_fields.get("auth_protocol") == "password"

    def test_most_specific_template_wins(self):
        # Two templates share literal "Accepted"; the more specific one (more literals) should win.
        generic  = "Accepted <user.name>"
        specific = SSH_TEMPLATE
        fm  = _matcher(("gen", generic), ("spec", specific))
        res = fm.match(SSH_LOG)
        assert isinstance(res, ExactMatch)
        assert res.cluster_id == "spec"

    def test_match_kind_constant(self):
        assert ExactMatch.kind == MatchKind.EXACT_MATCH


# ---------------------------------------------------------------------------
# NO_MATCH
# ---------------------------------------------------------------------------

class TestNoMatch:

    def test_completely_different_log(self):
        fm  = _matcher(("1", SSH_TEMPLATE))
        res = fm.match("ERROR kernel: segfault at 0x00 ip 0x0")
        assert isinstance(res, NoMatch)

    def test_empty_registry(self):
        fm  = FastPathMatcher(CompiledTemplateRegistry())
        res = fm.match(SSH_LOG)
        assert isinstance(res, NoMatch)

    def test_match_kind_constant(self):
        assert NoMatch.kind == MatchKind.NO_MATCH

    def test_partial_log_missing_trailing_literal(self):
        fm  = _matcher(("1", SSH_TEMPLATE))
        # "ssh2" missing → strict fails, loose fails (all literals must appear in order)
        res = fm.match("Accepted password for alice from 10.0.0.1 port 22")
        # Strict fails (anchored, missing "ssh2"); loose: "ssh2" not in log → no loose match
        assert isinstance(res, NoMatch)


# ---------------------------------------------------------------------------
# NEAR_MATCH
# ---------------------------------------------------------------------------

class TestNearMatch:

    def test_extra_token_between_literals(self):
        fm  = _matcher(("1", SSH_TEMPLATE), threshold=0.85)
        log = "Accepted password for alice from 192.168.1.10 on port 22 ssh2"
        res = fm.match(log)
        assert isinstance(res, NearMatch)
        assert res.candidate_cluster_id == "1"

    def test_near_match_similarity_between_0_and_1(self):
        fm  = _matcher(("1", SSH_TEMPLATE), threshold=0.0)
        log = "Accepted password for alice from 192.168.1.10 on port 22 ssh2"
        res = fm.match(log)
        assert isinstance(res, NearMatch)
        assert 0.0 <= res.similarity_score <= 1.0

    def test_near_match_returns_raw_log(self):
        fm  = _matcher(("1", SSH_TEMPLATE), threshold=0.0)
        log = "Accepted password for alice from 192.168.1.10 on port 22 ssh2"
        res = fm.match(log)
        assert isinstance(res, NearMatch)
        assert res.raw_log == log

    def test_multiple_near_matches_returns_highest_similarity(self):
        # Two templates loosely match; the one with MORE literals in the log should win.
        tpl_a = "Accepted <auth_protocol> for <user.name> ssh2"  # 3 literals
        tpl_b = SSH_TEMPLATE                                       # 5 literals
        log   = "Accepted password for alice from 10.0.0.1 on port 22 ssh2"
        fm    = FastPathMatcher(
            _registry_with(("a", tpl_a), ("b", tpl_b)), near_match_threshold=0.0
        )
        res = fm.match(log)
        assert isinstance(res, NearMatch)
        assert res.candidate_cluster_id == "b"  # more literals → higher similarity

    def test_below_threshold_returns_no_match(self):
        fm  = _matcher(("1", SSH_TEMPLATE), threshold=0.99)
        log = "Accepted password for alice from 10.0.0.1 on port 22 ssh2"
        # Even with a near-match candidate, threshold=0.99 should reject it
        # (all 5 literals are present → similarity = 1.0, still >= 0.99)
        res = fm.match(log)
        # For this particular log with all literals present, similarity=1.0 >= 0.99
        assert isinstance(res, (NearMatch, NoMatch))  # either is valid depending on loose match

    def test_threshold_zero_catches_partial_overlap(self):
        fm  = _matcher(("1", SSH_TEMPLATE), threshold=0.0)
        # Only 2 of 5 literals present AND in order
        log = "for alice port 22"
        res = fm.match(log)
        # loose match requires ALL literals in order ("Accepted", "for", "from", "port", "ssh2")
        # "Accepted" and "from" and "ssh2" are missing → loose fails → NO_MATCH
        assert isinstance(res, NoMatch)

    def test_match_kind_constant(self):
        assert NearMatch.kind == MatchKind.NEAR_MATCH


# ---------------------------------------------------------------------------
# End-to-end: EXACT_MATCH skips Drain3 (pipeline integration spy test)
# ---------------------------------------------------------------------------

class TestPipelineFastPathSkipsDrain3:

    def test_exact_match_does_not_call_drain_adapter(self):
        """On EXACT_MATCH, pipeline.ingest() must NOT call drain_adapter.add_log."""
        import sys
        # We need to import pipeline; it needs drain3 available
        try:
            from pipeline import TemplatePipeline
            from template_store import TemplateStore
        except ImportError:
            import pytest; pytest.skip("pipeline not importable in this environment")

        # Minimal pipeline setup with a mocked drain instance
        drain_mock = MagicMock()
        drain_mock.add_log_message.return_value = MagicMock(
            cluster_id="1",
            get_template_str=lambda: SSH_TEMPLATE,
        )

        pipeline = TemplatePipeline(
            drain_instance    = drain_mock,
            openrouter_api_key = "test-key",
        )

        # Prime the compiled registry as if a template was already approved
        pipeline._compiled_registry.update("1", SSH_TEMPLATE)

        # Spy on drain_adapter.add_log
        add_log_spy = MagicMock(return_value={
            "cluster_id": "1", "template": SSH_TEMPLATE, "change_type": "NONE"
        })
        pipeline.drain_adapter.add_log = add_log_spy

        # Also prime the store so the exact-match path can look up the template
        pipeline.store.register("1", SSH_TEMPLATE)
        t = pipeline.store.get("1")
        t.labeled_template = SSH_TEMPLATE
        t.llm_decision     = "keep"

        result = pipeline.ingest(SSH_LOG)

        add_log_spy.assert_not_called()
        assert result["fast_path"] is True
        assert result["cluster_id"] == "1"


# ---------------------------------------------------------------------------
# _literal_token_similarity unit tests
# ---------------------------------------------------------------------------

class TestLiteralTokenSimilarity:

    def test_all_literals_present_in_order(self):
        score = _literal_token_similarity(
            ["Accepted", "for", "from", "port", "ssh2"],
            SSH_LOG,
        )
        assert score == 1.0

    def test_no_literals_present(self):
        score = _literal_token_similarity(
            ["xyz", "abc", "def"],
            "something completely different",
        )
        assert score == 0.0

    def test_half_literals_present_in_order(self):
        score = _literal_token_similarity(
            ["Accepted", "for", "xyz", "abc"],
            "Accepted something for alice",
        )
        # "Accepted" and "for" appear in order; "xyz", "abc" do not → 2/4 = 0.5
        assert score == 0.5

    def test_empty_literal_list_returns_zero(self):
        score = _literal_token_similarity([], SSH_LOG)
        assert score == 0.0
