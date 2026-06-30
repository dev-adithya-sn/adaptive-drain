"""Tests for TemplateCompiler and CompiledTemplateRegistry."""

from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template_compiler import CompiledTemplate, CompiledTemplateRegistry, TemplateCompiler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SSH_TEMPLATE  = "Accepted <auth_protocol> for <user.name> from <src_endpoint.ip> port <dst_endpoint.port> ssh2"
SSH_LOG       = "Accepted password for alice from 192.168.1.10 port 22 ssh2"
HTTP_TEMPLATE = "GET <http_request.url.path> HTTP/1.1 <http_response.code> <http_response.length>"
HTTP_LOG      = "GET /api/v1/data HTTP/1.1 200 1234"


def _compiler() -> TemplateCompiler:
    return TemplateCompiler()


def _ssh() -> CompiledTemplate:
    return _compiler().compile("1", SSH_TEMPLATE)


def _http() -> CompiledTemplate:
    return _compiler().compile("2", HTTP_TEMPLATE)


# ---------------------------------------------------------------------------
# CompiledTemplate structure
# ---------------------------------------------------------------------------

class TestCompiledTemplateStructure:

    def test_token_count_only_literal_tokens(self):
        ct = _ssh()
        # "Accepted", "for", "from", "port", "ssh2" = 5 literals
        assert ct.token_count == 5

    def test_label_map_keys_are_valid_group_names(self):
        ct = _ssh()
        for key in ct.label_map:
            assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key), f"Invalid group name: {key!r}"

    def test_label_map_values_contain_ocsf_paths(self):
        ct = _ssh()
        values = list(ct.label_map.values())
        assert "auth_protocol" in values
        assert "user.name" in values
        assert "src_endpoint.ip" in values
        assert "dst_endpoint.port" in values

    def test_literal_tokens_list(self):
        ct = _ssh()
        assert ct.literal_tokens == ["Accepted", "for", "from", "port", "ssh2"]

    def test_cluster_id_stored(self):
        ct = _compiler().compile("cluster-99", SSH_TEMPLATE)
        assert ct.cluster_id == "cluster-99"

    def test_no_literal_template_token_count_zero(self):
        ct = _compiler().compile("x", "<user.name> <src_endpoint.ip>")
        assert ct.token_count == 0
        assert ct.literal_tokens == []


# ---------------------------------------------------------------------------
# strict_regex — exact field extraction
# ---------------------------------------------------------------------------

class TestStrictRegex:

    def test_exact_ssh_match(self):
        ct = _ssh()
        m  = ct.strict_regex.match(SSH_LOG)
        assert m is not None

    def test_exact_http_match(self):
        ct = _http()
        m  = ct.strict_regex.match(HTTP_LOG)
        assert m is not None

    def test_extracts_user_name(self):
        ct = _ssh()
        m  = ct.strict_regex.match(SSH_LOG)
        groups = m.groupdict()
        user_group = next(k for k, v in ct.label_map.items() if v == "user.name")
        assert groups[user_group] == "alice"

    def test_extracts_src_ip(self):
        ct = _ssh()
        m  = ct.strict_regex.match(SSH_LOG)
        groups = m.groupdict()
        ip_group = next(k for k, v in ct.label_map.items() if v == "src_endpoint.ip")
        assert groups[ip_group] == "192.168.1.10"

    def test_extracts_port(self):
        ct = _ssh()
        m  = ct.strict_regex.match(SSH_LOG)
        groups = m.groupdict()
        port_group = next(k for k, v in ct.label_map.items() if v == "dst_endpoint.port")
        assert groups[port_group] == "22"

    def test_extracts_http_path(self):
        ct = _http()
        m  = ct.strict_regex.match(HTTP_LOG)
        groups = m.groupdict()
        path_group = next(k for k, v in ct.label_map.items() if v == "http_request.url.path")
        assert groups[path_group] == "/api/v1/data"

    def test_does_not_match_different_log(self):
        ct = _ssh()
        m  = ct.strict_regex.match("ERROR something failed port 22")
        assert m is None

    def test_does_not_match_partial_log(self):
        ct = _ssh()
        m  = ct.strict_regex.match("Accepted password for alice from 10.0.0.1")
        assert m is None, "strict_regex is anchored; missing trailing literal must fail"

    def test_duplicate_label_slots_get_unique_group_names(self):
        # A template with the same OCSF field in two positions
        ct = _compiler().compile("dup", "<user.name> logged in as <user.name>")
        keys = list(ct.label_map.keys())
        assert len(keys) == len(set(keys)), "group names must be unique"
        m = ct.strict_regex.match("alice logged in as bob")
        assert m is not None


# ---------------------------------------------------------------------------
# loose_regex — near-match detection
# ---------------------------------------------------------------------------

class TestLooseRegex:

    def test_matches_ssh_log_with_extra_token(self):
        ct  = _ssh()
        log = "Accepted password for alice from 10.0.0.1 on port 22 ssh2"
        # strict would fail (extra "on"), loose should match
        assert ct.strict_regex.match(log) is None
        assert ct.loose_regex.search(log) is not None

    def test_loose_matches_core_ssh_log(self):
        ct = _ssh()
        assert ct.loose_regex.search(SSH_LOG) is not None

    def test_loose_does_not_match_unrelated_log(self):
        ct  = _ssh()
        log = "database connection error on host db01"
        # "for", "from", "port", "ssh2" are not all present in order
        assert ct.loose_regex.search(log) is None

    def test_no_literal_template_loose_never_matches(self):
        ct  = _compiler().compile("x", "<user.name> <src_endpoint.ip>")
        # Loose regex is "never match" when there are no literals
        assert ct.loose_regex.search("anything at all") is None


# ---------------------------------------------------------------------------
# CompiledTemplateRegistry
# ---------------------------------------------------------------------------

class TestCompiledTemplateRegistry:

    def test_update_and_retrieve(self):
        reg = CompiledTemplateRegistry()
        reg.update("1", SSH_TEMPLATE)
        ct = reg.get("1")
        assert ct is not None
        assert ct.cluster_id == "1"

    def test_get_all_sorted_by_token_count_descending(self):
        reg = CompiledTemplateRegistry()
        reg.update("a", "Hello <user.name>")            # 1 literal
        reg.update("b", SSH_TEMPLATE)                   # 5 literals
        reg.update("c", "GET <http_request.url.path>")  # 1 literal
        reg.update("d", "key1 key2 key3 <user.name>")  # 3 literals
        sorted_cts = reg.get_all_sorted()
        counts = [ct.token_count for ct in sorted_cts]
        assert counts == sorted(counts, reverse=True)

    def test_remove(self):
        reg = CompiledTemplateRegistry()
        reg.update("1", SSH_TEMPLATE)
        reg.remove("1")
        assert reg.get("1") is None

    def test_len(self):
        reg = CompiledTemplateRegistry()
        reg.update("1", SSH_TEMPLATE)
        reg.update("2", HTTP_TEMPLATE)
        assert len(reg) == 2

    def test_recompile_on_update(self):
        reg = CompiledTemplateRegistry()
        reg.update("1", SSH_TEMPLATE)
        old_ct = reg.get("1")
        reg.update("1", HTTP_TEMPLATE)
        new_ct = reg.get("1")
        assert old_ct is not new_ct
        assert new_ct.token_count != old_ct.token_count

    def test_rebuild_from_store_compiles_active_approved(self):
        """rebuild_from_store should compile templates that have llm_decision set."""
        from template_store import TemplateStore

        store = TemplateStore()
        store.register("1", SSH_TEMPLATE)
        t = store.get("1")
        t.labeled_template = SSH_TEMPLATE
        t.llm_decision     = "keep"

        reg = CompiledTemplateRegistry()
        n   = reg.rebuild_from_store(store)
        assert n == 1
        assert reg.get("1") is not None

    def test_rebuild_skips_templates_without_llm_decision(self):
        from template_store import TemplateStore

        store = TemplateStore()
        store.register("1", SSH_TEMPLATE)
        t = store.get("1")
        t.labeled_template = SSH_TEMPLATE
        t.llm_decision     = None  # not yet reviewed

        reg = CompiledTemplateRegistry()
        n   = reg.rebuild_from_store(store)
        assert n == 0
