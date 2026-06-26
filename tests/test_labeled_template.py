"""Tests for LLM wildcard labeling: validate_labeled_template, _apply_labeled_template,
and ManagedTemplate.labeled_template field."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_gate import LLMGate
from template_store import ManagedTemplate

gate = LLMGate(api_key="test")


def test_validate_valid_labeling():
    assert gate.validate_labeled_template(
        "Accepted <*> for <*> from <*> port <*> ssh2",
        "Accepted <auth_method> for <user> from <ip> port <port> ssh2",
    ) is True


def test_validate_token_count_mismatch():
    assert gate.validate_labeled_template(
        "connect from <*> port <*>",
        "connect from <ip>",  # missing one token
    ) is False


def test_validate_non_wildcard_token_changed():
    assert gate.validate_labeled_template(
        "Accepted <*> for <*>",
        "Rejected <auth_method> for <user>",  # "Accepted" → "Rejected"
    ) is False


def test_validate_label_not_in_vocab():
    assert gate.validate_labeled_template(
        "user <*> logged in",
        "user <username123> logged in",  # not in vocab
    ) is False


def test_apply_falls_back_on_invalid_output():
    decision = {
        "decision": "keep",
        "reasoning": "ok",
        "labeled_template": "user BOGUS logged in",  # invalid
    }
    original = "user <*> logged in"
    result = gate._apply_labeled_template(decision, original)
    assert result["labeled_template"] == "user <unknown> logged in"


def test_apply_accepts_valid_labeled_template():
    decision = {
        "decision": "keep",
        "reasoning": "ok",
        "labeled_template": "user <username> logged in",
    }
    original = "user <*> logged in"
    result = gate._apply_labeled_template(decision, original)
    assert result["labeled_template"] == "user <username> logged in"


def test_managed_template_labeled_defaults_to_none():
    t = ManagedTemplate(cluster_id="1", pattern="test <*>")
    assert t.labeled_template is None


def test_managed_template_labeled_can_be_set():
    t = ManagedTemplate(cluster_id="1", pattern="test <*>")
    t.labeled_template = "test <id>"
    assert t.labeled_template == "test <id>"
