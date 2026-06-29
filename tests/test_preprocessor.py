"""8 tests for LogPreprocessor."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessor import LogPreprocessor

pp = LogPreprocessor()


def test_ipv4_replaced():
    r = pp.process("Connection from 192.168.1.10 accepted")
    assert "<IP>" in r.processed
    assert "192.168.1.10" not in r.processed
    assert "192.168.1.10" in r.extractions.get("IP", [])


def test_apache_timestamp_replaced():
    r = pp.process("[Sun Dec 04 04:47:44 2005] request received")
    assert "[<TIMESTAMP>]" in r.processed
    assert "Sun Dec 04" not in r.processed


def test_iso_datetime_replaced():
    r = pp.process("2024-01-15T10:30:00Z user logged in")
    assert "<DATETIME>" in r.processed
    assert "2024-01-15" not in r.processed


def test_port_context_replaced():
    r = pp.process("SSH connection on port 22 accepted")
    assert "port <PORT>" in r.processed
    assert "port 22" not in r.processed


def test_file_path_replaced():
    r = pp.process("Reading config from /etc/nginx/nginx.conf failed")
    assert "<PATH>" in r.processed
    assert "/etc/nginx/nginx.conf" not in r.processed


def test_uuid_replaced():
    r = pp.process("Request ID 550e8400-e29b-41d4-a716-446655440000 received")
    assert "<UUID>" in r.processed
    assert "550e8400" not in r.processed


def test_batch_processes_multiple():
    logs = ["error from 10.0.0.1", "pid 1234 started", "done"]
    results = pp.batch(logs)
    assert len(results) == 3
    assert "<IP>" in results[0].processed
    assert "pid <PID>" in results[1].processed
    assert results[2].processed == "done"


def test_never_raises_on_malformed():
    r = pp.process("")
    assert r.processed == ""
    assert r.original == ""
    r2 = pp.process(None)  # type: ignore
    assert r2.original is None or r2.processed is not None


def test_username_after_for_keyword():
    r = pp.process("Accepted password for alice from 192.168.1.10")
    assert r.processed == "Accepted password for <USERNAME> from <IP>"
    assert r.extractions.get("USERNAME") == ["alice"]


def test_username_after_user_keyword():
    r = pp.process("authentication failed for user root attempt=3")
    assert "user <USERNAME>" in r.processed
    assert "root" not in r.processed
    assert r.extractions.get("USERNAME") == ["root"]


def test_password_value_masked():
    r = pp.process("password=secretvalue123")
    assert r.processed == "password <PASSWORD>"
