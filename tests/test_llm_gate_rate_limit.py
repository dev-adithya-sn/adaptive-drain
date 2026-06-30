"""Tests for LLMGate 429 retry logic and semaphore concurrency cap."""

import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_gate import LLMGate


def _make_gate():
    return LLMGate(api_key="test-key")


def _429_response():
    r = MagicMock()
    r.status_code = 429
    r.headers = {}
    import requests
    http_err = requests.HTTPError(response=r)
    return http_err


def _ok_response(content: str):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def test_classify_retries_on_429_then_succeeds():
    """classify_template should retry up to 3 times on 429, succeeding on 2nd attempt."""
    gate = _make_gate()
    import requests

    ok_json = (
        '{"log_source":"sshd","vendor":"OpenSSH","product":"","log_source_confidence":80,'
        '"security_relevant":true,"telemetry_type":"auth","semantic_event":"SSH login",'
        '"event_description":"","severity_id":1,"ocsf_class_uid":3002,"ocsf_class_name":"SSH Activity",'
        '"category_uid":3,"category_name":"IAM","activity_id":1,"activity_name":"Logon",'
        '"regex_pattern":"","template_confidence":80,"recommended_index":"","storage_class":"hot",'
        '"detection_tags":[],"mitre_attack_techniques":[],"ioc_candidates":[],'
        '"anomaly_indicators":[],"entities":{},"fields":{}}'
    )

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise requests.HTTPError(response=MagicMock(status_code=429, headers={}))
        return _ok_response(ok_json)

    with patch("requests.post", side_effect=side_effect), \
         patch("time.sleep"):  # don't actually sleep in tests
        result = gate.classify_template("Accepted <*> for <*>", ["Accepted password for alice"])

    assert call_count == 2, f"expected 2 calls (1 retry), got {call_count}"
    assert result["ocsf_class_uid"] == 3002
    assert result["matched_rule"] == "llm_classified"


def test_classify_exhausts_retries_returns_fallback():
    """classify_template should return Unknown fallback after 3 consecutive 429s."""
    gate = _make_gate()
    import requests

    def always_429(*args, **kwargs):
        raise requests.HTTPError(response=MagicMock(status_code=429, headers={}))

    with patch("requests.post", side_effect=always_429), \
         patch("time.sleep"):
        result = gate.classify_template("some <*> template", ["sample log"])

    assert result["ocsf_class_name"] == "Unknown"
    assert result["ocsf_class_uid"] == 0


def test_classify_non_429_http_error_returns_fallback_immediately():
    """Non-429 HTTP errors should not retry — return fallback immediately."""
    gate = _make_gate()
    import requests

    call_count = 0

    def server_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise requests.HTTPError(response=MagicMock(status_code=500, headers={}))

    with patch("requests.post", side_effect=server_error), \
         patch("time.sleep"):
        result = gate.classify_template("some <*> template", ["sample log"])

    assert call_count == 1, "should not retry on 500"
    assert result["ocsf_class_name"] == "Unknown"


def test_classify_respects_retry_after_header():
    """classify_template should use Retry-After header value when larger than backoff."""
    gate = _make_gate()
    import requests

    sleep_calls = []

    def side_effect(*args, **kwargs):
        if len(sleep_calls) == 0:
            raise requests.HTTPError(
                response=MagicMock(status_code=429, headers={"Retry-After": "15"})
            )
        return _ok_response('{"ocsf_class_uid":0,"ocsf_class_name":"","category_uid":0,'
                             '"activity_id":0,"severity_id":1}')

    with patch("requests.post", side_effect=side_effect), \
         patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        gate.classify_template("t <*>", ["log"])

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 15  # Retry-After=15 > backoff=2, so 15 wins


def test_semaphore_limits_concurrency():
    """At most GROQ_CONCURRENCY requests should be in-flight simultaneously."""
    gate = _make_gate()
    assert gate.GROQ_CONCURRENCY == 2

    concurrent_high_water = 0
    active = 0
    lock = threading.Lock()

    import requests

    def slow_post(*args, **kwargs):
        nonlocal active, concurrent_high_water
        with lock:
            active += 1
            concurrent_high_water = max(concurrent_high_water, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"choices": [{"message": {"content": '{"templates":[]}'}}]}
        return r

    templates = [{"cluster_id": str(i), "template": f"t{i} <*>", "samples": [], "wildcard_ratio": 0.5}
                 for i in range(6)]

    with patch("requests.post", side_effect=slow_post):
        gate.call_batch(templates)

    assert concurrent_high_water <= gate.GROQ_CONCURRENCY, (
        f"concurrency {concurrent_high_water} exceeded semaphore limit {gate.GROQ_CONCURRENCY}"
    )
