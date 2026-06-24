"""OCSFNormalizer — rule matching, reload, case-insensitivity, stats."""

from __future__ import annotations

import os

import yaml

import adaptive_drain
from adaptive_drain.normalizer import OCSFNormalizer

YML = os.path.join(os.path.dirname(adaptive_drain.__file__), "ocsf_map.yml")


def test_ssh_success_matches() -> None:
    n = OCSFNormalizer(YML)
    out = n.normalize("sshd[2451]: Accepted password for root from 10.0.0.5 port 22")
    assert out is not None
    assert out["ocsf_class_uid"] == 3002
    assert out["activity_name"] == "Logon"
    assert "matched_rule" in out


def test_http_get_matches() -> None:
    n = OCSFNormalizer(YML)
    out = n.normalize('127.0.0.1 - - "GET /index.html HTTP/1.1" 200')
    assert out is not None
    assert out["ocsf_class_uid"] == 4002
    assert out["activity_name"] == "GET"


def test_no_match_returns_none() -> None:
    n = OCSFNormalizer(YML)
    assert n.normalize("lorem ipsum dolor sit amet 12345 zzz") is None


def test_case_insensitive_match() -> None:
    n = OCSFNormalizer(YML)
    out = n.normalize("SSHD: ACCEPTED PASSWORD FOR ADMIN")
    assert out is not None
    assert out["ocsf_class_uid"] == 3002


def test_reload_updates_rules(tmp_path) -> None:
    path = tmp_path / "rules.yml"
    one_rule = {
        "rules": [{
            "pattern": "alpha",
            "ocsf_class_uid": 1, "ocsf_class_name": "X",
            "activity_id": 1, "activity_name": "A",
            "severity_id": 1, "category_uid": 1, "category_name": "C",
        }]
    }
    path.write_text(yaml.safe_dump(one_rule))
    n = OCSFNormalizer(str(path))
    assert n.stats()["rules_loaded"] == 1
    assert n.normalize("alpha event") is not None
    assert n.normalize("beta event") is None

    two_rules = yaml.safe_load(yaml.safe_dump(one_rule))
    two_rules["rules"].append({
        "pattern": "beta",
        "ocsf_class_uid": 2, "ocsf_class_name": "Y",
        "activity_id": 2, "activity_name": "B",
        "severity_id": 2, "category_uid": 2, "category_name": "D",
    })
    path.write_text(yaml.safe_dump(two_rules))
    n.reload()
    assert n.stats()["rules_loaded"] == 2
    assert n.normalize("beta event") is not None


def test_stats_reports_correct_rule_count() -> None:
    n = OCSFNormalizer(YML)
    with open(YML, "r", encoding="utf-8") as fh:
        expected = len(yaml.safe_load(fh)["rules"])
    stats = n.stats()
    assert stats["rules_loaded"] == expected
    assert expected >= 12  # spec minimum
    assert stats["yaml_path"] == YML
