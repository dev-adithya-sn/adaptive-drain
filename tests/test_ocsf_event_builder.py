"""Tests for OCSFEventBuilder — 8 cases covering all enrichers."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocsf_event_builder import OCSFEventBuilder

builder = OCSFEventBuilder()

SSH_LOGON_LABEL = {
    "ocsf_class_uid": 3002, "ocsf_class_name": "SSH Activity",
    "activity_id": 1, "activity_name": "Logon",
    "severity_id": 1, "category_uid": 3, "category_name": "Identity & Access Management",
}
SSH_FAIL_LABEL = dict(SSH_LOGON_LABEL, severity_id=4)
HTTP_GET_LABEL = {
    "ocsf_class_uid": 4002, "ocsf_class_name": "HTTP Activity",
    "activity_id": 1, "activity_name": "GET",
    "severity_id": 1, "category_uid": 4, "category_name": "Network Activity",
}
HTTP_ERR_LABEL = dict(HTTP_GET_LABEL, severity_id=4)
DB_OPEN_LABEL = {
    "ocsf_class_uid": 5001, "ocsf_class_name": "Datastore Activity",
    "activity_id": 1, "activity_name": "Open",
    "severity_id": 1, "category_uid": 5, "category_name": "Discovery",
}
DB_QUERY_LABEL = dict(DB_OPEN_LABEL, activity_id=3, activity_name="Query")
SVC_LABEL = {
    "ocsf_class_uid": 1007, "ocsf_class_name": "Application Lifecycle",
    "activity_id": 1, "activity_name": "Install",
    "severity_id": 1, "category_uid": 1, "category_name": "System Activity",
}


def test_ssh_logon_success_type_uid():
    ev = builder.build(
        "Accepted password for alice from 10.0.0.1 port 22 ssh2",
        SSH_LOGON_LABEL,
    )
    # type_uid = class_uid * 100 + activity_id = 3002 * 100 + 1 = 300201
    assert ev["type_uid"] == 300201


def test_ssh_failure_sets_status_id_2():
    ev = builder.build(
        "Failed password for bob from 192.168.1.5 port 44122 ssh2",
        SSH_FAIL_LABEL,
    )
    assert ev["status_id"] == 2
    assert ev["status"] == "Failure"


def test_ssh_extracts_user_and_src_ip():
    ev = builder.build(
        "Accepted publickey for deploy from 203.0.113.10 port 22 ssh2",
        SSH_LOGON_LABEL,
    )
    assert ev.get("user", {}).get("name") == "deploy"
    assert ev.get("src_endpoint", {}).get("ip") == "203.0.113.10"
    obs_types = {o["name"] for o in ev["observables"]}
    assert "user.name" in obs_types
    assert "src_endpoint.ip" in obs_types


def test_http_get_extracts_method_path_status():
    raw = '127.0.0.1 - - [01/Jan/2024] "GET /api/health HTTP/1.1" 200 42'
    ev = builder.build(raw, HTTP_GET_LABEL)
    assert ev.get("http_request", {}).get("http_method") == "GET"
    assert ev.get("http_request", {}).get("url", {}).get("path") == "/api/health"
    assert ev.get("http_response", {}).get("code") == 200
    assert ev["status_id"] == 1


def test_http_500_sets_status_failure_and_severity():
    raw = '10.1.1.1 - - "POST /submit HTTP/1.1" 500 0'
    ev = builder.build(raw, HTTP_ERR_LABEL)
    assert ev.get("http_response", {}).get("code") == 500
    assert ev["status_id"] == 2
    assert ev["status"] == "Failure"
    assert ev["severity_id"] == 4


def test_db_open_extracts_host_port_dbname():
    raw = "connecting to database db=myapp host=db.internal port=5432 user=app"
    ev = builder.build(raw, DB_OPEN_LABEL)
    assert ev.get("dst_endpoint", {}).get("hostname") == "db.internal"
    assert ev.get("dst_endpoint", {}).get("port") == 5432
    assert ev.get("database", {}).get("name") == "myapp"


def test_db_query_extracts_table_rows_duration():
    raw = "execute query table=users rows=150 duration=23 db=myapp"
    ev = builder.build(raw, DB_QUERY_LABEL)
    assert ev.get("database", {}).get("table") == "users"
    assert ev.get("affected_rows") == 150
    assert ev.get("duration") == 23


def test_service_start_extracts_app_name():
    raw = "Started service nginx successfully"
    ev = builder.build(raw, SVC_LABEL)
    assert ev.get("app", {}).get("name") == "nginx"
    obs_names = [o["name"] for o in ev["observables"]]
    assert "app.name" in obs_names
