"""Build fully compliant OCSF 1.1 event dicts from raw log + normalizer label."""

import re
import time
import uuid


class OCSFEventBuilder:
    """Constructs OCSF 1.1 compliant event dicts from raw log + normalizer output."""

    PRODUCT = {
        "name": "AdaptiveDrain",
        "vendor_name": "AdaptiveDrain",
        "version": "1.0.0",
    }
    OCSF_VERSION = "1.1.0"

    _IP_RE          = re.compile(r'\b(\d{1,3}(?:\.\d{1,3}){3})\b')
    _PORT_RE        = re.compile(r'\bport[= ](\d+)', re.IGNORECASE)
    _USER_RE        = re.compile(r'(?:for(?: invalid user)?|user[= ])\s+(\S+)', re.IGNORECASE)
    _HTTP_METHOD_RE = re.compile(r'\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b')
    # Accept optional closing quote between HTTP version and status (access log: HTTP/1.1" 200)
    _HTTP_STATUS_RE = re.compile(r'\bHTTP/[\d.]+"?\s+(\d{3})\b|\bstatus[= ](\d{3})\b', re.IGNORECASE)
    _HTTP_PATH_RE   = re.compile(r'(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) (/\S*)', re.IGNORECASE)
    _HTTP_BYTES_RE  = re.compile(r'HTTP/[\d.]+"?\s+\d{3}\s+(\d+)')
    _DB_HOST_RE     = re.compile(r'host[= ](\S+?)(?:\s|,|$)', re.IGNORECASE)
    _DB_PORT_RE     = re.compile(r'port[= ](\d+)', re.IGNORECASE)
    _DB_USER_RE     = re.compile(r'user[= ](\S+?)(?:\s|,|$)', re.IGNORECASE)
    # Require = separator so "database db=myapp" matches on "db=myapp" not "database "
    _DB_NAME_RE     = re.compile(r'(?:db|database)=(\S+?)(?:\s|,|$)', re.IGNORECASE)
    _DB_TABLE_RE    = re.compile(r'table[= ](\S+?)(?:\s|,|$)', re.IGNORECASE)
    _DB_ROWS_RE     = re.compile(r'rows[= ](\d+)', re.IGNORECASE)
    _DB_DURATION_RE = re.compile(r'duration[= ](\d+)', re.IGNORECASE)
    _SERVICE_RE     = re.compile(r'service\s+(\S+)', re.IGNORECASE)

    _SEVERITY_MAP = {
        0: "Unknown", 1: "Informational", 2: "Low",
        3: "Medium",  4: "High",          5: "Critical",
    }
    _STATUS_MAP = {
        "2": (1, "Success"), "3": (1, "Success"),
        "4": (2, "Failure"), "5": (2, "Failure"),
    }

    def build(self, raw_log: str, label: dict) -> dict:
        """Build a full OCSF 1.1 event from a raw log line and normalizer label."""
        try:
            class_uid   = label["ocsf_class_uid"]
            activity_id = label["activity_id"]
            severity_id = label["severity_id"]

            event = {
                "class_uid":     class_uid,
                "class_name":    label["ocsf_class_name"],
                "activity_id":   activity_id,
                "activity_name": label["activity_name"],
                "category_uid":  label["category_uid"],
                "category_name": label["category_name"],
                "severity_id":   severity_id,
                "severity":      self._SEVERITY_MAP.get(severity_id, "Unknown"),
                "type_uid":      class_uid * 100 + activity_id,
                "type_name":     f"{label['ocsf_class_name']}: {label['activity_name']}",
                "time":          int(time.time() * 1000),
                "message":       raw_log,
                "raw_data":      raw_log,
                "metadata": {
                    "version":     self.OCSF_VERSION,
                    "product":     self.PRODUCT,
                    "uid":         str(uuid.uuid4()),
                    "logged_time": int(time.time() * 1000),
                },
                "observables": [],
            }

            if class_uid == 3002:
                self._enrich_ssh_auth(raw_log, label, event)
            elif class_uid == 4002:
                self._enrich_http(raw_log, label, event)
            elif class_uid == 5001:
                self._enrich_database(raw_log, label, event)
            elif class_uid == 1007:
                self._enrich_service(raw_log, label, event)

            return event
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Class-specific enrichers
    # ------------------------------------------------------------------

    def _enrich_ssh_auth(self, raw: str, label: dict, event: dict) -> None:
        """OCSF 3002 — SSH Activity / Authentication"""
        try:
            ips    = self._IP_RE.findall(raw)
            port_m = self._PORT_RE.search(raw)
            user_m = self._USER_RE.search(raw)

            is_failure = label["severity_id"] >= 4
            event["status_id"] = 2 if is_failure else 1
            event["status"]    = "Failure" if is_failure else "Success"
            event["auth_protocol_id"] = 2
            event["auth_protocol"]    = "SSH"

            if user_m:
                username = user_m.group(1)
                event["user"] = {"name": username, "type_id": 1, "type": "User"}
                event["observables"].append({
                    "name": "user.name", "type_id": 4,
                    "type": "User Name", "value": username,
                })

            if ips:
                event["src_endpoint"] = {"ip": ips[0]}
                event["observables"].append({
                    "name": "src_endpoint.ip", "type_id": 2,
                    "type": "IP Address", "value": ips[0],
                })
                dst: dict = {"port": int(port_m.group(1)) if port_m else 22}
                if len(ips) > 1:
                    dst["ip"] = ips[1]
                event["dst_endpoint"] = dst

            if "publickey" in raw.lower():
                event["auth_protocol_id"] = 99
                event["auth_protocol"]    = "Public Key"
            elif "password" in raw.lower():
                event["auth_protocol_id"] = 2
                event["auth_protocol"]    = "Password"
        except Exception:
            pass

    def _enrich_http(self, raw: str, label: dict, event: dict) -> None:
        """OCSF 4002 — HTTP Activity"""
        try:
            method_m = self._HTTP_METHOD_RE.search(raw)
            status_m = self._HTTP_STATUS_RE.search(raw)
            path_m   = self._HTTP_PATH_RE.search(raw)
            bytes_m  = self._HTTP_BYTES_RE.search(raw)
            ips      = self._IP_RE.findall(raw)

            http_request: dict  = {}
            http_response: dict = {}

            if method_m:
                http_request["http_method"] = method_m.group(1)

            if path_m:
                http_request["url"] = {"path": path_m.group(1), "scheme": "http"}
                event["observables"].append({
                    "name": "http_request.url.path", "type_id": 6,
                    "type": "URL String", "value": path_m.group(1),
                })

            if status_m:
                code = int(status_m.group(1) or status_m.group(2))
                http_response["code"] = code
                sid, sname = self._STATUS_MAP.get(str(code)[0], (0, "Unknown"))
                event["status_id"] = sid
                event["status"]    = sname
                event["observables"].append({
                    "name": "http_response.code", "type_id": 12,
                    "type": "HTTP Response Code", "value": str(code),
                })
            else:
                event["status_id"] = 0
                event["status"]    = "Unknown"

            if bytes_m:
                http_response["length"] = int(bytes_m.group(1))

            if http_request:
                event["http_request"] = http_request
            if http_response:
                event["http_response"] = http_response

            if ips:
                event["src_endpoint"] = {"ip": ips[0]}
                event["observables"].append({
                    "name": "src_endpoint.ip", "type_id": 2,
                    "type": "IP Address", "value": ips[0],
                })
            if len(ips) > 1:
                event["dst_endpoint"] = {"ip": ips[1]}
        except Exception:
            pass

    def _enrich_database(self, raw: str, label: dict, event: dict) -> None:
        """OCSF 5001 — Datastore Activity"""
        try:
            host_m     = self._DB_HOST_RE.search(raw)
            port_m     = self._DB_PORT_RE.search(raw)
            user_m     = self._DB_USER_RE.search(raw)
            db_m       = self._DB_NAME_RE.search(raw)
            table_m    = self._DB_TABLE_RE.search(raw)
            rows_m     = self._DB_ROWS_RE.search(raw)
            duration_m = self._DB_DURATION_RE.search(raw)

            event["status_id"] = 1
            event["status"]    = "Success"

            database: dict = {}
            if db_m:
                database["name"] = db_m.group(1)
            if table_m:
                database["table"] = table_m.group(1)
                event["observables"].append({
                    "name": "database.table", "type_id": 26,
                    "type": "Table Name", "value": table_m.group(1),
                })
            if database:
                event["database"] = database

            dst_endpoint: dict = {}
            if host_m:
                dst_endpoint["hostname"] = host_m.group(1)
            if port_m:
                dst_endpoint["port"] = int(port_m.group(1))
            if dst_endpoint:
                event["dst_endpoint"] = dst_endpoint

            if user_m:
                event["actor"] = {"user": {"name": user_m.group(1), "type_id": 1}}

            if rows_m:
                event["affected_rows"] = int(rows_m.group(1))
            if duration_m:
                event["duration"] = int(duration_m.group(1))
        except Exception:
            pass

    def _enrich_service(self, raw: str, label: dict, event: dict) -> None:
        """OCSF 1007 — Application Lifecycle"""
        try:
            service_m = self._SERVICE_RE.search(raw)

            event["status_id"] = 1
            event["status"]    = "Success"

            if service_m:
                svc_name = service_m.group(1)
                event["app"]     = {"name": svc_name}
                event["service"] = {"name": svc_name}
                event["observables"].append({
                    "name": "app.name", "type_id": 7,
                    "type": "Resource Name", "value": svc_name,
                })
        except Exception:
            pass
