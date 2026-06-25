"""OCSF normalization: map a template string to OCSF event fields via YAML rules."""

from __future__ import annotations

import re

import yaml

from ocsf_event_builder import OCSFEventBuilder

_REQUIRED_FIELDS = (
    "ocsf_class_uid",
    "ocsf_class_name",
    "activity_id",
    "activity_name",
    "severity_id",
    "category_uid",
    "category_name",
)


class OCSFNormalizer:
    """Resolves the first matching OCSF rule for a template string.

    Rules are loaded from a YAML file and recompiled on construction (and on
    ``reload()``), so the rule set can be hot-updated without a restart.
    """

    def __init__(self, yaml_path: str) -> None:
        self._yaml_path = yaml_path
        # Each entry: (compiled_regex, pattern_str, field_dict)
        self._rules: list[tuple[re.Pattern, str, dict]] = []
        self._builder = OCSFEventBuilder()
        self._load()

    def _load(self) -> None:
        with open(self._yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        rules = data.get("rules", []) or []
        compiled: list[tuple[re.Pattern, str, dict]] = []
        for rule in rules:
            pattern = rule["pattern"]
            fields = {k: rule[k] for k in _REQUIRED_FIELDS}
            compiled.append((re.compile(pattern, re.IGNORECASE), pattern, fields))
        self._rules = compiled

    def normalize(self, template: str) -> dict | None:
        """Return OCSF fields for the first matching rule, or None if none match."""
        for regex, pattern, fields in self._rules:
            if regex.search(template):
                result = dict(fields)
                result["matched_rule"] = pattern
                return result
        return None

    def normalize_full(self, raw_log: str, template: str) -> dict | None:
        """Return a full OCSF 1.1 compliant event dict, or None if no rule matches."""
        label = self.normalize(template)
        if label is None:
            return None
        return self._builder.build(raw_log, label)

    def reload(self) -> None:
        """Re-read the YAML file and recompile all rules."""
        self._load()

    def stats(self) -> dict:
        """Return rule count and the source path."""
        return {"rules_loaded": len(self._rules), "yaml_path": self._yaml_path}
