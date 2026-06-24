"""Thin wrapper around drain3.TemplateMiner — all Drain3 mutations go through here."""

from __future__ import annotations
from typing import Any

_CHANGE_MAP = {
    "cluster_created": "CREATE",
    "cluster_template_changed": "UPDATE",
    "none": "NONE",
}


class DrainAdapter:
    """Wraps a drain3.TemplateMiner instance and normalises its API."""

    def __init__(self, drain_instance: Any) -> None:
        self._drain = drain_instance

    def add_log(self, raw_log: str) -> dict:
        """Process a raw log line through Drain3 and return a normalised result dict."""
        raw = self._drain.add_log_message(raw_log)
        change_type = _CHANGE_MAP.get(raw["change_type"], "NONE")
        cluster_id = str(raw["cluster_id"])
        template_str: str = raw["template_mined"]

        # Fetch token list from Drain3 internals
        cluster = self._drain.drain.id_to_cluster.get(raw["cluster_id"])
        tokens: list[str] = list(cluster.log_template_tokens) if cluster else template_str.split()

        return {
            "change_type": change_type,
            "cluster_id": cluster_id,
            "template": template_str,
            "tokens": tokens,
        }

    @staticmethod
    def _as_int(cluster_id: str) -> int | None:
        """Coerce a cluster_id to drain3's int key, or None if it isn't numeric.

        Drain3 always issues numeric ids, so a non-numeric id is simply 'not
        found' — callers treat None as a miss rather than crashing on bad input.
        """
        try:
            return int(cluster_id)
        except (TypeError, ValueError):
            return None

    def update_template(self, cluster_id: str, new_tokens: list[str]) -> bool:
        """Overwrite the token list for an existing cluster. Returns True on success."""
        id_int = self._as_int(cluster_id)
        cluster = self._drain.drain.id_to_cluster.get(id_int) if id_int is not None else None
        if cluster is None:
            return False
        cluster.log_template_tokens = tuple(new_tokens)
        return True

    def remove_cluster(self, cluster_id: str) -> bool:
        """Delete a cluster from Drain3's internal state. Returns True on success."""
        id_int = self._as_int(cluster_id)
        if id_int is None or id_int not in self._drain.drain.id_to_cluster:
            return False
        del self._drain.drain.id_to_cluster[id_int]
        return True

    def get_template(self, cluster_id: str) -> str | None:
        """Return the current template string for a cluster, or None if not found."""
        id_int = self._as_int(cluster_id)
        cluster = self._drain.drain.id_to_cluster.get(id_int) if id_int is not None else None
        if cluster is None:
            return None
        return " ".join(cluster.log_template_tokens)

    def all_clusters(self) -> list[dict]:
        """Return a list of dicts describing every known cluster."""
        out: list[dict] = []
        for cluster in self._drain.drain.id_to_cluster.values():
            tokens = list(cluster.log_template_tokens)
            out.append({
                "cluster_id": str(cluster.cluster_id),
                "template": " ".join(tokens),
                "token_count": len(tokens),
            })
        return out
