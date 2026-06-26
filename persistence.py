"""Disk persistence for AdaptiveDrain state (TemplateStore + ReservoirSampler).

Drain3's own internals are intentionally NOT persisted here — only the
AdaptiveDrain-managed state, serialised as a single JSON document. Writes are
atomic (temp file + os.replace) and every method is failure-tolerant: errors
are logged and surfaced as a False return, never raised.
"""

from __future__ import annotations

import json
import os
import time

from reservoir_sampler import ReservoirSampler
from template_store import ManagedTemplate, TemplateStatus, TemplateStore

_VERSION = 1


class StatePersistence:
    """Saves/restores TemplateStore + ReservoirSampler state to a JSON file."""

    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, store: TemplateStore, sampler: ReservoirSampler) -> bool:
        """Atomically write current state to disk. Returns True on success."""
        try:
            templates = [
                {
                    "cluster_id": t.cluster_id,
                    "pattern": t.pattern,
                    "status": t.status.value,
                    "merge_target_id": t.merge_target_id,
                    "confirmation_count": t.confirmation_count,
                    "created_at": t.created_at,
                    "labeled_template": t.labeled_template,
                }
                for t in store._store.values()
            ]
            payload = {
                "version": _VERSION,
                "saved_at": time.time(),
                "templates": templates,
                "reservoirs": {cid: list(logs) for cid, logs in sampler._reservoirs.items()},
                "reservoir_counts": dict(sampler._counts),
            }
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp_path, self.path)
            return True
        except Exception as exc:
            print(f"[persistence] save failed: {exc}")
            return False

    def load(self, store: TemplateStore, sampler: ReservoirSampler) -> bool:
        """Restore state from disk into the given store and sampler.

        Returns False if the file is missing, unparseable, or the wrong version.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            print(f"[persistence] load failed: {exc}")
            return False

        if payload.get("version") != _VERSION:
            print(f"[persistence] version mismatch: {payload.get('version')} != {_VERSION}")
            return False

        try:
            for t in payload.get("templates", []):
                cid = t["cluster_id"]
                store.register(cid, t["pattern"])
                managed: ManagedTemplate | None = store.get(cid)
                if managed is None:
                    continue
                managed.status = TemplateStatus(t["status"])
                managed.merge_target_id = t["merge_target_id"]
                managed.confirmation_count = t["confirmation_count"]
                managed.created_at = t["created_at"]
                managed.labeled_template = t.get("labeled_template")

            sampler._reservoirs = {cid: list(logs) for cid, logs in payload.get("reservoirs", {}).items()}
            sampler._counts = dict(payload.get("reservoir_counts", {}))

            # Rebuild the reverse index from restored PENDING_MERGE templates.
            store._pending_by_target = {}
            for managed in store._store.values():
                if managed.status is TemplateStatus.PENDING_MERGE and managed.merge_target_id is not None:
                    store._pending_by_target.setdefault(managed.merge_target_id, set()).add(managed.cluster_id)
            return True
        except Exception as exc:
            print(f"[persistence] restore failed: {exc}")
            return False

    def exists(self) -> bool:
        """Return True if the state file exists on disk."""
        return os.path.exists(self.path)
