"""Template lifecycle management: tracks ACTIVE, PENDING_MERGE, and MERGED states."""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class TemplateStatus(Enum):
    ACTIVE            = "ACTIVE"
    PENDING_MERGE     = "PENDING_MERGE"
    MERGED            = "MERGED"
    # Populated when a log nearly matched an existing template and the LLM +
    # human still need to decide whether it belongs to that template or is
    # genuinely new.  Not used on ManagedTemplate itself; exposed here so that
    # the near-match queue can share the same status vocabulary.
    NEAR_MATCH_REVIEW = "NEAR_MATCH_REVIEW"


@dataclass
class ManagedTemplate:
    """Holds metadata for a single Drain3 cluster managed by AdaptiveDrain."""

    cluster_id: str
    pattern: str
    status: TemplateStatus = TemplateStatus.ACTIVE
    merge_target_id: str | None = None
    confirmation_count: int = 0
    created_at: float = field(default_factory=time.time)
    labeled_template:   str | None = None
    llm_decision:       str | None = None
    llm_reasoning:      str | None = None
    versions:           list = field(default_factory=list)


class TemplateStore:
    """Central registry for ManagedTemplate instances with soft-merge support."""

    def __init__(self, confirm_threshold: int = 100) -> None:
        self._confirm_threshold = confirm_threshold
        self._store: dict[str, ManagedTemplate] = {}
        # Reverse index: target_id -> {pending new_id, ...}. Keeps confirm_merge_hit
        # O(pending-for-target) on the hot path instead of O(all templates).
        self._pending_by_target: dict[str, set[str]] = {}

    def register(self, cluster_id: str, pattern: str) -> None:
        """Add a new ACTIVE template. No-op if cluster_id already exists."""
        if cluster_id not in self._store:
            self._store[cluster_id] = ManagedTemplate(cluster_id=cluster_id, pattern=pattern)

    def add_version(self, cluster_id: str, template: str, trigger_log: str = "") -> None:
        """Append a version snapshot to a template's history."""
        t = self._store.get(cluster_id)
        if not t:
            return
        t.versions.append({
            "version":     len(t.versions) + 1,
            "template":    template,
            "trigger_log": trigger_log,
            "timestamp":   time.time(),
        })

    def stage_merge(self, new_id: str, target_id: str) -> None:
        """Mark new_id as PENDING_MERGE pointing to target_id."""
        tmpl = self._store.get(new_id)
        if tmpl is None:
            return
        # If it was already pending toward a different target, drop the stale link.
        if tmpl.status == TemplateStatus.PENDING_MERGE and tmpl.merge_target_id is not None:
            self._unindex_pending(new_id, tmpl.merge_target_id)
        tmpl.status = TemplateStatus.PENDING_MERGE
        tmpl.merge_target_id = target_id
        self._pending_by_target.setdefault(target_id, set()).add(new_id)

    def confirm_merge_hit(self, target_id: str) -> None:
        """Increment confirmation_count for every PENDING_MERGE pointing to target_id.

        Hard-deletes the pending template when count reaches the threshold.
        Uses the reverse index, so cost is proportional to the number of templates
        pending against this target, not the whole store.
        """
        pending_ids = self._pending_by_target.get(target_id)
        if not pending_ids:
            return
        to_delete: list[str] = []
        # Snapshot: hard_delete mutates the set we are iterating.
        for new_id in list(pending_ids):
            tmpl = self._store.get(new_id)
            if tmpl is None or tmpl.status != TemplateStatus.PENDING_MERGE:
                continue
            tmpl.confirmation_count += 1
            if tmpl.confirmation_count >= self._confirm_threshold:
                to_delete.append(tmpl.cluster_id)
        for cid in to_delete:
            self.hard_delete(cid)

    def rollback_merge(self, new_id: str) -> None:
        """Revert a PENDING_MERGE template back to ACTIVE."""
        tmpl = self._store.get(new_id)
        if tmpl is not None and tmpl.status == TemplateStatus.PENDING_MERGE:
            self._unindex_pending(new_id, tmpl.merge_target_id)
            tmpl.status = TemplateStatus.ACTIVE
            tmpl.merge_target_id = None

    def hard_delete(self, cluster_id: str) -> None:
        """Unconditionally remove a template from the store."""
        tmpl = self._store.pop(cluster_id, None)
        if tmpl is not None and tmpl.merge_target_id is not None:
            self._unindex_pending(cluster_id, tmpl.merge_target_id)

    def _unindex_pending(self, new_id: str, target_id: str | None) -> None:
        """Remove new_id from the reverse index entry for target_id, if present."""
        if target_id is None:
            return
        bucket = self._pending_by_target.get(target_id)
        if bucket is not None:
            bucket.discard(new_id)
            if not bucket:
                del self._pending_by_target[target_id]

    def get(self, cluster_id: str) -> ManagedTemplate | None:
        """Return the ManagedTemplate for a cluster, or None if not found."""
        return self._store.get(cluster_id)

    def all_active(self) -> list[ManagedTemplate]:
        """Return all templates currently in ACTIVE status."""
        return [t for t in self._store.values() if t.status == TemplateStatus.ACTIVE]

    def stats(self) -> dict:
        """Return a count breakdown by TemplateStatus."""
        counts: dict[str, int] = {s.value: 0 for s in TemplateStatus}
        for tmpl in self._store.values():
            counts[tmpl.status.value] += 1
        return counts


# ---------------------------------------------------------------------------
# Near-match queue
# ---------------------------------------------------------------------------

@dataclass
class NearMatchItem:
    """One log line that nearly matched an existing template, awaiting review."""

    item_id:              str
    raw_log:              str
    candidate_cluster_id: str
    similarity_score:     float
    status:               str = "pending"  # "pending" | "approved" | "rejected"
    llm_decision:         str | None = None
    corrected_template:   str | None = None
    added_at:             float = field(default_factory=time.time)


class NearMatchQueue:
    """Stores near-match log items awaiting LLM + human review.

    Items can transition from ``pending`` → ``approved`` or ``rejected``
    (corresponding to ``NEAR_MATCH_REVIEW`` status vocabulary in
    ``TemplateStatus``).
    """

    def __init__(self) -> None:
        self._items: dict[str, NearMatchItem] = {}

    def add(
        self,
        raw_log:              str,
        candidate_cluster_id: str,
        similarity_score:     float,
        item_id:              str | None = None,
    ) -> str:
        """Enqueue a near-match item and return its item_id."""
        if item_id is None:
            item_id = str(uuid.uuid4())
        self._items[item_id] = NearMatchItem(
            item_id              = item_id,
            raw_log              = raw_log,
            candidate_cluster_id = candidate_cluster_id,
            similarity_score     = similarity_score,
        )
        return item_id

    def get(self, item_id: str) -> NearMatchItem | None:
        return self._items.get(item_id)

    def get_pending(self) -> list[NearMatchItem]:
        """Return all items currently in ``pending`` status."""
        return [i for i in self._items.values() if i.status == "pending"]

    def get_all(self) -> list[NearMatchItem]:
        return list(self._items.values())

    def approve(
        self,
        item_id:            str,
        llm_decision:       str,
        corrected_template: str | None = None,
    ) -> bool:
        """Mark an item as approved with the given LLM decision.

        Returns True if found, False if not.
        """
        item = self._items.get(item_id)
        if item is None:
            return False
        item.status             = "approved"
        item.llm_decision       = llm_decision
        item.corrected_template = corrected_template
        return True

    def reject(self, item_id: str) -> bool:
        """Mark an item as rejected (will be routed to Drain3)."""
        item = self._items.get(item_id)
        if item is None:
            return False
        item.status = "rejected"
        return True

    def remove(self, item_id: str) -> bool:
        """Remove item from the queue entirely."""
        return self._items.pop(item_id, None) is not None

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
