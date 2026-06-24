"""Safe execution of LLM-suggested template splits.

When the LLM decides a degraded template actually covers several distinct
event types, this module materialises the split *conservatively*: only the
first sub-template reuses the original Drain3 slot; the rest live purely in
the AdaptiveDrain TemplateStore, so we never inject fabricated clusters into
Drain3's internal state.
"""

from __future__ import annotations

from drain_adapter import DrainAdapter
from reservoir_sampler import ReservoirSampler
from template_store import TemplateStore


class TemplateSplitter:
    """Executes a validated split of one cluster into several sub-templates."""

    def __init__(
        self,
        drain_adapter: DrainAdapter,
        template_store: TemplateStore,
        sampler: ReservoirSampler,
    ) -> None:
        self._drain = drain_adapter
        self._store = template_store
        self._sampler = sampler

    def execute_split(
        self,
        original_cluster_id: str,
        sub_templates: list[str],
        samples: list[str],
    ) -> list[str]:
        """Materialise a split. Returns the new cluster_ids, or [] if rejected."""
        if not self._validate(sub_templates):
            return []

        n = len(sub_templates)
        # Distribute samples round-robin across the sub-templates.
        buckets: list[list[str]] = [[] for _ in range(n)]
        for j, sample in enumerate(samples):
            buckets[j % n].append(sample)

        new_ids: list[str] = []
        for index, sub_template in enumerate(sub_templates):
            new_id = f"split_{original_cluster_id}_{index}"
            self._store.register(new_id, sub_template)

            # Only index 0 reuses the original Drain3 slot; higher indices stay
            # store-only (injecting synthetic clusters into Drain3 is too risky).
            if index == 0:
                self._drain.update_template(original_cluster_id, sub_template.split())

            for sample in buckets[index]:
                self._sampler.add(new_id, sample)

            new_ids.append(new_id)

        # Soft-retire the original cluster: it folds into the first sub-template
        # once enough confirming hits arrive.
        self._store.stage_merge(original_cluster_id, new_ids[0])
        return new_ids

    @staticmethod
    def _validate(sub_templates: list[str]) -> bool:
        """A split needs >= 2 non-empty string sub-templates."""
        if not isinstance(sub_templates, list) or len(sub_templates) < 2:
            print(f"[splitter] rejected: need >= 2 sub_templates, got {len(sub_templates) if isinstance(sub_templates, list) else 'non-list'}")
            return False
        for st in sub_templates:
            if not isinstance(st, str) or not st.strip():
                print("[splitter] rejected: every sub_template must be a non-empty string")
                return False
        return True
