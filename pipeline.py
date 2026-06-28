"""Main orchestrator — wires all AdaptiveDrain modules and runs the LLM worker thread."""

from __future__ import annotations
import threading
import time
from typing import Any

from approver import HumanApprover
from drain_adapter import DrainAdapter
from preprocessor import LogPreprocessor
from llm_gate import LLMGate
from metrics import MetricsCollector
from normalizer import OCSFNormalizer
from pending_set import PendingReviewSet
from persistence import StatePersistence
from reservoir_sampler import ReservoirSampler
from review_queue import ReviewQueue
from splitter import TemplateSplitter
from template_store import TemplateStore


class TemplatePipeline:
    """Hot-path log ingestion pipeline with an async LLM decision gate."""

    def __init__(
        self,
        drain_instance: Any,
        openrouter_api_key: str,
        confirm_threshold: int = 100,
        queue_maxsize: int = 500,
        pending_ttl: float = 30.0,
        degradation_threshold: float = 0.4,
        normalizer: OCSFNormalizer | None = None,
        persistence: StatePersistence | None = None,
        metrics: MetricsCollector | None = None,
        approver: HumanApprover | None = None,
    ) -> None:
        self._degradation_threshold = degradation_threshold

        self.drain_adapter = DrainAdapter(drain_instance)
        self.preprocessor  = LogPreprocessor()
        self.sampler = ReservoirSampler()
        self.pending = PendingReviewSet(ttl_seconds=pending_ttl)
        self.queue = ReviewQueue(maxsize=queue_maxsize)
        self.store = TemplateStore(confirm_threshold=confirm_threshold)
        self.llm_gate = LLMGate(api_key=openrouter_api_key)

        # Optional add-ons.
        self.normalizer = normalizer
        self.persistence = persistence
        self.metrics = metrics
        self.approver = approver

        # Instantiated after the core modules it depends on.
        self.splitter = TemplateSplitter(self.drain_adapter, self.store, self.sampler)

        self._worker_thread = threading.Thread(
            target=self._llm_worker, daemon=True, name="llm-worker"
        )
        self._worker_thread.start()

    def _bump(self, counter: str, by: int = 1) -> None:
        """Increment a metrics counter if a collector is attached."""
        if self.metrics is not None:
            self.metrics.increment(counter, by)

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def ingest(self, raw_log: str) -> dict:
        """Process a single raw log line. Never blocks on LLM."""
        pre    = self.preprocessor.process(raw_log)
        result = self.drain_adapter.add_log(pre.processed)
        result["original_log"]  = raw_log
        result["processed_log"] = pre.processed
        result["extractions"]   = pre.extractions

        cluster_id: str = result["cluster_id"]
        template: str = result["template"]
        tokens: list[str] = result["tokens"]
        change_type: str = result["change_type"]

        self.sampler.add(cluster_id, raw_log)  # always original
        self._bump("logs_ingested")

        if change_type == "NONE":
            self.store.confirm_merge_hit(cluster_id)

        elif change_type == "CREATE":
            self._bump("templates_created")
            self.store.register(cluster_id, template)
            self.store.add_version(cluster_id, template, trigger_log=raw_log)
            if self.pending.should_review(template):
                self.queue.put({
                    "type": "create",
                    "cluster_id": cluster_id,
                    "template": template,
                    "samples": self.sampler.get(cluster_id),
                })

        elif change_type == "UPDATE":
            self._bump("templates_updated")
            self.store.add_version(cluster_id, template, trigger_log=raw_log)
            wildcard_count = sum(1 for t in tokens if t == "<*>")
            wildcard_ratio = wildcard_count / len(tokens) if tokens else 0.0
            if wildcard_ratio > self._degradation_threshold and self.pending.should_review(template):
                self.queue.put({
                    "type": "degradation",
                    "cluster_id": cluster_id,
                    "template": template,
                    "wildcard_ratio": wildcard_ratio,
                    "samples": self.sampler.get(cluster_id),
                })

        # OCSF normalization (optional): attach event classification to the result.
        if self.normalizer is not None:
            ocsf_label = self.normalizer.normalize(template)
            ocsf_full  = self.normalizer.normalize_full(raw_log, template) if ocsf_label else None
            result["ocsf"]       = ocsf_label
            result["ocsf_event"] = ocsf_full
            self._bump("ocsf_matched" if ocsf_label is not None else "ocsf_unmatched")
        else:
            result["ocsf"]       = None
            result["ocsf_event"] = None

        return result

    # ------------------------------------------------------------------
    # Background LLM worker
    # ------------------------------------------------------------------

    def _llm_worker(self) -> None:
        """Continuously drain the review queue and process items via LLM."""
        while True:
            item = self.queue.get()
            if item is None:
                time.sleep(0.1)
                continue
            try:
                self._process_item(item)
            except Exception as exc:
                print(f"[LLM worker] unhandled error: {exc}")
            finally:
                self.pending.release(item.get("template", ""))

    def _process_item(self, item: dict) -> None:
        """Dispatch a queued review item to the appropriate LLM handler."""
        if item["type"] == "create":
            self._handle_create(item)
        elif item["type"] == "degradation":
            self._handle_degradation(item)

    def _handle_create(self, item: dict) -> None:
        active_dicts = [
            {"cluster_id": t.cluster_id, "pattern": t.pattern}
            for t in self.store.all_active()
            if t.cluster_id != item["cluster_id"]
        ]
        candidates = self.llm_gate.find_candidates(item["template"], active_dicts)
        prompt = self.llm_gate.build_prompt_create(item["template"], item["samples"], candidates)
        self._bump("llm_calls")
        decision = self.llm_gate.call(prompt, original_template=item["template"])

        if self.approver is not None:
            decision = self.approver.review(item, decision)

        print(
            f"[LLM create] cluster={item['cluster_id']} "
            f"decision={decision.get('decision')} "
            f"reason={decision.get('reasoning', '')[:80]}"
        )

        if decision.get("reasoning") == "llm_error":
            self._bump("llm_errors")
        if decision.get("decision") == "keep":
            self._bump("llm_fallback_keep")

        cluster = self.store.get(item["cluster_id"])
        if cluster:
            if decision.get("labeled_template"):
                cluster.labeled_template = decision["labeled_template"]
            cluster.llm_decision  = decision.get("decision")
            cluster.llm_reasoning = decision.get("reasoning")
            quality = decision.get("quality", {})
            if quality:
                cluster.quality_score      = quality.get("score")
                cluster.quality_issues     = quality.get("issues", [])
                cluster.quality_suggestion = quality.get("suggestion", "")

        # Re-evaluation: update metadata only — never execute merge/split
        if item.get("reevaluation"):
            self.pending.release(item["template"])
            return

        if decision.get("decision") == "merge":
            target_id: str | None = decision.get("target_cluster_id")
            if target_id:
                self.store.stage_merge(item["cluster_id"], target_id)
                self._bump("templates_merged")
                merged_template: str | None = decision.get("merged_template")
                if merged_template:
                    new_tokens = merged_template.split()
                    self.drain_adapter.update_template(target_id, new_tokens)

    def _handle_degradation(self, item: dict) -> None:
        prompt = self.llm_gate.build_prompt_degradation(
            item["template"], item["wildcard_ratio"], item["samples"]
        )
        self._bump("llm_calls")
        decision = self.llm_gate.call(prompt, original_template=item["template"])

        if self.approver is not None:
            decision = self.approver.review(item, decision)

        print(
            f"[LLM degradation] cluster={item['cluster_id']} "
            f"decision={decision.get('decision')} "
            f"reason={decision.get('reasoning', '')[:80]}"
        )

        if decision.get("reasoning") == "llm_error":
            self._bump("llm_errors")

        deg_cluster = self.store.get(item["cluster_id"])
        if deg_cluster:
            deg_cluster.llm_decision  = decision.get("decision")
            deg_cluster.llm_reasoning = decision.get("reasoning")
            quality = decision.get("quality", {})
            if quality:
                deg_cluster.quality_score      = quality.get("score")
                deg_cluster.quality_issues     = quality.get("issues", [])
                deg_cluster.quality_suggestion = quality.get("suggestion", "")

        if decision.get("decision") == "split":
            new_ids = self.splitter.execute_split(
                item["cluster_id"],
                decision.get("sub_templates", []),
                item["samples"],
            )
            if new_ids:
                self._bump("templates_split")
                print(f"[SPLIT EXECUTED] {item['cluster_id']} → {new_ids}")

        elif decision.get("decision") == "reset":
            reset_template: str | None = decision.get("reset_template")
            if reset_template:
                new_tokens = reset_template.split()
                self.drain_adapter.update_template(item["cluster_id"], new_tokens)
                managed = self.store.get(item["cluster_id"])
                if managed is not None:
                    managed.pattern = reset_template
                    labeled = decision.get("labeled_template")
                    if labeled:
                        managed.labeled_template = labeled

    # ------------------------------------------------------------------
    # Re-evaluation
    # ------------------------------------------------------------------

    def reevaluate_all(self, min_score: int = 10) -> int:
        """Re-queue ACTIVE templates for LLM review. Returns count queued."""
        try:
            count = 0
            for t in self.store.all_active():
                if t.quality_score is not None and t.quality_score >= min_score:
                    continue
                if not self.pending.should_review(t.pattern):
                    continue
                self.queue.put({
                    "type":         "create",
                    "cluster_id":   t.cluster_id,
                    "template":     t.pattern,
                    "samples":      self.sampler.get(t.cluster_id),
                    "reevaluation": True,
                })
                count += 1
            return count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return combined stats from the queue and template store."""
        return {
            "queue": self.queue.stats(),
            "templates": self.store.stats(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> bool:
        """Persist store + sampler state to disk if persistence is configured."""
        if self.persistence is not None:
            return self.persistence.save(self.store, self.sampler)
        return False

    def load(self) -> bool:
        """Restore store + sampler state from disk if persistence is configured."""
        if self.persistence is not None:
            return self.persistence.load(self.store, self.sampler)
        return False
