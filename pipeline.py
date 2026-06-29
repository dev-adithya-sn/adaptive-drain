"""Main orchestrator — wires all AdaptiveDrain modules and runs batch LLM review."""

from __future__ import annotations
from typing import Any

from approver import HumanApprover
from drain_adapter import DrainAdapter
from preprocessor import LogPreprocessor
from llm_gate import LLMGate
from metrics import MetricsCollector
from normalizer import OCSFNormalizer
from persistence import StatePersistence
from reservoir_sampler import ReservoirSampler
from splitter import TemplateSplitter
from template_store import TemplateStore


class TemplatePipeline:
    """Log ingestion pipeline with synchronous batch LLM review after upload."""

    def __init__(
        self,
        drain_instance: Any,
        openrouter_api_key: str,
        confirm_threshold: int = 100,
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
        self.store = TemplateStore(confirm_threshold=confirm_threshold)
        self.llm_gate = LLMGate(api_key=openrouter_api_key)

        self.normalizer  = normalizer
        self.persistence = persistence
        self.metrics     = metrics
        self.approver    = approver

        self.splitter = TemplateSplitter(self.drain_adapter, self.store, self.sampler)

    def _bump(self, counter: str, by: int = 1) -> None:
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

        cluster_id: str  = result["cluster_id"]
        template: str    = result["template"]
        change_type: str = result["change_type"]

        self.sampler.add(cluster_id, raw_log)
        self._bump("logs_ingested")

        if change_type == "NONE":
            self.store.confirm_merge_hit(cluster_id)

        elif change_type == "CREATE":
            self._bump("templates_created")
            self.store.register(cluster_id, template)
            self.store.add_version(cluster_id, template, trigger_log=raw_log)

        elif change_type == "UPDATE":
            self._bump("templates_updated")
            self.store.add_version(cluster_id, template, trigger_log=raw_log)

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
    # Batch LLM review
    # ------------------------------------------------------------------

    def batch_review(self, session_results: list[dict]) -> dict:
        """
        Called once after all logs are ingested for a session.
        Collects new/updated templates, makes ONE LLM batch call, stores
        decisions in the approver for UI display (or auto-executes if no approver).
        """
        import uuid

        seen      = set()
        to_review = []

        for r in session_results:
            cid         = str(r.get("cluster_id", ""))
            change_type = r.get("change_type", "")

            if cid in seen or change_type not in ("CREATE", "UPDATE"):
                continue

            t = self.store.get(cid)
            if not t:
                continue

            tokens         = t.pattern.split()
            wildcard_count = sum(1 for tok in tokens if tok == "<*>")
            wildcard_ratio = wildcard_count / max(len(tokens), 1)

            seen.add(cid)
            to_review.append({
                "cluster_id":    cid,
                "template":      t.pattern,
                "samples":       self.sampler.get(cid)[:3],
                "wildcard_ratio": wildcard_ratio,
            })

        if not to_review:
            return {"batch_id": "", "decisions": [], "queued": 0}

        self._bump("llm_calls")
        batch_id  = str(uuid.uuid4())
        decisions = self.llm_gate.call_batch(to_review)

        if self.approver and hasattr(self.approver, "set_batch"):
            self.approver.set_batch(batch_id, decisions, to_review)
        else:
            for dec in decisions:
                self._execute_decision(dec)

        return {
            "batch_id": batch_id,
            "decisions": decisions,
            "queued":   len(decisions),
        }

    def _execute_decision(self, dec: dict) -> None:
        """Execute a single LLM decision dict after user approval."""
        cid      = str(dec.get("cluster_id", ""))
        decision = dec.get("decision", "keep")
        t        = self.store.get(cid)

        if not t:
            return

        t.labeled_template   = dec.get("labeled_template")
        t.llm_decision       = decision
        t.llm_reasoning      = dec.get("reasoning", "")
        quality              = dec.get("quality") or {}
        t.quality_score      = quality.get("score")
        t.quality_issues     = quality.get("issues", [])
        t.quality_suggestion = quality.get("suggestion", "")

        if decision == "merge":
            target_id  = str(dec.get("merge_into_cluster_id") or "")
            merged_tpl = dec.get("merged_template")
            if target_id and target_id != cid:
                self.store.stage_merge(cid, target_id)
                if merged_tpl:
                    self.drain_adapter.update_template(target_id, merged_tpl.split())
                if self.metrics:
                    self.metrics.increment("templates_merged")

        elif decision == "split":
            sub_templates = dec.get("sub_templates", [])
            if sub_templates and len(sub_templates) >= 2:
                new_ids = self.splitter.execute_split(cid, sub_templates, self.sampler.get(cid))
                if new_ids and self.metrics:
                    self.metrics.increment("templates_split")

        elif decision == "reset":
            reset_tpl = dec.get("reset_template")
            if reset_tpl:
                self.drain_adapter.update_template(cid, reset_tpl.split())
                t.pattern = reset_tpl

    def execute_batch(self, batch_id: str) -> int:
        """Execute all approved decisions for a batch. Returns count executed."""
        if not self.approver or not hasattr(self.approver, "get_batch"):
            return 0

        decisions = self.approver.get_batch(batch_id)
        if not decisions:
            return 0

        for dec in decisions:
            try:
                self._execute_decision(dec)
            except Exception as e:
                print(f"[pipeline] execute_decision error: {e}")

        self.approver.clear_batch(batch_id)
        if self.persistence:
            self.persistence.save(self.store, self.sampler)

        return len(decisions)

    # ------------------------------------------------------------------
    # Re-evaluation
    # ------------------------------------------------------------------

    def reevaluate_all(self, min_score: int = 10) -> int:
        """Re-evaluate ACTIVE templates via batch LLM call. Returns count queued."""
        try:
            to_review = []
            for t in self.store.all_active():
                if t.quality_score is not None and t.quality_score >= min_score:
                    continue
                tokens = t.pattern.split()
                wc     = sum(1 for tok in tokens if tok == "<*>")
                to_review.append({
                    "cluster_id":    t.cluster_id,
                    "template":      t.pattern,
                    "samples":       self.sampler.get(t.cluster_id)[:3],
                    "wildcard_ratio": wc / max(len(tokens), 1),
                })

            if not to_review:
                return 0

            self._bump("llm_calls")
            decisions = self.llm_gate.call_batch(to_review)

            if self.approver and hasattr(self.approver, "set_batch"):
                import uuid
                batch_id = str(uuid.uuid4())
                self.approver.set_batch(batch_id, decisions, to_review)
                return len(decisions)

            for dec in decisions:
                self._execute_decision(dec)
            return len(decisions)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {"templates": self.store.stats()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> bool:
        if self.persistence is not None:
            return self.persistence.save(self.store, self.sampler)
        return False

    def load(self) -> bool:
        if self.persistence is not None:
            return self.persistence.load(self.store, self.sampler)
        return False
