"""Main orchestrator — wires all AdaptiveDrain modules and runs batch LLM review."""

from __future__ import annotations
import time
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
        self._llm_decision_cache: dict[str, dict] = {}
        self._ocsf_classify_cache: dict[str, dict] = {}
        from collections import deque
        self._parsed_logs: deque = deque(maxlen=100)

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
            result["ocsf"]       = ocsf_label
            result["ocsf_event"] = None
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
        decisions = self.llm_gate.call_batch(to_review, prior_decisions=self._llm_decision_cache)

        if self.approver and hasattr(self.approver, "set_batch"):
            self.approver.set_batch(batch_id, decisions, to_review)
        else:
            for dec in decisions:
                self._execute_decision(dec)
                self._llm_decision_cache[dec.get("cluster_id", "")] = {
                    "decision":  dec.get("decision", "keep"),
                    "template":  dec.get("labeled_template") or dec.get("merged_template") or dec.get("template", ""),
                    "reasoning": dec.get("reasoning", "")
                }

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
        raw_score = quality.get("score")
        if raw_score is not None:
            try:
                raw_score = max(0, min(10, int(float(raw_score))))
            except (TypeError, ValueError):
                raw_score = None
        t.quality_score      = raw_score
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

    def _parse_cluster_logs(self, cid: str) -> None:
        """Parse all sampled logs for a confirmed template and push to _parsed_logs."""
        t = self.store.get(cid)
        if not t or not self.normalizer:
            return
        if t.llm_decision is None:
            return  # only parse logs for templates that have been LLM-reviewed and approved
        template = t.labeled_template or t.pattern
        samples  = self.sampler.get(cid)
        for raw_log in samples:
            try:
                ocsf_label = self.normalizer.normalize(template)
                if ocsf_label is None:
                    # Check classify cache first
                    if template not in self._ocsf_classify_cache:
                        samples = self.sampler.get(cid)[:3]
                        self._ocsf_classify_cache[template] = self.llm_gate.classify_template(template, samples)
                    ocsf_label = self._ocsf_classify_cache[template]
                ocsf_full = self.normalizer.normalize_full(raw_log, template) if self.normalizer.normalize(template) else {}
                if not ocsf_full and ocsf_label:
                    # Build minimal event from LLM classification
                    import uuid as _uuid
                    ocsf_full = {
                        "class_name":    ocsf_label.get("ocsf_class_name", "Unknown"),
                        "activity_name": ocsf_label.get("activity_name", "Unknown"),
                        "category_name": ocsf_label.get("category_name", "Other"),
                        "severity_id":   ocsf_label.get("severity_id", 1),
                        "severity":      {1:"Informational",2:"Low",3:"Medium",4:"High",5:"Critical"}.get(ocsf_label.get("severity_id",1),"Unknown"),
                        "status":        "",
                        "message":       raw_log,
                        "raw_data":      raw_log,
                    }
                ocsf       = ocsf_full or {}
                cached = self._ocsf_classify_cache.get(template, {})
                entry = {
                    "raw_log":      raw_log,
                    "template":     template,
                    "cluster_id":   cid,
                    "ocsf_class":   ocsf.get("class_name", ""),
                    "activity":     ocsf.get("activity_name", ""),
                    "severity":     ocsf.get("severity", ""),
                    "status":       ocsf.get("status", ""),
                    "username":     (ocsf.get("user") or {}).get("name", ""),
                    "src_ip":       (ocsf.get("src_endpoint") or {}).get("ip", ""),
                    "dst_ip":       (ocsf.get("dst_endpoint") or {}).get("ip", ""),
                    "src_port":     (ocsf.get("src_endpoint") or {}).get("port", ""),
                    "http_method":  (ocsf.get("http_request") or {}).get("http_method", ""),
                    "http_path":    (ocsf.get("http_request") or {}).get("url", {}).get("path", ""),
                    "http_status":  (ocsf.get("http_response") or {}).get("code", ""),
                    "db_name":      (ocsf.get("database") or {}).get("name", ""),
                    "service":      (ocsf.get("service") or {}).get("name", ""),
                    "ingested_at":  int(time.time() * 1000),
                    "log_source":      cached.get("log_source", ""),
                    "vendor":          cached.get("vendor", ""),
                    "semantic_event":  cached.get("semantic_event", ""),
                    "security_relevant": cached.get("security_relevant", False),
                    "storage_class":   cached.get("storage_class", "hot"),
                    "mitre_techniques": cached.get("mitre_attack_techniques", []),
                    "detection_tags":  cached.get("detection_tags", []),
                    "telemetry_type":  cached.get("telemetry_type", ""),
                }
                self._parsed_logs.append(entry)
            except Exception:
                pass

    def execute_batch(self, batch_id: str, reparse: bool = False) -> dict:
        """Execute all approved decisions for a batch.

        Returns dict with executed count and optional reparse results.
        """
        if not self.approver or not hasattr(self.approver, "get_batch"):
            return {"executed": 0, "reparse": None}

        decisions = self.approver.get_batch(batch_id)
        if not decisions:
            return {"executed": 0, "reparse": None}

        for dec in decisions:
            try:
                self._execute_decision(dec)
                self._parse_cluster_logs(str(dec.get("cluster_id", "")))
            except Exception as e:
                print(f"[pipeline] execute_decision error: {e}")

        self.approver.clear_batch(batch_id)
        if self.persistence:
            self.persistence.save(self.store, self.sampler)

        reparse_result = None
        if reparse:
            try:
                reparse_result = self.reparse_and_review()
            except Exception as e:
                print(f"[pipeline] reparse error: {e}")

        return {"executed": len(decisions), "reparse": reparse_result}

    def reparse_and_review(self) -> dict:
        """Re-run all reservoir samples through current Drain3, then batch-review
        any templates that changed. Never raises."""
        import uuid

        empty = {
            "batch_id": "", "decisions": [], "queued": 0,
            "reparse_stats": {"logs_reparsed": 0, "new_creates": 0, "new_updates": 0},
        }

        try:
            # collect all original logs from reservoir across all active clusters
            all_logs = []
            for t in self.store.all_active():
                all_logs.extend(self.sampler.get(t.cluster_id))

            if not all_logs:
                return empty

            # re-ingest through updated Drain3
            reparse_results = []
            for log in all_logs:
                try:
                    pre    = self.preprocessor.process(log)
                    result = self.drain_adapter.add_log(pre.processed)
                    result["original_log"]  = log
                    result["processed_log"] = pre.processed

                    cid = str(result.get("cluster_id", ""))
                    ct  = result.get("change_type", "NONE")

                    if cid:
                        self.sampler.add(cid, log)

                    if ct == "CREATE":
                        template = result.get("template", "")
                        if cid and not self.store.get(cid):
                            self.store.register(cid, template)
                            self.store.add_version(cid, template, trigger_log=log)

                    elif ct == "UPDATE":
                        template = result.get("template", "")
                        if cid:
                            self.store.add_version(cid, template, trigger_log=log)

                    reparse_results.append(result)
                except Exception as e:
                    print(f"[reparse] error on log: {e}")
                    continue

            new_creates = sum(1 for r in reparse_results if r.get("change_type") == "CREATE")
            new_updates = sum(1 for r in reparse_results if r.get("change_type") == "UPDATE")

            batch_result = self.batch_review(reparse_results)
            batch_result["reparse_stats"] = {
                "logs_reparsed": len(all_logs),
                "new_creates":   new_creates,
                "new_updates":   new_updates,
            }
            return batch_result

        except Exception as e:
            print(f"[reparse_and_review] error: {e}")
            return empty

    # ------------------------------------------------------------------
    # Re-evaluation
    # ------------------------------------------------------------------

    def reevaluate_all(self, min_score: int = 10) -> int:
        """Re-evaluate ACTIVE templates via batch LLM call. Returns count queued."""
        try:
            to_review = []
            for t in self.store.all_active():
                if min_score > 0 and t.quality_score is not None and t.quality_score >= min_score:
                    continue
                tokens = t.pattern.split()
                wc     = sum(1 for tok in tokens if tok == "<*>")
                to_review.append({
                    "cluster_id":    t.cluster_id,
                    "template":      t.pattern,
                    "samples":       self.sampler.get(t.cluster_id)[:3],
                    "wildcard_ratio": wc / max(len(tokens), 1),
                })

            print(f"[reevaluate] active templates: {len(list(self.store.all_active()))}, to_review: {len(to_review)}, min_score={min_score}", flush=True)

            if not to_review:
                return 0

            self._bump("llm_calls")
            decisions = self.llm_gate.call_batch(to_review, prior_decisions=self._llm_decision_cache)
            print(f"[reevaluate] decisions received: {len(decisions)}", flush=True)

            if self.approver and hasattr(self.approver, "set_batch"):
                import uuid
                batch_id = str(uuid.uuid4())
                self.approver.set_batch(batch_id, decisions, to_review)
                return len(decisions)

            for dec in decisions:
                self._execute_decision(dec)
                self._llm_decision_cache[dec.get("cluster_id", "")] = {
                    "decision":  dec.get("decision", "keep"),
                    "template":  dec.get("labeled_template") or dec.get("merged_template") or dec.get("template", ""),
                    "reasoning": dec.get("reasoning", "")
                }
            return len(decisions)
        except Exception:
            import traceback; traceback.print_exc()
            return 0

    def reset(self) -> None:
        """Wipe all in-memory state — templates, decisions, samples, metrics."""
        # Clear template store
        self.store._store.clear()

        # Clear reservoir sampler
        self.sampler._reservoirs.clear()
        self.sampler._counts.clear()

        # Reset drain tree
        try:
            self.drain_adapter.drain.drain_tree.root.key_to_child_node.clear()
            self.drain_adapter.drain.id_to_cluster.clear()
        except Exception:
            pass

        # Clear LLM decision cache
        self._llm_decision_cache.clear()
        self._ocsf_classify_cache.clear()
        self._parsed_logs.clear()

        # Clear approver batches if present
        if self.approver and hasattr(self.approver, '_batches'):
            self.approver._batches.clear()

        # Reset metrics counters
        if self.metrics:
            with self.metrics._lock:
                for k in self.metrics._counters:
                    self.metrics._counters[k] = 0

        print("[pipeline] state reset on page load", flush=True)

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
