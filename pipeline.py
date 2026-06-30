"""Main orchestrator — wires all AdaptiveDrain modules and runs batch LLM review."""

from __future__ import annotations
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from approver import HumanApprover
from drain_adapter import DrainAdapter
from fast_path_matcher import ExactMatch, FastPathMatcher, NearMatch, NoMatch
from preprocessor import LogPreprocessor
from llm_gate import LLMGate
from metrics import MetricsCollector
from normalizer import OCSFNormalizer
from persistence import StatePersistence
from reservoir_sampler import ReservoirSampler
from splitter import TemplateSplitter
from template_compiler import CompiledTemplateRegistry
from template_store import NearMatchQueue, TemplateStore


class TemplatePipeline:
    """Log ingestion pipeline with synchronous batch LLM review after upload."""

    CLASSIFY_WORKERS  = 2       # threads for parallel classify_template; real cap is LLMGate.GROQ_CONCURRENCY
    # 10k × ~5 KB/entry ≈ 50 MB — safe on Render free/starter (512 MB).
    # Parsed logs are derived from persisted classify cache + reservoir samples
    # so in-memory-only is fine; raise here to increase the live dashboard window.
    PARSED_LOGS_MAXLEN = 10_000

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
        self._classify_cache_lock = threading.Lock()
        self._classify_executor = ThreadPoolExecutor(max_workers=self.CLASSIFY_WORKERS)
        self._parsed_logs: deque = deque(maxlen=self.PARSED_LOGS_MAXLEN)

        # Fast-path matcher (compiled regex against approved templates)
        self._compiled_registry = CompiledTemplateRegistry()
        self._fast_path_matcher = FastPathMatcher(self._compiled_registry)
        self._near_match_queue  = NearMatchQueue()

    def _bump(self, counter: str, by: int = 1) -> None:
        if self.metrics is not None:
            self.metrics.increment(counter, by)

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def ingest(self, raw_log: str) -> dict:
        """Process a single raw log line.  Never blocks on LLM.

        Three-way dispatch:
          1. EXACT_MATCH  → skip preprocessor + Drain3; extract fields via compiled regex.
          2. NEAR_MATCH   → queue for LLM/human review; hold the log pending decision.
          3. NO_MATCH     → existing Drain3 flow unchanged.
        """
        self._bump("logs_ingested")

        # Fast-path: only active when at least one template has been compiled
        if len(self._compiled_registry) > 0:
            fp = self._fast_path_matcher.match(raw_log)

            if isinstance(fp, ExactMatch):
                self._bump("fast_path_hits")
                self.sampler.add(fp.cluster_id, raw_log)
                t = self.store.get(fp.cluster_id)
                template = (t.labeled_template or t.pattern) if t else ""
                ocsf_label = self.normalizer.normalize(template) if self.normalizer else None
                return {
                    "cluster_id":   fp.cluster_id,
                    "template":     template,
                    "change_type":  "NONE",
                    "original_log": raw_log,
                    "processed_log": raw_log,
                    "extractions":  fp.extracted_fields,
                    "ocsf":         ocsf_label,
                    "ocsf_event":   None,
                    "fast_path":    True,
                }

            if isinstance(fp, NearMatch):
                self._bump("near_match_queued")
                self._near_match_queue.add(
                    raw_log              = raw_log,
                    candidate_cluster_id = fp.candidate_cluster_id,
                    similarity_score     = fp.similarity_score,
                )
                return {
                    "cluster_id":   fp.candidate_cluster_id,
                    "template":     None,
                    "change_type":  "NEAR_MATCH",
                    "original_log": raw_log,
                    "processed_log": raw_log,
                    "extractions":  {},
                    "ocsf":         None,
                    "ocsf_event":   None,
                    "fast_path":    False,
                    "near_match":   True,
                }

        # NO_MATCH (or registry empty): existing Drain3 flow
        pre    = self.preprocessor.process(raw_log)
        result = self.drain_adapter.add_log(pre.processed)
        result["original_log"]  = raw_log
        result["processed_log"] = pre.processed
        result["extractions"]   = pre.extractions

        cluster_id: str  = result["cluster_id"]
        template: str    = result["template"]
        change_type: str = result["change_type"]

        self.sampler.add(cluster_id, raw_log)

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

        result["fast_path"] = False
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

        if decision == "merge":
            target_id  = str(dec.get("merge_into_cluster_id") or "")
            merged_tpl = dec.get("merged_template")
            if target_id and target_id != cid:
                self.store.stage_merge(cid, target_id)
                if merged_tpl:
                    self.drain_adapter.update_template(target_id, merged_tpl.split())
                if self.metrics:
                    self.metrics.increment("templates_merged")
            # Merged template is no longer independently matched
            self._compiled_registry.remove(cid)

        elif decision == "split":
            sub_templates = dec.get("sub_templates", [])
            if sub_templates and len(sub_templates) >= 2:
                new_ids = self.splitter.execute_split(cid, sub_templates, self.sampler.get(cid))
                if new_ids and self.metrics:
                    self.metrics.increment("templates_split")
            self._compiled_registry.remove(cid)

        elif decision == "reset":
            reset_tpl = dec.get("reset_template")
            if reset_tpl:
                self.drain_adapter.update_template(cid, reset_tpl.split())
                t.pattern = reset_tpl

        # Compile/update the fast-path regex for templates that stay ACTIVE
        if decision in ("keep", "reset"):
            label = t.labeled_template or t.pattern
            if label:
                self._compiled_registry.update(cid, label)

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
                with self._classify_cache_lock:
                    cached = self._ocsf_classify_cache.get(template, {})

                # Use OCSF rule match for class/activity/severity if available
                ocsf_label = self.normalizer.normalize(template)
                ocsf_full  = self.normalizer.normalize_full(raw_log, template) if ocsf_label else {}

                # Use regex_pattern from LLM to extract fields if available
                llm_regex = cached.get("regex_pattern", "")
                llm_fields_extracted = {}
                if llm_regex:
                    try:
                        import re as _re
                        m = _re.search(llm_regex, raw_log)
                        if m:
                            llm_fields_extracted = m.groupdict()
                    except Exception:
                        pass

                # Merge LLM fields with regex extracted fields
                llm_fields = {**cached.get("fields", {}), **llm_fields_extracted}

                if not ocsf_full:
                    ocsf_full = {
                        "class_name":    cached.get("ocsf_class_name", "Unknown"),
                        "activity_name": cached.get("activity_name", "Unknown"),
                        "category_name": cached.get("category_name", "Other"),
                        "severity_id":   cached.get("severity_id", 1),
                        "severity":      {1:"Informational",2:"Low",3:"Medium",4:"High",5:"Critical"}.get(cached.get("severity_id",1),"Unknown"),
                        "status":        "",
                        "message":       raw_log,
                        "raw_data":      raw_log,
                    }
                ocsf = ocsf_full or {}
                import re as _re2
                _svc_m = _re2.search(r'\bservice\s+(\w+)', raw_log, _re2.IGNORECASE)
                _svc_direct = _svc_m.group(1) if _svc_m else ""
                entry = {
                    "raw_log":      raw_log,
                    "template":     template,
                    "cluster_id":   cid,
                    "ocsf_class":   ocsf.get("class_name", "") or cached.get("ocsf_class_name", ""),
                    "activity":     ocsf.get("activity_name", "") or cached.get("activity_name", ""),
                    "severity":     ocsf.get("severity", "") or {1:"Informational",2:"Low",3:"Medium",4:"High",5:"Critical"}.get(cached.get("severity_id",1),""),
                    "status":       ocsf.get("status", ""),
                    "username":     llm_fields_extracted.get("username") or (ocsf.get("user") or {}).get("name", "") or llm_fields.get("username", ""),
                    "src_ip":       llm_fields_extracted.get("src_ip") or (ocsf.get("src_endpoint") or {}).get("ip", "") or llm_fields.get("src_ip", ""),
                    "dst_ip":       llm_fields_extracted.get("dst_ip") or (ocsf.get("dst_endpoint") or {}).get("ip", "") or llm_fields.get("dst_ip", ""),
                    "src_port":     llm_fields_extracted.get("src_port") or (ocsf.get("src_endpoint") or {}).get("port", "") or llm_fields.get("src_port", ""),
                    "http_method":  llm_fields_extracted.get("http_method") or (ocsf.get("http_request") or {}).get("http_method", "") or llm_fields.get("http_method", ""),
                    "http_path":    llm_fields_extracted.get("http_path") or (ocsf.get("http_request") or {}).get("url", {}).get("path", "") or llm_fields.get("http_path", ""),
                    "http_status":  llm_fields_extracted.get("http_status") or (ocsf.get("http_response") or {}).get("code", "") or llm_fields.get("http_status", ""),
                    "db_name":      llm_fields_extracted.get("db_name") or (ocsf.get("database") or {}).get("name", "") or llm_fields.get("db_name", ""),
                    "service":      _svc_direct or llm_fields_extracted.get("service", "") or (ocsf.get("service") or {}).get("name", "") or llm_fields.get("service", ""),
                    "log_source":   cached.get("log_source", ""),
                    "vendor":       cached.get("vendor", ""),
                    "semantic_event": cached.get("semantic_event", ""),
                    "security_relevant": cached.get("security_relevant", False),
                    "storage_class":  cached.get("storage_class", "hot"),
                    "mitre_techniques": cached.get("mitre_attack_techniques", []),
                    "detection_tags":  cached.get("detection_tags", []),
                    "telemetry_type":  cached.get("telemetry_type", ""),
                    "entities":     cached.get("entities", {}),
                    "ingested_at":  int(time.time() * 1000),
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

        # Phase 1: execute all decisions first (fast, no LLM calls)
        approved_cluster_ids = []
        for dec in decisions:
            try:
                self._execute_decision(dec)
                approved_cluster_ids.append(str(dec.get("cluster_id", "")))
            except Exception as e:
                print(f"[pipeline] execute_decision error: {e}")

        # Phase 2: classify all NEW unique templates in parallel before parsing
        templates_to_classify = []
        for cid in approved_cluster_ids:
            t = self.store.get(cid)
            if not t:
                continue
            template = t.labeled_template or t.pattern
            with self._classify_cache_lock:
                already_cached = template in self._ocsf_classify_cache
            if not already_cached:
                templates_to_classify.append((cid, template))

        if templates_to_classify:
            def _classify_one(cid_template):
                cid, template = cid_template
                samples = self.sampler.get(cid)[:3]
                try:
                    result = self.llm_gate.classify_template(template, samples)
                except Exception as e:
                    print(f"[pipeline] classify_template error for {cid}: {e}", flush=True)
                    result = {"ocsf_class_uid": 0, "ocsf_class_name": "Unknown"}
                return template, result

            futures = {
                self._classify_executor.submit(_classify_one, item): item
                for item in templates_to_classify
            }
            try:
                for future in as_completed(futures, timeout=60):
                    try:
                        template, result = future.result()
                        with self._classify_cache_lock:
                            self._ocsf_classify_cache[template] = result
                    except Exception as e:
                        print(f"[pipeline] classify future error: {e}", flush=True)
            except TimeoutError:
                print("[pipeline] classify timed out after 60s, continuing with partial results", flush=True)

        # Phase 3: parse all logs now that classify cache is warm
        for cid in approved_cluster_ids:
            try:
                self._parse_cluster_logs(cid)
            except Exception as e:
                print(f"[pipeline] parse_cluster_logs error: {e}")

        self.approver.clear_batch(batch_id)
        if self.persistence:
            self.persistence.save(self.store, self.sampler, self._ocsf_classify_cache)

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

    def reevaluate_all(self) -> int:
        """Re-evaluate ACTIVE templates via batch LLM call. Returns count queued."""
        try:
            to_review = []
            for t in self.store.all_active():
                tokens = t.pattern.split()
                wc     = sum(1 for tok in tokens if tok == "<*>")
                to_review.append({
                    "cluster_id":    t.cluster_id,
                    "template":      t.pattern,
                    "samples":       self.sampler.get(t.cluster_id)[:3],
                    "wildcard_ratio": wc / max(len(tokens), 1),
                })

            print(f"[reevaluate] active templates: {len(to_review)}, to_review: {len(to_review)}", flush=True)

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

    # ------------------------------------------------------------------
    # Near-match LLM review + execution
    # ------------------------------------------------------------------

    def batch_near_match_review(self) -> dict:
        """Send all pending near-match items to the LLM for triage.

        Reuses the same Groq semaphore and approver infrastructure as
        ``batch_review()``.  Returns the batch summary dict.
        """
        import uuid as _uuid

        items = self._near_match_queue.get_pending()
        if not items:
            return {"batch_id": "", "decisions": [], "queued": 0}

        # Build templates_info for the LLM prompt
        templates_info: dict[str, dict] = {}
        for item in items:
            cid = item.candidate_cluster_id
            if cid not in templates_info:
                t = self.store.get(cid)
                if t:
                    templates_info[cid] = {
                        "template":         t.pattern,
                        "labeled_template": t.labeled_template or t.pattern,
                        "samples":          self.sampler.get(cid)[:3],
                    }

        item_dicts = [
            {
                "item_id":              item.item_id,
                "raw_log":              item.raw_log,
                "candidate_cluster_id": item.candidate_cluster_id,
                "similarity_score":     item.similarity_score,
            }
            for item in items
        ]

        self._bump("llm_calls")
        decisions = self.llm_gate.call_near_match_batch(item_dicts, templates_info)
        batch_id  = str(_uuid.uuid4())

        if self.approver and hasattr(self.approver, "set_batch"):
            # Near-match decisions go through the same approver queue as template
            # decisions so the existing UI can render them.
            self.approver.set_batch(
                batch_id,
                decisions,
                [{"cluster_id": d["item_id"], **d} for d in decisions],
            )
        else:
            # No approver: auto-execute
            for dec in decisions:
                self._execute_near_match_decision(dec)

        return {"batch_id": batch_id, "decisions": decisions, "queued": len(decisions)}

    def _execute_near_match_decision(self, dec: dict) -> None:
        """Apply an approved near-match decision.

        ``same_template_fix_regex``: recompile the candidate template's regex
        from the corrected labeled_template.

        ``new_template_send_to_drain``: route the raw_log through the existing
        Drain3 ingest path as a fresh pending template.
        """
        item_id  = dec.get("item_id", "")
        decision = dec.get("decision", "new_template_send_to_drain")
        item     = self._near_match_queue.get(item_id)
        if item is None:
            return

        if decision == "same_template_fix_regex":
            corrected = dec.get("corrected_template")
            if corrected:
                cid = item.candidate_cluster_id
                t   = self.store.get(cid)
                if t:
                    t.labeled_template = corrected
                    # Recompile the fast-path regex for this cluster
                    self._compiled_registry.update(cid, corrected)
            self._near_match_queue.approve(
                item_id,
                llm_decision       = decision,
                corrected_template = dec.get("corrected_template"),
            )

        elif decision == "new_template_send_to_drain":
            # Re-ingest through existing Drain3 pipeline
            # Temporarily remove from fast-path check to avoid re-triggering near-match
            raw_log = item.raw_log
            self._near_match_queue.reject(item_id)
            pre    = self.preprocessor.process(raw_log)
            result = self.drain_adapter.add_log(pre.processed)
            cid    = str(result.get("cluster_id", ""))
            tpl    = result.get("template", "")
            ct     = result.get("change_type", "NONE")
            if cid:
                self.sampler.add(cid, raw_log)
            if ct == "CREATE":
                self.store.register(cid, tpl)
                self.store.add_version(cid, tpl, trigger_log=raw_log)
            elif ct == "UPDATE":
                self.store.add_version(cid, tpl, trigger_log=raw_log)

        self._near_match_queue.remove(item_id)

    def execute_near_match_batch(self, batch_id: str) -> dict:
        """Execute all approved near-match decisions for a batch.

        Called by the server when the human approves a near-match batch
        (same approval endpoint pattern as ``execute_batch``).
        """
        if not self.approver or not hasattr(self.approver, "get_batch"):
            return {"executed": 0}

        decisions = self.approver.get_batch(batch_id)
        executed  = 0
        for dec in decisions:
            try:
                self._execute_near_match_decision(dec)
                executed += 1
            except Exception as e:
                print(f"[pipeline] execute_near_match_decision error: {e}")

        self.approver.clear_batch(batch_id)
        if self.persistence:
            self.persistence.save(self.store, self.sampler, self._ocsf_classify_cache)

        return {"executed": executed}

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

        # Clear fast-path state
        self._compiled_registry = CompiledTemplateRegistry()
        self._fast_path_matcher = FastPathMatcher(self._compiled_registry)
        self._near_match_queue.clear()

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
            return self.persistence.save(self.store, self.sampler, self._ocsf_classify_cache)
        return False

    def load(self) -> bool:
        if self.persistence is not None:
            ok = self.persistence.load(self.store, self.sampler, self._ocsf_classify_cache)
            if ok:
                # Rebuild the compiled registry from persisted templates
                n = self._compiled_registry.rebuild_from_store(self.store)
                print(f"[pipeline] rebuilt compiled registry: {n} templates", flush=True)
                # Repopulate _parsed_logs from persisted samples + classify cache so the
                # OCSF Events panel is non-empty after a server restart.  Only active
                # templates that have already been LLM-reviewed qualify (_parse_cluster_logs
                # early-returns for llm_decision=None anyway, but being explicit here avoids
                # iterating over unreviewed templates at all).
                for t in self.store.all_active():
                    if getattr(t, "llm_decision", None) is not None:
                        self._parse_cluster_logs(t.cluster_id)
                print(f"[pipeline] repopulated {len(self._parsed_logs)} parsed log entries", flush=True)
            return ok
        return False
