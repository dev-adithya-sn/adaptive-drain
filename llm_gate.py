"""LLM decision gate: all OpenRouter API logic and candidate-finding lives here."""

from __future__ import annotations
import difflib
import time
import json
import re
import requests


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_FALLBACK_KEEP: dict = {"decision": "keep", "reasoning": "llm_error"}
_FALLBACK_KEEP_DEGRADATION: dict = {
    "decision": "keep",
    "sub_templates": [],
    "reset_template": None,
    "reasoning": "llm_error",
}

_LABEL_VOCAB = {
    "<user.name>", "<user.uid>", "<user.email>",
    "<src_endpoint.ip>", "<src_endpoint.port>", "<src_endpoint.hostname>",
    "<dst_endpoint.ip>", "<dst_endpoint.port>", "<dst_endpoint.hostname>",
    "<auth_protocol>", "<auth_protocol_id>",
    "<http_request.url.path>", "<http_request.url.query>",
    "<http_request.http_method>",
    "<http_response.code>", "<http_response.length>",
    "<database.name>", "<database.table>", "<database.uid>",
    "<actor.user.name>", "<actor.process.name>", "<actor.process.pid>",
    "<app.name>", "<app.version>",
    "<file.path>", "<file.name>", "<file.uid>",
    "<network.bytes>", "<network.packets>",
    "<metadata.uid>", "<metadata.version>",
    "<severity>", "<status>",
    "<duration>", "<count>", "<size>",
    "<timestamp>", "<datetime>",
    "<unknown>",
}

_WILDCARD_LABELING_INSTRUCTION = """\
Wildcard labeling using OCSF field paths:
- labeled_template: copy the template exactly but replace each <*> \
with the most specific OCSF field path from this vocabulary ONLY:
  User fields:      <user.name> <user.uid> <user.email>
  Source network:   <src_endpoint.ip> <src_endpoint.port> <src_endpoint.hostname>
  Dest network:     <dst_endpoint.ip> <dst_endpoint.port> <dst_endpoint.hostname>
  Auth:             <auth_protocol> <auth_protocol_id>
  HTTP request:     <http_request.url.path> <http_request.url.query> <http_request.http_method>
  HTTP response:    <http_response.code> <http_response.length>
  Database:         <database.name> <database.table> <database.uid>
  Actor:            <actor.user.name> <actor.process.name> <actor.process.pid>
  Application:      <app.name> <app.version>
  File:             <file.path> <file.name> <file.uid>
  Network:          <network.bytes> <network.packets>
  Metadata:         <metadata.uid> <metadata.version>
  Generic:          <severity> <status> <duration> <count> <size> <timestamp> <datetime>
  Fallback:         <unknown>
- Use sample logs to determine what each <*> position represents.
- labeled_template MUST have exactly the same number of tokens as \
the original template.
- Non-wildcard tokens must be identical to the original.
- Only use labels from the vocabulary above. Never invent new ones.
- If a position doesn't fit any label, use <unknown>.
Examples:
  "Accepted <*> for <*> from <*> port <*> ssh2"
  → "Accepted <auth_protocol> for <user.name> from <src_endpoint.ip> port <dst_endpoint.port> ssh2"
  "GET <*> HTTP/1.1 <*> <*>"
  → "GET <http_request.url.path> HTTP/1.1 <http_response.code> <http_response.length>"
"""


class LLMGate:
    """Sends template decisions to an OpenRouter-hosted LLM and parses the JSON reply."""

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
    ) -> None:
        self._api_key = api_key
        self._url = "https://openrouter.ai/api/v1/chat/completions"
        self._model = "deepseek/deepseek-chat"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://adaptive-drain.onrender.com",
            "X-Title": "AdaptiveDrain",
        }

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def build_prompt_create(
        self,
        new_template: str,
        samples: list[str],
        candidates: list[dict],
    ) -> str:
        """Build the merge-or-keep prompt for a newly created template."""
        candidate_block = json.dumps(candidates, indent=2) if candidates else "[]"
        sample_block = "\n".join(f"  - {s}" for s in samples[:10]) or "  (none)"
        return (
            "You are a log-template analyst. A new log template was discovered by a log parser.\n"
            "Decide whether it should be MERGED into an existing template or KEPT as a new one.\n\n"
            f"New template:\n  {new_template}\n\n"
            f"Sample log lines that matched this template:\n{sample_block}\n\n"
            f"Existing candidate templates (similarity > 0.5):\n{candidate_block}\n\n"
            "Rules:\n"
            "  - Choose 'merge' only when the new template and a candidate clearly describe the\n"
            "    same event type and merging them would reduce noise without losing specificity.\n"
            "  - Provide a merged_template that generalises both using '<*>' for variable tokens.\n"
            "  - If no good merge exists, choose 'keep'.\n\n"
            f"{_WILDCARD_LABELING_INSTRUCTION}\n"
            "Respond ONLY with valid JSON in this exact shape (no markdown, no extra text):\n"
            '{"decision": "merge" | "keep", "target_cluster_id": "string or null", '
            '"merged_template": "string or null", '
            '"labeled_template": "string — template with named wildcards", "reasoning": "string"}'
        )

    def build_prompt_degradation(
        self,
        template: str,
        wildcard_ratio: float,
        samples: list[str],
    ) -> str:
        """Build the split/reset/keep prompt for an over-generalised template."""
        sample_block = "\n".join(f"  - {s}" for s in samples[:10]) or "  (none)"
        return (
            "You are a log-template analyst. A log template has degraded: too many tokens\n"
            f"have been replaced with wildcards (wildcard ratio: {wildcard_ratio:.2f}).\n\n"
            f"Degraded template:\n  {template}\n\n"
            f"Sample log lines matched by this template:\n{sample_block}\n\n"
            "Choose one action:\n"
            "  'split'  — the samples actually represent multiple distinct event types.\n"
            "             Provide sub_templates (list of specific template strings).\n"
            "  'reset'  — the samples ARE the same event type but the template is too broad.\n"
            "             Provide a tighter reset_template string using '<*>' only where needed.\n"
            "  'keep'   — the wildcard ratio is acceptable given the sample diversity.\n\n"
            f"{_WILDCARD_LABELING_INSTRUCTION}\n"
            "Respond ONLY with valid JSON in this exact shape (no markdown, no extra text):\n"
            '{"decision": "split" | "reset" | "keep", "sub_templates": ["string"] or [], '
            '"reset_template": "string or null", '
            '"labeled_template": "string — template with named wildcards", "reasoning": "string"}'
        )

    def build_prompt_batch(self, templates: list[dict], prior_decisions: dict | None = None) -> str:
        """Build a single prompt reviewing all templates at once."""
        prior_block = ""
        if prior_decisions:
            lines = []
            for cid, pd in prior_decisions.items():
                lines.append(
                    f"  - cluster {cid}: {pd['decision'].upper()} "
                    f"— \"{pd['template']}\" (reason: {pd['reasoning']})"
                )
            prior_block = (
                "PRIOR DECISIONS (from previous evaluation — do NOT contradict these "
                "without strong new evidence. Do NOT merge templates you previously kept separate):\n"
                + "\n".join(lines)
                + "\n\n"
            )

        template_block = ""
        for i, t in enumerate(templates):
            samples_str = "\n".join(
                f"    • {s[:120]}" for s in (t.get("samples") or [])[:3]
            )
            template_block += (
                f"\nTemplate {i+1} (cluster_id={t['cluster_id']}):\n"
                f"  Pattern: \"{t['template']}\"\n"
                f"  Wildcard ratio: {t.get('wildcard_ratio', 0):.0%}\n"
                f"  Samples:\n{samples_str}\n"
            )

        return (
            "You are a log template quality reviewer.\n"
            "Review ALL of the following templates together and decide for each:\n"
            "1. Should it be kept as-is?\n"
            "2. Should it be merged into another template in this list?\n"
            "3. Is it too generic and should be split or reset?\n\n"
            "Also label each <*> wildcard with the correct OCSF field path.\n"
            f"{prior_block}{template_block}\n"
            f"{_WILDCARD_LABELING_INSTRUCTION}\n"
            "Respond ONLY as a JSON object with this exact structure:\n"
            '{\n  "templates": [\n'
            '    {\n'
            '      "cluster_id": "string — must match exactly",\n'
            '      "decision": "keep" | "merge" | "split" | "reset",\n'
            '      "merge_into_cluster_id": "string or null — only if decision=merge",\n'
            '      "merged_template": "string or null — only if decision=merge",\n'
            '      "sub_templates": [],\n'
            '      "reset_template": "string or null — only if decision=reset",\n'
            '      "labeled_template": "string — template with OCSF field path labels",\n'
            '      "reasoning": "string — one sentence"\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "Rules:\n"
            "- Every template in the input MUST appear in the output templates array.\n"
            "- cluster_id values must match exactly what was given.\n"
            "- For merge: merge_into_cluster_id must be a cluster_id in THIS list.\n"
            "- labeled_template must have same token count as original template.\n"
            "- Only use OCSF field path labels from the vocabulary.\n"
            "- Respond with valid JSON only, no markdown fences.\n"
        )

    BATCH_CHUNK_SIZE = 3

    def call_batch(self, templates: list[dict], prior_decisions: dict | None = None) -> list[dict]:
        """Send templates in chunks of BATCH_CHUNK_SIZE. Returns combined list of decisions."""
        if not templates:
            return []

        def _chunk_fallback(chunk: list[dict], reason: str = "llm_error") -> list[dict]:
            return [
                {
                    "cluster_id":       t["cluster_id"],
                    "decision":         "keep",
                    "labeled_template": t["template"].replace("<*>", "<unknown>"),
                    "reasoning":        reason,
                }
                for t in chunk
            ]

        def _call_chunk(chunk: list[dict]) -> list[dict]:
            """POST one chunk; returns parsed+validated decisions for that chunk."""
            prompt = self.build_prompt_batch(chunk, prior_decisions=prior_decisions)
            body = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": "Respond only in JSON."},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 4000,
                "temperature": 0,
            }
            resp = requests.post(self._url, headers=self._headers, json=body, timeout=60)
            resp.raise_for_status()
            data    = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed  = self._parse_json(content)

            results = parsed.get("templates", [])
            if not isinstance(results, list):
                return _chunk_fallback(chunk)

            cluster_ids  = {t["cluster_id"] for t in chunk}
            template_map = {t["cluster_id"]: t["template"] for t in chunk}

            out = []
            for r in results:
                cid = str(r.get("cluster_id", ""))
                if cid not in cluster_ids:
                    continue
                orig    = template_map.get(cid, "")
                labeled = r.get("labeled_template", "")
                if not labeled or not self.validate_labeled_template(orig, labeled):
                    r["labeled_template"] = orig.replace("<*>", "<unknown>")
                r["cluster_id"] = cid
                out.append(r)

            # fill in any templates the LLM forgot
            returned_ids = {r["cluster_id"] for r in out}
            for t in chunk:
                if t["cluster_id"] not in returned_ids:
                    out.append({
                        "cluster_id":       t["cluster_id"],
                        "decision":         "keep",
                        "labeled_template": t["template"].replace("<*>", "<unknown>"),
                        "reasoning":        "not_returned_by_llm",
                    })

            return out

        chunks = [
            templates[i : i + self.BATCH_CHUNK_SIZE]
            for i in range(0, len(templates), self.BATCH_CHUNK_SIZE)
        ]

        all_results: list[dict] = []
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                time.sleep(5)
            try:
                all_results.extend(_call_chunk(chunk))
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    print(f"[llm_gate] 429 on chunk {idx}, retrying after 30s", flush=True)
                    time.sleep(30)
                    try:
                        all_results.extend(_call_chunk(chunk))
                    except Exception:
                        print(f"[llm_gate] 429 retry failed for chunk, using fallback", flush=True)
                        all_results.extend([{"cluster_id": t["cluster_id"], "decision": "keep", "reasoning": "rate_limited", "labeled_template": t["template"]} for t in chunk])
                else:
                    print(f"[llm_gate] call_batch ERROR on chunk {idx}: {exc}", flush=True)
                    import traceback; traceback.print_exc()
                    all_results.extend(_chunk_fallback(chunk))
            except Exception as exc:
                print(f"[llm_gate] call_batch ERROR on chunk {idx}: {exc}", flush=True)
                import traceback; traceback.print_exc()
                all_results.extend(_chunk_fallback(chunk))

        return all_results

    def classify_template(self, template: str, samples: list[str]) -> dict:
        """Ask the LLM to classify an unmatched template into an OCSF class.
        Returns a dict with ocsf_class_uid, ocsf_class_name, activity_id,
        activity_name, category_uid, category_name, severity_id.
        Never raises — returns a generic Unknown fallback on any error.
        """
        _FALLBACK_CLASSIFY = {
            "ocsf_class_uid":  0,
            "ocsf_class_name": "Unknown",
            "activity_id":     0,
            "activity_name":   "Unknown",
            "category_uid":    0,
            "category_name":   "Other",
            "severity_id":     1,
            "matched_rule":    "llm_classified",
        }

        sample_block = "\n".join(f"  - {s[:120]}" for s in samples[:3]) or "  (none)"
        prompt = (
            "You are PanthX AI Normalize Engine, an expert security telemetry analyst "
            "specialized in understanding arbitrary machine logs from unknown systems.\n\n"

            "The logs may originate from ANY source, including but not limited to:\n"
            "- Linux, Windows, macOS\n"
            "- Apache, Nginx, IIS, HAProxy\n"
            "- Kubernetes, Docker, Containers\n"
            "- AWS, Azure, GCP\n"
            "- Active Directory, LDAP, Okta, SSO systems\n"
            "- Firewalls, IDS, IPS, EDR, VPNs\n"
            "- Databases and message queues\n"
            "- SaaS applications\n"
            "- Custom applications and proprietary systems\n"
            "- Completely unknown log formats\n\n"

            "Your objective is to reverse-engineer the telemetry and provide a semantic "
            "understanding of the event.\n\n"

            f"Template:\n{template}\n\n"
            f"Raw sample logs:\n{sample_block}\n\n"

            "Tasks:\n"
            "1. Identify the most likely log source and vendor.\n"
            "2. Estimate confidence in the source identification (0-100).\n"
            "3. Determine whether the event is security relevant.\n"
            "4. Infer the semantic event represented by the logs.\n"
            "5. Map the event to the best OCSF 1.1 class.\n"
            "6. Extract entities, observables and important fields.\n"
            "7. Generate a regex capable of extracting these fields from future logs.\n"
            "8. Infer the schema of future logs of this template.\n"
            "9. Identify indicators useful for threat hunting and detections.\n"
            "10. If uncertain, make the best effort classification and lower confidence accordingly.\n\n"

            "Reasoning Guidelines:\n"
            "- Use ALL sample lines together to infer meaning.\n"
            "- Similar wording across samples often indicates the product type.\n"
            "- Consider timestamps, hostnames, event IDs, process names, URLs, ports, and field ordering.\n"
            "- Prefer semantic understanding over exact vendor matching.\n"
            "- Proprietary logs are expected.\n"
            "- Never invent values that do not appear in the logs.\n"
            "- Unknown values must be empty strings.\n"
            "- If no OCSF class fits, use class_uid 0.\n"
            "- Respond ONLY with valid JSON. Do not include markdown.\n\n"

            "{\n"
            '  "log_source": "",\n'
            '  "vendor": "",\n'
            '  "product": "",\n'
            '  "log_source_confidence": 0,\n'
            '  "security_relevant": true,\n'
            '  "telemetry_type": "",\n'
            '  "semantic_event": "",\n'
            '  "event_description": "",\n'
            '  "severity_id": 1,\n'
            '  "ocsf_class_uid": 0,\n'
            '  "ocsf_class_name": "",\n'
            '  "category_uid": 0,\n'
            '  "category_name": "",\n'
            '  "activity_id": 0,\n'
            '  "activity_name": "",\n'
            '  "regex_pattern": "",\n'
            '  "template_confidence": 0,\n'
            '  "recommended_index": "",\n'
            '  "storage_class": "hot",\n'
            '  "detection_tags": [],\n'
            '  "mitre_attack_techniques": [],\n'
            '  "ioc_candidates": [],\n'
            '  "anomaly_indicators": [],\n'
            '  "entities": {\n'
            '      "users": [],\n'
            '      "hosts": [],\n'
            '      "ips": [],\n'
            '      "domains": [],\n'
            '      "urls": [],\n'
            '      "processes": [],\n'
            '      "files": [],\n'
            '      "services": [],\n'
            '      "containers": [],\n'
            '      "cloud_resources": [],\n'
            '      "email_addresses": [],\n'
            '      "hashes": []\n'
            '  },\n'
            '  "fields": {\n'
            '      "timestamp": "",\n'
            '      "hostname": "",\n'
            '      "username": "",\n'
            '      "src_ip": "",\n'
            '      "dst_ip": "",\n'
            '      "dst_port": "",\n'
            '      "src_port": "",\n'
            '      "protocol": "",\n'
            '      "http_method": "",\n'
            '      "http_path": "",\n'
            '      "http_status": "",\n'
            '      "process_name": "",\n'
            '      "process_id": "",\n'
            '      "parent_process": "",\n'
            '      "command_line": "",\n'
            '      "file_path": "",\n'
            '      "service": "",\n'
            '      "db_name": "",\n'
            '      "container_name": "",\n'
            '      "event_id": "",\n'
            '      "message": ""\n'
            '  }\n'
            "}\n"
        )

        try:
            resp = requests.post(
                self._url,
                headers=self._headers,
                json={
                    "model": self._model,
                    "max_tokens": 1024,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # strip fences if present
            m = _FENCE_RE.search(raw)
            if m:
                raw = m.group(1).strip()
            result = json.loads(raw)
            # validate required fields
            uid = int(result.get("ocsf_class_uid", 0))
            result["ocsf_class_uid"]  = uid
            result["category_uid"]    = int(result.get("category_uid", 0))
            result["activity_id"]     = int(result.get("activity_id", 0))
            result["severity_id"]     = max(1, min(5, int(result.get("severity_id", 1))))
            result["matched_rule"]    = "llm_classified"
            return result
        except Exception as exc:
            print(f"[llm_gate] classify_template ERROR: {exc}", flush=True)
            return _FALLBACK_CLASSIFY

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def call(self, prompt: str, original_template: str = "") -> dict:
        """POST prompt to the LLM and return the parsed JSON decision dict.

        Never raises — returns a safe 'keep' fallback on any error.
        """
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "Respond only in JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 600,
            "temperature": 0,
        }
        try:
            time.sleep(2)
            resp = requests.post(self._url, headers=self._headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content: str = data["choices"][0]["message"]["content"]
            result = self._parse_json(content)
            if original_template:
                result = self._apply_labeled_template(result, original_template)
            return result
        except Exception as exc:
            print(f"[llm_gate] inner ERROR: {exc}", flush=True)
            import traceback; traceback.print_exc()
            print(f"[LLMGate] error: {exc}")
            return dict(_FALLBACK_KEEP)

    def validate_labeled_template(self, original: str, labeled: str) -> bool:
        """Verify labeled_template has same token count and same non-wildcard tokens.

        Returns True if valid, False on any mismatch or error.
        """
        try:
            orig_tokens    = original.split()
            labeled_tokens = labeled.split()

            if len(orig_tokens) != len(labeled_tokens):
                return False

            for orig, lab in zip(orig_tokens, labeled_tokens):
                if orig == "<*>":
                    if lab not in _LABEL_VOCAB:
                        return False
                else:
                    if orig != lab:
                        return False

            return True
        except Exception as exc:
            print(f"[llm_gate] parse ERROR: {exc}", flush=True)
            return False

    def _apply_labeled_template(self, decision: dict, original_template: str) -> dict:
        """Validate and attach labeled_template to decision. Fallback on failure."""
        try:
            labeled = decision.get("labeled_template", "")
            if labeled and self.validate_labeled_template(original_template, labeled):
                decision["labeled_template"] = labeled
            else:
                decision["labeled_template"] = original_template.replace("<*>", "<unknown>")
        except Exception:
            decision["labeled_template"] = original_template.replace("<*>", "<unknown>")
        return decision

    def _parse_json(self, text: str) -> dict:
        """Strip markdown fences then parse JSON; return keep-fallback on failure."""
        stripped = text.strip()
        match = _FENCE_RE.search(stripped)
        if match:
            stripped = match.group(1).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            print(f"[llm_gate] JSON parse ERROR: {exc}", flush=True)
            print(f"[LLMGate] JSON parse error: {exc} | raw: {stripped[:200]}")
            return dict(_FALLBACK_KEEP)

    # ------------------------------------------------------------------
    # Candidate finder
    # ------------------------------------------------------------------

    def find_candidates(
        self,
        new_template: str,
        all_templates: list[dict],
    ) -> list[dict]:
        """Return up to 5 existing templates with SequenceMatcher ratio > 0.5."""
        scored: list[tuple[float, dict]] = []
        for tmpl in all_templates:
            ratio = difflib.SequenceMatcher(
                None, new_template, tmpl["pattern"]
            ).ratio()
            if ratio > 0.5:
                scored.append((ratio, tmpl))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "cluster_id": t["cluster_id"],
                "template": t["pattern"],
                "similarity": round(r, 4),
            }
            for r, t in scored[:5]
        ]
