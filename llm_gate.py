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
    "<ip>", "<port>", "<user>", "<username>", "<password>",
    "<auth_method>", "<host>", "<hostname>", "<domain>", "<url>",
    "<path>", "<method>", "<status_code>", "<bytes>", "<duration>",
    "<timestamp>", "<date>", "<time_val>", "<pid>", "<process>",
    "<service>", "<table>", "<database>", "<query>", "<rows>",
    "<key>", "<value>", "<id>", "<hash>", "<email>", "<phone>",
    "<count>", "<size>", "<level>", "<message>", "<error>", "<unknown>",
}

_WILDCARD_LABELING_INSTRUCTION = """\
Wildcard labeling rules:
- The template uses <*> as a placeholder for variable values.
- For labeled_template: copy the template exactly but replace each <*> \
with the most specific semantic label from this vocabulary:
  <ip>, <port>, <user>, <username>, <password>, <auth_method>,
  <host>, <hostname>, <domain>, <url>, <path>, <method>,
  <status_code>, <bytes>, <duration>, <timestamp>, <date>, <time_val>,
  <pid>, <process>, <service>, <table>, <database>, <query>,
  <rows>, <key>, <value>, <id>, <hash>, <email>, <phone>,
  <count>, <size>, <level>, <message>, <error>, <unknown>
- Use context from the sample logs to determine each position's type.
- If a position varies and doesn't fit any label, use <unknown>.
- labeled_template must have the same number of placeholders as the \
original template (count of <*> must equal count of <label> tokens).
- Non-wildcard tokens must remain exactly as they appear in the template.
"""


class LLMGate:
    """Sends template decisions to an OpenRouter-hosted LLM and parses the JSON reply."""

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
    ) -> None:
        self._api_key = api_key
        self._url = "https://api.groq.com/openai/v1/chat/completions"
        self._model = "llama-3.1-8b-instant"  # or llama-3.3-70b-versatile
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "adaptive-drain",
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
            "Quality scoring: evaluate the template and set quality.score 0-10 (10=perfect),\n"
            "  quality.issues from: too_generic, timestamp_not_masked, over_wildcarded,\n"
            "  missing_field_labels, should_split, looks_good\n"
            "  quality.suggestion: one-line improvement if score < 8, else empty string.\n\n"
            "Respond ONLY with valid JSON in this exact shape (no markdown, no extra text):\n"
            '{"decision": "merge" | "keep", "target_cluster_id": "string or null", '
            '"merged_template": "string or null", '
            '"labeled_template": "string — template with named wildcards", "reasoning": "string", '
            '"quality": {"score": 0, "issues": [], "suggestion": ""}}'
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
            "Quality scoring: evaluate the template and set quality.score 0-10 (10=perfect),\n"
            "  quality.issues from: too_generic, timestamp_not_masked, over_wildcarded,\n"
            "  missing_field_labels, should_split, looks_good\n"
            "  quality.suggestion: one-line improvement if score < 8, else empty string.\n\n"
            "Respond ONLY with valid JSON in this exact shape (no markdown, no extra text):\n"
            '{"decision": "split" | "reset" | "keep", "sub_templates": ["string"] or [], '
            '"reset_template": "string or null", '
            '"labeled_template": "string — template with named wildcards", "reasoning": "string", '
            '"quality": {"score": 0, "issues": [], "suggestion": ""}}'
        )

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
            time.sleep(5)
            resp = requests.post(self._url, headers=self._headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content: str = data["choices"][0]["message"]["content"]
            result = self._parse_json(content)
            if original_template:
                result = self._apply_labeled_template(result, original_template)
            return result
        except Exception as exc:
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
        except Exception:
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
