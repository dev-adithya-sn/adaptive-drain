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
            "Respond ONLY with valid JSON in this exact shape (no markdown, no extra text):\n"
            '{"decision": "merge" | "keep", "target_cluster_id": "string or null", '
            '"merged_template": "string or null", "reasoning": "string"}'
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
            "Respond ONLY with valid JSON in this exact shape (no markdown, no extra text):\n"
            '{"decision": "split" | "reset" | "keep", "sub_templates": ["string"] or [], '
            '"reset_template": "string or null", "reasoning": "string"}'
        )

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def call(self, prompt: str) -> dict:
        """POST prompt to OpenRouter and return the parsed JSON decision dict.

        Never raises — returns a safe 'keep' fallback on any error.
        """
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "Respond only in JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,
            "temperature": 0,
        }
        try:
            time.sleep(5)
            resp = requests.post(self._url, headers=self._headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content: str = data["choices"][0]["message"]["content"]
            return self._parse_json(content)
        except Exception as exc:
            print(f"[LLMGate] error: {exc}")
            return dict(_FALLBACK_KEEP)

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
