"""Compile LLM-labeled Drain3 templates into strict + loose regex matchers.

A labeled_template looks like:
  "Accepted <auth_protocol> for <user.name> from <src_endpoint.ip> port <dst_endpoint.port> ssh2"

strict_regex is anchored and uses named capture groups keyed by OCSF field path (dots →
underscores, disambiguated with a positional suffix so duplicate labels don't clash).

loose_regex is built from the literal-token subsequence only (wildcards stripped), joined by
".*?" so it matches any log that contains all literal tokens in order with arbitrary content
between them — used exclusively for near-match candidate detection.
"""

from __future__ import annotations

import re
import dataclasses


@dataclasses.dataclass
class CompiledTemplate:
    """Compiled regex matchers for a single active template."""

    cluster_id:     str
    strict_regex:   re.Pattern
    loose_regex:    re.Pattern
    token_count:    int             # number of literal (non-wildcard) tokens
    label_map:      dict[str, str]  # regex group name → OCSF field path
    literal_tokens: list[str]       # literal tokens in order, for similarity scoring


_WILDCARD_RE = re.compile(r'^<([^>]*)>$')


class TemplateCompiler:
    """Compiles a labeled_template string into a CompiledTemplate."""

    def compile(self, cluster_id: str, labeled_template: str) -> CompiledTemplate:
        """Compile *labeled_template* for *cluster_id*.

        Wildcards are any <…> token.  Named capture groups use the inner label
        with dots replaced by underscores plus a positional index to guarantee
        uniqueness (e.g. ``user_name_0``, ``src_endpoint_ip_1``).
        """
        tokens       = labeled_template.split()
        strict_parts: list[str] = []
        loose_parts:  list[str] = []
        label_map:    dict[str, str] = {}
        literal_tokens: list[str] = []
        field_idx = 0

        for token in tokens:
            m = _WILDCARD_RE.match(token)
            if m:
                label    = m.group(1)  # e.g. "user.name", "*", "unknown", ""
                safe     = re.sub(r'[^a-zA-Z0-9]', '_', label) or "field"
                if safe[0].isdigit():
                    safe = f"f_{safe}"
                group_name = f"{safe}_{field_idx}"
                field_idx += 1

                ocsf_path = label if label not in ("*", "unknown", "") else ""
                label_map[group_name] = ocsf_path

                # \S+ matches exactly one whitespace-free token, consistent with
                # Drain3's own tokenisation model where each wildcard position
                # represents a single variable token.  This keeps strict matching
                # tight while loose_regex (which uses .*?) handles near-match.
                strict_parts.append(f"(?P<{group_name}>\\S+)")
                loose_parts.append(None)  # placeholder; will be omitted from loose pattern
            else:
                strict_parts.append(re.escape(token))
                loose_parts.append(re.escape(token))
                literal_tokens.append(token)

        strict_pattern = r"\s+".join(strict_parts)
        strict_regex   = re.compile(f"^{strict_pattern}$")

        # Loose regex: literal tokens in sequence with ".*?" between them.
        # Templates with no literal tokens get a never-matching pattern.
        literal_escaped = [p for p in loose_parts if p is not None]
        if literal_escaped:
            loose_pattern = ".*?".join(literal_escaped)
            loose_regex   = re.compile(loose_pattern, re.DOTALL)
        else:
            loose_regex = re.compile(r"(?!)")  # never matches

        return CompiledTemplate(
            cluster_id    = cluster_id,
            strict_regex  = strict_regex,
            loose_regex   = loose_regex,
            token_count   = len(literal_tokens),
            label_map     = label_map,
            literal_tokens= literal_tokens,
        )


class CompiledTemplateRegistry:
    """Incremental in-memory registry of compiled templates.

    Updated whenever a template becomes active (approved + labeled).
    Rebuilt from persisted labeled_template strings on process startup via
    ``rebuild_from_store()``.
    """

    def __init__(self) -> None:
        self._compiled: dict[str, CompiledTemplate] = {}
        self._compiler = TemplateCompiler()

    def update(self, cluster_id: str, labeled_template: str) -> None:
        """Compile (or recompile) the template for *cluster_id*."""
        ct = self._compiler.compile(cluster_id, labeled_template)
        self._compiled[cluster_id] = ct

    def remove(self, cluster_id: str) -> None:
        """Drop the compiled entry for *cluster_id* (e.g. on merge/hard-delete)."""
        self._compiled.pop(cluster_id, None)

    def get(self, cluster_id: str) -> CompiledTemplate | None:
        return self._compiled.get(cluster_id)

    def get_all_sorted(self) -> list[CompiledTemplate]:
        """Return templates sorted by specificity (most literal tokens first)."""
        return sorted(self._compiled.values(), key=lambda c: c.token_count, reverse=True)

    def __len__(self) -> int:
        return len(self._compiled)

    def rebuild_from_store(self, store: object) -> int:
        """Populate the registry from all active templates in *store* that have
        a ``labeled_template`` and a completed ``llm_decision``.

        Returns the number of templates compiled.
        """
        compiled = 0
        for t in store.all_active():
            label = getattr(t, "labeled_template", None)
            if label and getattr(t, "llm_decision", None) is not None:
                self.update(t.cluster_id, label)
                compiled += 1
        return compiled
