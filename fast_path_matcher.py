"""Three-way fast-path matcher for compiled Drain3 templates.

Matching priority: iterate templates by token_count descending (most specific
first), try strict_regex across ALL templates before falling back to
loose_regex for near-match detection.
"""

from __future__ import annotations

import dataclasses
import difflib
from enum import Enum
from typing import ClassVar

from template_compiler import CompiledTemplate, CompiledTemplateRegistry


# ---------------------------------------------------------------------------
# Match result variants
# ---------------------------------------------------------------------------

class MatchKind(Enum):
    EXACT_MATCH = "EXACT_MATCH"
    NEAR_MATCH  = "NEAR_MATCH"
    NO_MATCH    = "NO_MATCH"


@dataclasses.dataclass
class ExactMatch:
    """Log matched a strict_regex cleanly; fields extracted from named groups."""
    kind:             ClassVar[MatchKind] = MatchKind.EXACT_MATCH
    cluster_id:       str
    extracted_fields: dict[str, str]  # OCSF field path → captured value


@dataclasses.dataclass
class NearMatch:
    """Log matched a loose_regex with similarity >= threshold but failed strict."""
    kind:                  ClassVar[MatchKind] = MatchKind.NEAR_MATCH
    candidate_cluster_id:  str
    raw_log:               str
    similarity_score:      float


@dataclasses.dataclass
class NoMatch:
    """No strict or qualifying near-match found; route to Drain3."""
    kind: ClassVar[MatchKind] = MatchKind.NO_MATCH


MatchResult = ExactMatch | NearMatch | NoMatch


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class FastPathMatcher:
    """Match a raw log line against the compiled template registry.

    Pass 1 — strict:  try every compiled template's strict_regex (anchored,
                       named groups).  First match wins; returns ExactMatch.
    Pass 2 — loose:   for every template whose loose_regex matches the log,
                       compute the fraction of literal tokens that appear in
                       order inside the log (SequenceMatcher on token lists).
                       If the best score >= threshold, return NearMatch.
    Otherwise         return NoMatch.
    """

    DEFAULT_THRESHOLD: float = 0.85

    def __init__(
        self,
        registry:              CompiledTemplateRegistry,
        near_match_threshold:  float = DEFAULT_THRESHOLD,
    ) -> None:
        self._registry  = registry
        self._threshold = near_match_threshold

    def match(self, raw_log: str) -> MatchResult:
        templates = self._registry.get_all_sorted()

        # Pass 1: strict match across all templates
        for ct in templates:
            m = ct.strict_regex.match(raw_log)
            if m:
                extracted_fields: dict[str, str] = {}
                for group_name, ocsf_path in ct.label_map.items():
                    value = m.group(group_name)
                    if value is not None and ocsf_path:
                        extracted_fields[ocsf_path] = value
                return ExactMatch(
                    cluster_id=ct.cluster_id,
                    extracted_fields=extracted_fields,
                )

        # Pass 2: loose-match for near-match detection
        best_score: float = -1.0
        best_ct:    CompiledTemplate | None = None

        for ct in templates:
            if ct.token_count == 0:
                continue  # no literals → too generic to near-match
            if not ct.loose_regex.search(raw_log):
                continue
            score = _literal_token_similarity(ct.literal_tokens, raw_log)
            if score > best_score:
                best_score = score
                best_ct    = ct

        if best_ct is not None and best_score >= self._threshold:
            return NearMatch(
                candidate_cluster_id=best_ct.cluster_id,
                raw_log=raw_log,
                similarity_score=best_score,
            )

        return NoMatch()


# ---------------------------------------------------------------------------
# Similarity helper
# ---------------------------------------------------------------------------

def _literal_token_similarity(literal_tokens: list[str], raw_log: str) -> float:
    """Return the fraction of *literal_tokens* that appear in order in the log.

    Uses SequenceMatcher on token lists (not full strings) so that small
    changes in variable values don't inflate or deflate the score.
    """
    if not literal_tokens:
        return 0.0
    log_tokens = raw_log.split()
    sm = difflib.SequenceMatcher(None, literal_tokens, log_tokens, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(literal_tokens)
