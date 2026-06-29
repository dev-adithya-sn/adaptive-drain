"""Pre-processing layer: normalises high-cardinality tokens before Drain3."""

import re
from dataclasses import dataclass, field


@dataclass
class PreprocessResult:
    processed:   str
    original:    str
    extractions: dict = field(default_factory=dict)


class LogPreprocessor:
    """Replaces concrete values with typed placeholders so Drain3 clusters on structure."""

    # Each entry is (pattern, placeholder, extraction_key[, capture_group]).
    # capture_group defaults to 0 (full match). Use 1 when the value to extract
    # is in group(1) but the placeholder replaces the whole match.
    MASKS = [
        (r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}',
         '<TIMESTAMP>', 'TIMESTAMP'),
        (r'\[\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}\]',
         '[<TIMESTAMP>]', 'TIMESTAMP'),
        (r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b',
         '<DATETIME>', 'TIMESTAMP'),
        (r'\b\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\b',
         '<DATETIME>', 'TIMESTAMP'),
        (r'\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b',
         '<TIME>', 'TIMESTAMP'),
        (r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
         '<IPV6>', 'IP'),
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
         '<IP>', 'IP'),
        (r'\b(?:[0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}\b',
         '<MAC>', 'MAC'),
        (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
         '<UUID>', 'ID'),
        (r'\b[0-9a-fA-F]{16,}\b',
         '<HEX>', 'ID'),
        (r'\bport[= ](\d{1,5})\b',
         'port <PORT>', 'PORT'),
        # Username after SSH/auth "for <name> from/at/on/(" — IP already replaced above
        (r'\bfor\s+([a-zA-Z][a-zA-Z0-9_.\-]{1,32})\s+(?=from|at|on|\()',
         'for <USERNAME> ', 'USERNAME', 1),
        # Username after "user=" or "user <name>"
        (r'\buser[= ]([a-zA-Z][a-zA-Z0-9_.\-]{1,32})\b',
         'user <USERNAME>', 'USERNAME', 1),
        # Password value — requires "=" to avoid false matches on "password for user"
        (r'\bpassword=\S+',
         'password <PASSWORD>', 'PASSWORD'),
        (r'\bpid[= ](\d+)\b',
         'pid <PID>', 'PID'),
        (r'\b0x[0-9a-fA-F]{4,}\b',
         '<ADDR>', 'ADDR'),
        (r'(?:/[\w.\-]+){2,}',
         '<PATH>', 'PATH'),
        (r'https?://\S+',
         '<URL>', 'URL'),
        (r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b',
         '<EMAIL>', 'EMAIL'),
        (r'\b\d{7,}\b',
         '<NUM>', 'NUM'),
        (r'\b\d+(?:\.\d+)?(?:ms|us|ns|s|KB|MB|GB|TB|B)\b',
         '<SIZE>', 'SIZE'),
    ]

    def __init__(self) -> None:
        self._compiled = []
        for mask in self.MASKS:
            pattern, placeholder, key = mask[0], mask[1], mask[2]
            capture_group = mask[3] if len(mask) > 3 else 0
            self._compiled.append((re.compile(pattern), placeholder, key, capture_group))

    def process(self, raw_log: str) -> PreprocessResult:
        """Apply all masks in order. Never raises — returns original on any error."""
        try:
            text        = raw_log
            extractions: dict = {}

            for regex, placeholder, key, capture_group in self._compiled:
                def replacer(m, ph=placeholder, k=key, ex=extractions, cg=capture_group):
                    val = m.group(cg) if cg else m.group(0)
                    ex.setdefault(k, []).append(val)
                    return ph
                text = regex.sub(replacer, text)

            return PreprocessResult(processed=text, original=raw_log, extractions=extractions)
        except Exception:
            return PreprocessResult(processed=raw_log, original=raw_log, extractions={})

    def batch(self, logs: list[str]) -> list[PreprocessResult]:
        return [self.process(log) for log in logs]
