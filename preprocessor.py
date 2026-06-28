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
        self._compiled = [
            (re.compile(pattern), placeholder, key)
            for pattern, placeholder, key in self.MASKS
        ]

    def process(self, raw_log: str) -> PreprocessResult:
        """Apply all masks in order. Never raises — returns original on any error."""
        try:
            text        = raw_log
            extractions: dict = {}

            for regex, placeholder, key in self._compiled:
                def replacer(m, ph=placeholder, k=key, ex=extractions):
                    ex.setdefault(k, []).append(m.group(0))
                    return ph
                text = regex.sub(replacer, text)

            return PreprocessResult(processed=text, original=raw_log, extractions=extractions)
        except Exception:
            return PreprocessResult(processed=raw_log, original=raw_log, extractions={})

    def batch(self, logs: list[str]) -> list[PreprocessResult]:
        return [self.process(log) for log in logs]
