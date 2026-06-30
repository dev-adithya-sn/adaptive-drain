"""AdaptiveDrain: LLM-powered adaptive log template pipeline built on top of Drain3."""

from pipeline import TemplatePipeline
from template_store import TemplateStore, TemplateStatus, ManagedTemplate, NearMatchQueue, NearMatchItem
from drain_adapter import DrainAdapter
from splitter import TemplateSplitter
from normalizer import OCSFNormalizer
from persistence import StatePersistence
from metrics import MetricsCollector
from approver import HumanApprover, WebApprover
from ocsf_event_builder import OCSFEventBuilder
from preprocessor import LogPreprocessor
from template_compiler import TemplateCompiler, CompiledTemplate, CompiledTemplateRegistry
from fast_path_matcher import FastPathMatcher, ExactMatch, NearMatch, NoMatch, MatchKind

__all__ = [
    "TemplatePipeline",
    "TemplateStore",
    "TemplateStatus",
    "ManagedTemplate",
    "NearMatchQueue",
    "NearMatchItem",
    "DrainAdapter",
    "TemplateSplitter",
    "OCSFNormalizer",
    "StatePersistence",
    "MetricsCollector",
    "HumanApprover",
    "WebApprover",
    "OCSFEventBuilder",
    "LogPreprocessor",
    "TemplateCompiler",
    "CompiledTemplate",
    "CompiledTemplateRegistry",
    "FastPathMatcher",
    "ExactMatch",
    "NearMatch",
    "NoMatch",
    "MatchKind",
]
