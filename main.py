"""Demo entrypoint for AdaptiveDrain — exercises all modules end-to-end.

Usage:
    cd /path/to/parent/of/adaptive_drain
    OPENROUTER_API_KEY=sk-or-... python -m adaptive_drain.main
    python -m adaptive_drain.main   # LLM gate no-ops, everything else runs
"""

from __future__ import annotations

import os
import time

from drain3 import TemplateMiner
from dotenv import load_dotenv

from pipeline import TemplatePipeline
from normalizer import OCSFNormalizer
from persistence import StatePersistence
from metrics import MetricsCollector
from approver import HumanApprover

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Path to ocsf_map.yml — relative to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
OCSF_MAP_PATH = os.path.join(_HERE, "ocsf_map.yml")
STATE_PATH = os.path.join(_HERE, "state.json")

SAMPLE_LOGS = [
    # SSH success variants
    "Accepted password for alice from 192.168.1.10 port 22 ssh2",
    "Accepted password for bob from 10.0.0.5 port 22 ssh2",
    "Accepted publickey for carol from 172.16.0.1 port 54321 ssh2",
    "Accepted password for dave from 203.0.113.42 port 22 ssh2",
    # SSH failure variants
    "Failed password for invalid user mallory from 198.51.100.7 port 31337 ssh2",
    "Failed password for root from 198.51.100.7 port 54322 ssh2",
    "Failed password for invalid user admin from 192.0.2.1 port 22 ssh2",
    "Connection closed by 10.10.10.10 port 12345 [preauth]",
    # HTTP GET
    'GET /api/v1/users HTTP/1.1 200 1024 "Mozilla/5.0"',
    'GET /api/v1/products HTTP/1.1 200 2048 "curl/7.68.0"',
    'GET /api/v1/orders/123 HTTP/1.1 404 256 "Mozilla/5.0"',
    'GET /health HTTP/1.1 200 12 "kube-probe/1.25"',
    # HTTP POST
    'POST /api/v1/users HTTP/1.1 201 128 "Mozilla/5.0"',
    'POST /api/v1/login HTTP/1.1 200 256 "Mozilla/5.0"',
    'POST /api/v1/login HTTP/1.1 401 64 "curl/7.68.0"',
    # HTTP errors
    'GET /api/v1/missing HTTP/1.1 500 64 "Mozilla/5.0"',
    'POST /api/v1/data HTTP/1.1 503 32 "python-requests/2.28"',
    # Database
    "DB connection established host=db-primary port=5432 user=app_user db=production latency=2ms",
    "DB connection established host=db-replica port=5432 user=readonly db=production latency=5ms",
    "DB query executed table=users rows=42 duration=12ms",
    "DB query executed table=orders rows=1 duration=3ms",
    "DB connection closed host=db-primary port=5432 user=app_user session_duration=30s",
    # Auth generic
    "authentication failed for user root attempt=3",
    "authentication failed for user admin attempt=1",
    "user jenkins authenticated successfully",
    # Service lifecycle
    "service nginx started successfully",
    "service nginx stopped",
    "service postgres started successfully",
    # App noise (no OCSF match expected)
    "Cache hit key=session:abc123 ttl=3540s",
    "Cache miss key=session:xyz789",
    "Rate limit exceeded ip=203.0.113.7 endpoint=/api/v1/login count=10 window=60s",
    "ERROR: Timeout connecting to redis://cache:6379 after 5000ms attempt=1",
    "pod/web-7d8f9b-abc started in namespace production node=node-01",
    "Cron job cleanup-logs finished duration=12.3s records_deleted=5000",
    # Repeats to exercise NONE path and confirm_merge_hit
    "Accepted password for alice from 192.168.1.10 port 22 ssh2",
    "Failed password for root from 198.51.100.7 port 54322 ssh2",
    "DB query executed table=sessions rows=0 duration=1ms",
    'GET /api/v1/users HTTP/1.1 200 1024 "Mozilla/5.0"',
]


def main() -> None:
    print("=" * 65)
    print("AdaptiveDrain — full demo (OCSF + metrics + persistence)")
    print("=" * 65)

    if not GROQ_API_KEY:
        print("[main] No GROQ_API_KEY — LLM gate will fallback to keep; all other features active.\n")

    # --- Wire up all 4 modules ---
    normalizer  = OCSFNormalizer(OCSF_MAP_PATH)
    persistence = StatePersistence(STATE_PATH)
    metrics     = MetricsCollector(emit_interval_seconds=999)  # manual snapshot only
    metrics.start()
    approver    = HumanApprover(auto_approve=False)

    drain    = TemplateMiner()
    pipeline = TemplatePipeline(
        drain_instance=drain,
        openrouter_api_key=GROQ_API_KEY,
        confirm_threshold=3,
        normalizer=normalizer,
        persistence=persistence,
        metrics=metrics,
        approver=approver,
    )

    # --- Load persisted state if available ---
    if persistence.exists():
        ok = pipeline.load()
        print(f"[persistence] loaded previous state: {ok}\n")

    # --- Ingest logs ---
    print("Human approval mode ON — you will be prompted for each LLM decision.\n")
    ocsf_hits   = 0
    ocsf_misses = 0

    for i, log in enumerate(SAMPLE_LOGS, 1):
        result = pipeline.ingest(log)
        ocsf   = result["ocsf"]
        ocsf_label = ocsf["ocsf_class_name"] if ocsf else "—"

        if ocsf:
            ocsf_hits += 1
        else:
            ocsf_misses += 1

        print(
            f"[{i:02d}] {result['change_type']:6s} | "
            f"cluster={result['cluster_id']:>3s} | "
            f"ocsf={ocsf_label:<30s} | "
            f"{result['template'][:55]}"
        )

    # --- Wait for LLM worker (long, to allow human review of each decision) ---
    print(f"\n--- Ingestion done. Waiting up to 300s for LLM worker + human review ---")
    time.sleep(300)

    # --- Stats ---
    print("\n--- Pipeline stats ---")
    print(pipeline.stats())

    print("\n--- OCSF coverage ---")
    total = ocsf_hits + ocsf_misses
    print(f"  matched : {ocsf_hits}/{total} ({100*ocsf_hits//total}%)")
    print(f"  unmatched: {ocsf_misses}/{total}")

    print("\n--- Metrics snapshot ---")
    snap = metrics.snapshot()
    snap.pop("timestamp")
    for k, v in snap.items():
        print(f"  {k:<25s}: {v}")

    # --- Save state ---
    ok = pipeline.save()
    print(f"\n[persistence] state saved: {ok}  →  {STATE_PATH}")

    metrics.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
