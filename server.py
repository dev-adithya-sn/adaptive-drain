"""Flask web server wrapping the AdaptiveDrain pipeline."""

import os
import sys
import threading
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from drain3 import TemplateMiner
from dotenv import load_dotenv

from pipeline import TemplatePipeline
from normalizer import OCSFNormalizer
from persistence import StatePersistence
from metrics import MetricsCollector
from approver import WebApprover

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_HERE = os.path.dirname(os.path.abspath(__file__))

approver    = WebApprover()
normalizer  = OCSFNormalizer(os.path.join(_HERE, "ocsf_map.yml"))
persistence = StatePersistence(os.path.join(_HERE, "state.json"))
metrics     = MetricsCollector(emit_interval_seconds=999)
metrics.start()

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

if persistence.exists():
    pipeline.load()

_results_store: dict = {}
_results_lock = threading.Lock()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400

    import uuid
    session_id = str(uuid.uuid4())
    lines = file.read().decode("utf-8", errors="replace").splitlines()
    lines = [l for l in lines if l.strip()][:200_000]

    results = []
    for line in lines:
        result = pipeline.ingest(line)
        results.append({
            "log":              line[:120],
            "change_type":      result.get("change_type"),
            "cluster_id":       result.get("cluster_id"),
            "template":         result.get("template"),
            "processed_log":    result.get("processed_log"),
            "extractions":      result.get("extractions", {}),
            "labeled_template": None,
            "llm_decision":     None,
            "llm_reasoning":    None,
            "ocsf":             result.get("ocsf"),
            "ocsf_event":       result.get("ocsf_event"),
        })

    with _results_lock:
        _results_store[session_id] = results

    # Batch LLM review (synchronous — happens before response)
    batch_result = pipeline.batch_review(results)

    pipeline.save()

    # Build upload report
    ocsf_classes  = Counter()
    change_types  = Counter()
    unmatched_map: dict = {}
    for r in results:
        change_types[r["change_type"]] += 1
        if r.get("ocsf") and r["ocsf"].get("ocsf_class_name"):
            ocsf_classes[r["ocsf"]["ocsf_class_name"]] += 1
        if not r.get("ocsf"):
            cid = r.get("cluster_id")
            if cid:
                if cid not in unmatched_map:
                    unmatched_map[cid] = {"cluster_id": cid, "template": r.get("template", ""), "count": 0}
                unmatched_map[cid]["count"] += 1

    unique_clusters = len({r["cluster_id"] for r in results if r.get("cluster_id")})
    report = {
        "filename":          file.filename,
        "total_logs":        len(results),
        "unique_templates":  unique_clusters,
        "compression_pct":   round((1 - unique_clusters / max(len(results), 1)) * 100, 1),
        "change_types":      dict(change_types),
        "ocsf_matched":      sum(1 for r in results if r.get("ocsf")),
        "ocsf_unmatched":    sum(1 for r in results if not r.get("ocsf")),
        "ocsf_breakdown":    dict(ocsf_classes),
        "unmatched_templates": sorted(unmatched_map.values(), key=lambda x: x["count"], reverse=True)[:5],
    }

    return jsonify({
        "session_id": session_id,
        "total":      len(results),
        "results":    results,
        "report":     report,
        "batch_id":   batch_result.get("batch_id", ""),
        "llm_queued": batch_result.get("queued", 0),
    })


@app.route("/templates/reevaluate", methods=["POST"])
def reevaluate_templates():
    queued = pipeline.reevaluate_all()
    return jsonify({"queued": queued})


@app.route("/decisions", methods=["GET"])
def get_decisions():
    batches = approver.get_all_batches() if hasattr(approver, "get_all_batches") else []
    return jsonify({"batches": batches, "decisions": []})


@app.route("/decisions/batch/<batch_id>/approve", methods=["POST"])
def approve_batch(batch_id):
    body    = request.get_json() or {}
    reparse = body.get("reparse", False)
    result  = pipeline.execute_batch(batch_id, reparse=reparse)
    response = {"ok": True, "executed": result["executed"]}
    if result.get("reparse"):
        rp = result["reparse"]
        response["reparse"] = {
            "batch_id":      rp.get("batch_id", ""),
            "queued":        rp.get("queued", 0),
            "reparse_stats": rp.get("reparse_stats", {}),
        }
    return jsonify(response)


@app.route("/decisions/batch/<batch_id>/update", methods=["POST"])
def update_batch_decision(batch_id):
    body       = request.get_json() or {}
    cluster_id = body.get("cluster_id")
    updated    = body.get("decision", {})
    ok = approver.update_decision(batch_id, cluster_id, updated)
    return jsonify({"ok": ok})


@app.route("/decisions/<decision_id>/respond", methods=["POST"])
def respond_decision(decision_id):
    body = request.get_json() or {}
    action = body.get("action")
    edited = body.get("edited_decision")

    if action not in ("accept", "reject", "edit"):
        return jsonify({"error": "invalid action"}), 400

    ok = approver.respond(decision_id, action, edited)
    if not ok:
        return jsonify({"error": "decision not found or timed out"}), 404

    return jsonify({"ok": True})


@app.route("/templates", methods=["GET"])
def get_templates():
    """Return all active templates with their labeled versions."""
    templates = []
    for t in pipeline.store.all_active():
        templates.append({
            "cluster_id":         t.cluster_id,
            "pattern":            t.pattern,
            "labeled_template":   t.labeled_template,
            "llm_decision":       t.llm_decision,
            "llm_reasoning":      t.llm_reasoning,
            "versions":           len(t.versions),
            "status":             t.status.value,
            "created_at":         t.created_at,
        })
    return jsonify({"templates": templates})


@app.route("/templates/<cluster_id>/history", methods=["GET"])
def get_history(cluster_id):
    t = pipeline.store.get(cluster_id)
    if not t:
        return jsonify({"error": "not found"}), 404
    return jsonify({"cluster_id": cluster_id, "versions": t.versions})


@app.route("/templates/<cluster_id>/samples", methods=["GET"])
def get_samples(cluster_id):
    """Return up to 5 sample logs for a cluster from the reservoir."""
    samples = pipeline.sampler.get(cluster_id)[:5]
    return jsonify({"cluster_id": cluster_id, "samples": samples})


@app.route("/events", methods=["GET"])
def get_events():
    """Return the last N parsed log events from _parsed_logs."""
    n = min(int(request.args.get("n", 30)), 100)
    if not hasattr(pipeline, "_parsed_logs") or not pipeline._parsed_logs:
        return jsonify({"events": []})
    logs = list(pipeline._parsed_logs)[-n:]
    # Format to match what the OCSF Events panel expects
    events = []
    for l in logs:
        events.append({
            "class_name":    l.get("ocsf_class", ""),
            "activity_name": l.get("activity", ""),
            "severity":      l.get("severity", ""),
            "status":        l.get("status", ""),
            "message":       l.get("raw_log", ""),
            "user":          {"name": l.get("username", "")} if l.get("username") else None,
            "src_endpoint":  {"ip": l.get("src_ip", "")} if l.get("src_ip") else None,
            "http_request":  {"http_method": l.get("http_method", ""), "url": {"path": l.get("http_path", "")}} if l.get("http_method") else None,
            "http_response": {"code": l.get("http_status", "")} if l.get("http_status") else None,
        })
    return jsonify({"events": events})


@app.route("/parsed-logs", methods=["GET"])
def get_parsed_logs():
    """Return the last 100 parsed logs with extracted fields."""
    n = min(int(request.args.get("n", 100)), 100)
    if not hasattr(pipeline, "_parsed_logs") or not pipeline._parsed_logs:
        return jsonify({"logs": []})
    return jsonify({"logs": list(pipeline._parsed_logs)[-n:]})


@app.route("/stats", methods=["GET"])
def get_stats():
    snap = metrics.snapshot()
    snap.pop("timestamp", None)
    return jsonify({
        "pipeline": pipeline.stats(),
        "metrics": snap,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
