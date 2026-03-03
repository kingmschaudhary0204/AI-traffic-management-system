"""
app.py — Flask entry point for Dynamic AI Traffic Flow Optimizer
Handles video upload, processing pipeline, simulation mode, and serving results.
"""

import os
import json
import time
import threading
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, Response, stream_with_context
)
from werkzeug.utils import secure_filename

from detect import TrafficDetector
from simulate import TrafficSimulator
from graph import generate_density_graph

# ── App Config ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER   = "uploads",
    OUTPUT_FOLDER   = "static/output",
    GRAPH_FOLDER    = "static/graphs",
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024,   # 200 MB max upload
    SECRET_KEY      = "nexus-traffic-ai-2024",
)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Shared processing state (thread-safe via lock)
_state_lock = threading.Lock()
processing_state = {
    "status":       "idle",      # idle | processing | done | error
    "progress":     0,
    "total_frames": 0,
    "lane_counts":  {},
    "signals":      {},
    "emergency":    False,
    "emergency_lane": None,
    "message":      "",
    "graph_path":   "",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def update_state(**kwargs):
    with _state_lock:
        processing_state.update(kwargs)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    """Receive uploaded video, kick off background processing."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    input_path  = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], "output.mp4")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
    file.save(input_path)

    # Reset state and start background thread
    update_state(status="processing", progress=0, message="Starting detection…",
                 emergency=False, emergency_lane=None, lane_counts={}, signals={})

    thread = threading.Thread(
        target=_run_detection,
        args=(input_path, output_path),
        daemon=True,
    )
    thread.start()

    return jsonify({"message": "Processing started", "filename": filename})


@app.route("/simulate", methods=["POST"])
def simulate():
    """Run simulation mode — no video needed."""
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], "output.mp4")
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    update_state(status="processing", progress=0, message="Running simulation…",
                 emergency=False, emergency_lane=None, lane_counts={}, signals={})

    thread = threading.Thread(
        target=_run_simulation,
        args=(output_path,),
        daemon=True,
    )
    thread.start()
    return jsonify({"message": "Simulation started"})


@app.route("/status")
def status():
    """Poll endpoint — returns current processing state."""
    with _state_lock:
        return jsonify(dict(processing_state))


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/static/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.route("/static/graphs/<path:filename>")
def serve_graph(filename):
    return send_from_directory(app.config["GRAPH_FOLDER"], filename)


# ── Background workers ─────────────────────────────────────────────────────────

def _run_detection(input_path: str, output_path: str):
    """Run YOLO detection pipeline in background thread."""
    try:
        detector = TrafficDetector()

        def progress_cb(frame_num, total, lane_counts, signals, emergency, emergency_lane):
            pct = int((frame_num / max(total, 1)) * 100)
            update_state(
                progress=pct,
                total_frames=total,
                lane_counts=lane_counts,
                signals=signals,
                emergency=emergency,
                emergency_lane=emergency_lane,
                message=f"Processing frame {frame_num}/{total}",
            )

        final_counts = detector.process_video(input_path, output_path, progress_cb)

        # Generate density graph
        graph_path = generate_density_graph(
            final_counts,
            os.path.join(app.config["GRAPH_FOLDER"], "density.png")
        )

        update_state(
            status="done",
            progress=100,
            lane_counts=final_counts,
            graph_path="static/graphs/density.png",
            message="Processing complete!",
        )
    except Exception as e:
        update_state(status="error", message=str(e))


def _run_simulation(output_path: str):
    """Run simulation mode in background thread."""
    try:
        sim = TrafficSimulator()

        def progress_cb(frame_num, total, lane_counts, signals, emergency, emergency_lane):
            pct = int((frame_num / max(total, 1)) * 100)
            update_state(
                progress=pct,
                total_frames=total,
                lane_counts=lane_counts,
                signals=signals,
                emergency=emergency,
                emergency_lane=emergency_lane,
                message=f"Simulating frame {frame_num}/{total}",
            )

        final_counts = sim.run(output_path, progress_cb)

        graph_path = generate_density_graph(
            final_counts,
            os.path.join(app.config["GRAPH_FOLDER"], "density.png")
        )

        update_state(
            status="done",
            progress=100,
            lane_counts=final_counts,
            graph_path="static/graphs/density.png",
            message="Simulation complete!",
        )
    except Exception as e:
        update_state(status="error", message=str(e))


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/output", exist_ok=True)
    os.makedirs("static/graphs", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
