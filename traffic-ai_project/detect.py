"""
detect.py — YOLOv8 vehicle detection, lane assignment, and AI signal logic.

Pipeline:
  1. Load YOLOv8 model (CPU optimised)
  2. Per-frame: detect vehicles → assign lanes → decide signals
  3. Draw overlays (bboxes, lane dividers, signal status, timers)
  4. Write annotated frames to output video
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── Try importing ultralytics YOLO ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠  ultralytics not found — using mock detector")


# ── Constants ──────────────────────────────────────────────────────────────────

# COCO class IDs used by YOLOv8
VEHICLE_CLASSES = {
    2:  "Car",
    3:  "Motorcycle",
    5:  "Bus",
    7:  "Truck",
    # Custom IDs (if fine-tuned model used)
    80: "Ambulance",
    81: "Fire Brigade",
}

EMERGENCY_LABELS = {"ambulance", "fire brigade", "fire truck"}

# Signal colours (BGR)
COLOR_GREEN     = (0,   220,  80)
COLOR_RED       = (0,   40,  220)
COLOR_YELLOW    = (0,   210, 255)
COLOR_WHITE     = (255, 255, 255)
COLOR_BLACK     = (0,   0,   0)
COLOR_EMERGENCY = (0,   0,   255)   # pure red for emergency bbox

# Font
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Lane config: fraction of frame width boundaries
#  Lane 1: 0–0.25,  Lane 2: 0.25–0.50,  Lane 3: 0.50–0.75,  Lane 4: 0.75–1.0
NUM_LANES   = 4
LANE_BOUNDS = [i / NUM_LANES for i in range(NUM_LANES + 1)]   # [0, 0.25, 0.5, 0.75, 1.0]

# Signal timing
MIN_GREEN_SEC = 10
MAX_GREEN_SEC = 60


# ── AI Signal Decision ─────────────────────────────────────────────────────────

def decide_signal(
    vehicle_counts: Dict[int, int],
    emergency_lane: Optional[int],
) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    Core AI decision function.

    Args:
        vehicle_counts : {lane_id: count}
        emergency_lane : lane containing emergency vehicle, or None

    Returns:
        signals  : {lane_id: "GREEN" | "RED" | "YELLOW"}
        timings  : {lane_id: seconds}
    """
    num_lanes = len(vehicle_counts)
    signals  = {i: "RED"   for i in range(num_lanes)}
    timings  = {i: 0       for i in range(num_lanes)}

    if emergency_lane is not None:
        # Emergency pre-emption: force corridor lane green
        signals[emergency_lane] = "GREEN"
        timings[emergency_lane] = MAX_GREEN_SEC
        return signals, timings

    if not vehicle_counts or all(v == 0 for v in vehicle_counts.values()):
        # No vehicles — default cycle
        signals[0] = "GREEN"
        timings[0] = MIN_GREEN_SEC
        return signals, timings

    # Weighted scoring: weight = count * lane_priority (all equal here)
    priority = {i: 1.0 for i in range(num_lanes)}
    scores   = {i: vehicle_counts.get(i, 0) * priority[i] for i in range(num_lanes)}

    green_lane = max(scores, key=scores.get)
    signals[green_lane] = "GREEN"

    max_count = max(vehicle_counts.values()) if vehicle_counts else 1
    max_count = max(max_count, 1)
    ratio     = vehicle_counts.get(green_lane, 0) / max_count
    green_sec = int(MIN_GREEN_SEC + ratio * (MAX_GREEN_SEC - MIN_GREEN_SEC))
    timings[green_lane] = green_sec

    return signals, timings


# ── Drawing Helpers ────────────────────────────────────────────────────────────

def draw_lane_dividers(frame: np.ndarray) -> np.ndarray:
    """Draw vertical lane boundary lines on frame."""
    h, w = frame.shape[:2]
    for i in range(1, NUM_LANES):
        x = int(w * LANE_BOUNDS[i])
        cv2.line(frame, (x, 0), (x, h), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Lane {i}", (x - 55, 22),
                    FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Lane 1", (6, 22), FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    return frame


def draw_signal_bar(
    frame: np.ndarray,
    signals: Dict[int, str],
    timings: Dict[int, int],
    vehicle_counts: Dict[int, int],
    emergency: bool,
    emergency_lane: Optional[int],
    countdown: int,
) -> np.ndarray:
    """Draw signal status bar at bottom of frame."""
    h, w = frame.shape[:2]
    bar_h = 52
    bar_y = h - bar_h

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    lane_w = w // NUM_LANES
    for lane_id in range(NUM_LANES):
        sig   = signals.get(lane_id, "RED")
        count = vehicle_counts.get(lane_id, 0)
        timer = timings.get(lane_id, 0)
        x0    = lane_id * lane_w
        xc    = x0 + lane_w // 2

        # Signal dot
        dot_col = (COLOR_GREEN if sig == "GREEN"
                   else COLOR_YELLOW if sig == "YELLOW"
                   else COLOR_RED)
        cv2.circle(frame, (xc - 44, bar_y + 26), 10, dot_col, -1, cv2.LINE_AA)

        # Lane label
        label = f"L{lane_id+1} {sig}"
        if sig == "GREEN":
            label += f"  {countdown}s"
        cv2.putText(frame, label, (xc - 30, bar_y + 20), FONT, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

        # Vehicle count
        cv2.putText(frame, f"{count} veh", (xc - 30, bar_y + 38), FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # Separator
        if lane_id < NUM_LANES - 1:
            cv2.line(frame, (x0 + lane_w, bar_y), (x0 + lane_w, h), (50, 50, 60), 1)

    # Emergency banner
    if emergency:
        banner = "  🚨 EMERGENCY VEHICLE DETECTED — GREEN CORRIDOR ACTIVE  "
        (tw, th), _ = cv2.getTextSize(banner, FONT, 0.5, 1)
        cv2.rectangle(frame, (0, bar_y - 28), (w, bar_y - 2), (0, 0, 180), -1)
        cv2.putText(frame, banner, ((w - tw) // 2, bar_y - 8),
                    FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: List[dict],
) -> np.ndarray:
    """Draw bounding boxes and labels for each detection."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label         = det["label"]
        conf          = det["confidence"]
        is_emergency  = det["is_emergency"]

        color     = COLOR_EMERGENCY if is_emergency else COLOR_GREEN
        thickness = 3 if is_emergency else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Label background
        text  = f"{'⚠ ' if is_emergency else ''}{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    FONT, 0.5, COLOR_WHITE if is_emergency else COLOR_BLACK, 1, cv2.LINE_AA)
    return frame


def draw_timestamp(frame: np.ndarray, frame_num: int, fps: float) -> np.ndarray:
    """Draw frame number and elapsed time top-right."""
    elapsed = frame_num / max(fps, 1)
    ts = f"Frame {frame_num}  |  {elapsed:.1f}s"
    cv2.putText(frame, ts, (frame.shape[1] - 230, 20),
                FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ── Mock YOLO (fallback) ───────────────────────────────────────────────────────

class MockDetector:
    """Produces randomised detections so the app works without a YOLO model."""
    import random as _rand

    def detect(self, frame: np.ndarray) -> List[dict]:
        import random
        h, w = frame.shape[:2]
        results = []
        for _ in range(random.randint(2, 10)):
            cls_id = random.choice(list(VEHICLE_CLASSES.keys())[:6])
            label  = VEHICLE_CLASSES[cls_id]
            x1 = random.randint(0, w - 80)
            y1 = random.randint(0, h - 80)
            x2 = min(x1 + random.randint(50, 120), w - 1)
            y2 = min(y1 + random.randint(40, 90),  h - 1)
            results.append({
                "bbox":         (x1, y1, x2, y2),
                "label":        label,
                "confidence":   round(random.uniform(0.55, 0.98), 2),
                "is_emergency": label.lower() in EMERGENCY_CLASSES,
                "cx":           (x1 + x2) / 2,
            })
        return results

EMERGENCY_CLASSES = {"ambulance", "fire brigade"}


# ── Main Detector ──────────────────────────────────────────────────────────────

class TrafficDetector:
    """
    Full pipeline:
      load model → open video → per-frame detect → decide signal → annotate → write output
    """

    def __init__(self, model_path: str = "models/yolov8n.pt"):
        self.model        = None
        self.mock         = None
        self.model_path   = model_path
        self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("⚠  YOLO unavailable — mock detector active")
            self.mock = MockDetector()
            return
        path = Path(self.model_path)
        try:
            if path.exists():
                self.model = YOLO(str(path))
            else:
                print(f"  Model not at {path} — downloading yolov8n …")
                self.model = YOLO("yolov8n.pt")   # auto-download
            # Force CPU
            self.model.to("cpu")
            print("✅  YOLOv8 model loaded (CPU mode)")
        except Exception as e:
            print(f"⚠  Could not load YOLO: {e} — mock detector active")
            self.mock = MockDetector()

    def _detect_frame(self, frame: np.ndarray) -> List[dict]:
        """Run detection on a single frame, return list of detection dicts."""
        if self.mock:
            return self.mock.detect(frame)

        h, w     = frame.shape[:2]
        results  = self.model(frame, verbose=False, device="cpu")[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf  = float(box.conf[0])
            if conf < 0.30:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label  = VEHICLE_CLASSES[cls_id]
            is_em  = label.lower() in EMERGENCY_CLASSES
            detections.append({
                "bbox":        (x1, y1, x2, y2),
                "label":       label,
                "confidence":  round(conf, 2),
                "is_emergency": is_em,
                "cx":          (x1 + x2) / 2,
            })
        return detections

    def _assign_lanes(self, detections: List[dict], frame_w: int) -> Dict[int, int]:
        """Count vehicles per lane based on bbox centre-x."""
        counts = {i: 0 for i in range(NUM_LANES)}
        for det in detections:
            cx_norm = det["cx"] / frame_w
            for lane_id in range(NUM_LANES):
                if LANE_BOUNDS[lane_id] <= cx_norm < LANE_BOUNDS[lane_id + 1]:
                    counts[lane_id] += 1
                    det["lane"] = lane_id
                    break
        return counts

    def _find_emergency_lane(self, detections: List[dict]) -> Optional[int]:
        for det in detections:
            if det["is_emergency"]:
                return det.get("lane")
        return None

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_cb: Optional[Callable] = None,
    ) -> Dict[int, int]:
        """
        Process entire video. Returns final vehicle counts per lane.
        Calls progress_cb(frame_num, total, lane_counts, signals, emergency, em_lane)
        on each frame.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        out     = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        lane_counts    = {i: 0 for i in range(NUM_LANES)}
        signals        = {i: "RED" for i in range(NUM_LANES)}
        timings        = {i: 0    for i in range(NUM_LANES)}
        emergency      = False
        emergency_lane = None
        countdown      = 0
        green_start    = 0
        frame_num      = 0

        # Process every Nth frame for speed (skip=2 → ~2× faster)
        SKIP = max(1, int(fps // 12))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % SKIP == 0:
                detections     = self._detect_frame(frame)
                lane_counts    = self._assign_lanes(detections, w)
                emergency_lane = self._find_emergency_lane(detections)
                emergency      = emergency_lane is not None
                signals, timings = decide_signal(lane_counts, emergency_lane)

                # Countdown logic
                green_lane = next((l for l, s in signals.items() if s == "GREEN"), None)
                if green_lane is not None:
                    elapsed   = (frame_num - green_start) / fps
                    countdown = max(0, timings.get(green_lane, 0) - int(elapsed))
                    if countdown == 0:
                        green_start = frame_num

                # Draw overlays
                frame = draw_lane_dividers(frame)
                frame = draw_detections(frame, detections)
                frame = draw_signal_bar(frame, signals, timings, lane_counts,
                                        emergency, emergency_lane, countdown)
                frame = draw_timestamp(frame, frame_num, fps)

            out.write(frame)

            if progress_cb and frame_num % 10 == 0:
                progress_cb(frame_num, total, lane_counts, signals,
                            emergency, emergency_lane)

        cap.release()
        out.release()

        if progress_cb:
            progress_cb(total, total, lane_counts, signals, emergency, emergency_lane)

        return lane_counts
