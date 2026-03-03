"""
simulate.py — Synthetic traffic simulation (no real video needed).

Draws animated vehicles on a road background, applies the same
detection/signal logic as the real pipeline, and writes output.mp4.
"""

import cv2
import numpy as np
import random
import math
from typing import Callable, Dict, List, Optional, Tuple

from detect import (
    NUM_LANES, LANE_BOUNDS, decide_signal,
    draw_lane_dividers, draw_signal_bar, draw_timestamp,
    COLOR_GREEN, COLOR_EMERGENCY, COLOR_WHITE, FONT,
)

# ── Sim constants ──────────────────────────────────────────────────────────────
SIM_WIDTH   = 960
SIM_HEIGHT  = 540
SIM_FPS     = 20
SIM_FRAMES  = 300   # ~15 seconds

VEHICLE_TYPES = [
    {"label": "Car",          "color": (200, 200, 255), "w": 42, "h": 24, "is_emergency": False},
    {"label": "Bus",          "color": (255, 230, 100), "w": 70, "h": 28, "is_emergency": False},
    {"label": "Truck",        "color": (180, 255, 200), "w": 65, "h": 26, "is_emergency": False},
    {"label": "Motorcycle",   "color": (220, 190, 255), "w": 24, "h": 16, "is_emergency": False},
    {"label": "Ambulance",    "color": (60,  60,  255), "w": 50, "h": 26, "is_emergency": True },
    {"label": "Fire Brigade", "color": (30,  30,  220), "w": 65, "h": 28, "is_emergency": True },
]


class SimVehicle:
    """A simulated vehicle moving down a lane."""

    def __init__(self, lane_id: int, frame_w: int, frame_h: int):
        self.lane_id    = lane_id
        vtype           = random.choice(VEHICLE_TYPES)
        self.label      = vtype["label"]
        self.color      = vtype["color"]
        self.vw         = vtype["w"]
        self.vh         = vtype["h"]
        self.is_emergency = vtype["is_emergency"]

        # Lane centre x
        lb       = LANE_BOUNDS
        lane_cx  = int(frame_w * (lb[lane_id] + lb[lane_id + 1]) / 2)
        self.x   = lane_cx
        self.y   = random.randint(-frame_h, 0)   # start above frame
        self.speed = random.uniform(2.5, 5.5)
        self.frame_h = frame_h

    def update(self, signal: str):
        """Move vehicle; stop if signal is RED."""
        if signal == "GREEN" or self.is_emergency:
            self.y += self.speed
        elif signal == "YELLOW":
            self.y += self.speed * 0.4

    @property
    def alive(self) -> bool:
        return self.y < self.frame_h + self.vh

    def draw(self, frame: np.ndarray):
        x1 = self.x - self.vw // 2
        y1 = int(self.y) - self.vh // 2
        x2 = x1 + self.vw
        y2 = y1 + self.vh

        color = (0, 0, 220) if self.is_emergency else self.color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 0, 180) if self.is_emergency else (100, 100, 100), 2)

        # Label
        prefix = "EMRG " if self.is_emergency else ""
        cv2.putText(frame, prefix + self.label, (x1 + 2, y1 + 14),
                    FONT, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # Emergency siren flash
        if self.is_emergency and (int(self.y) // 6) % 2 == 0:
            cv2.rectangle(frame, (x1, y1 - 6), (x2, y1), (0, 120, 255), -1)


def _road_background(w: int, h: int) -> np.ndarray:
    """Draw a simple road background."""
    bg = np.full((h, w, 3), (50, 55, 50), dtype=np.uint8)
    # Road surface
    cv2.rectangle(bg, (0, 60), (w, h - 60), (80, 80, 80), -1)
    # Kerb lines
    cv2.line(bg, (0, 60),      (w, 60),      (200, 200, 200), 3)
    cv2.line(bg, (0, h - 60),  (w, h - 60),  (200, 200, 200), 3)
    return bg


class TrafficSimulator:
    """Generates a synthetic traffic video with signal control."""

    def run(
        self,
        output_path: str,
        progress_cb: Optional[Callable] = None,
    ) -> Dict[int, int]:

        w, h    = SIM_WIDTH, SIM_HEIGHT
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        out     = cv2.VideoWriter(output_path, fourcc, SIM_FPS, (w, h))

        background  = _road_background(w, h)
        vehicles: List[SimVehicle] = []

        lane_counts    = {i: 0 for i in range(NUM_LANES)}
        signals        = {i: "RED" for i in range(NUM_LANES)}
        timings        = {i: 0    for i in range(NUM_LANES)}
        emergency      = False
        emergency_lane = None
        countdown      = 0
        green_start    = 0
        spawn_timer    = 0

        # Inject an emergency vehicle mid-simulation
        EMERGENCY_SPAWN_FRAME = SIM_FRAMES // 2

        for frame_num in range(1, SIM_FRAMES + 1):
            frame = background.copy()

            # Spawn regular vehicles
            spawn_timer += 1
            if spawn_timer >= 8:
                spawn_timer = 0
                lane = random.randint(0, NUM_LANES - 1)
                vehicles.append(SimVehicle(lane, w, h))

            # Spawn emergency vehicle at midpoint
            if frame_num == EMERGENCY_SPAWN_FRAME:
                em_lane = random.randint(0, NUM_LANES - 1)
                ev = SimVehicle(em_lane, w, h)
                ev.label        = "Ambulance"
                ev.is_emergency = True
                ev.color        = (60, 60, 255)
                ev.speed        = 4.0
                vehicles.append(ev)

            # Count vehicles per lane & find emergency
            lane_counts    = {i: 0 for i in range(NUM_LANES)}
            emergency_lane = None
            for v in vehicles:
                lane_counts[v.lane_id] = lane_counts.get(v.lane_id, 0) + 1
                if v.is_emergency:
                    emergency_lane = v.lane_id

            emergency = emergency_lane is not None
            signals, timings = decide_signal(lane_counts, emergency_lane)

            # Update & draw vehicles
            for v in vehicles:
                v.update(signals.get(v.lane_id, "RED"))
                v.draw(frame)

            # Remove off-screen vehicles
            vehicles = [v for v in vehicles if v.alive]

            # Countdown
            green_lane = next((l for l, s in signals.items() if s == "GREEN"), None)
            if green_lane is not None:
                elapsed   = (frame_num - green_start) / SIM_FPS
                countdown = max(0, timings.get(green_lane, 0) - int(elapsed))
                if countdown == 0:
                    green_start = frame_num

            # Draw overlays
            frame = draw_lane_dividers(frame)
            frame = draw_signal_bar(frame, signals, timings, lane_counts,
                                    emergency, emergency_lane, countdown)
            frame = draw_timestamp(frame, frame_num, SIM_FPS)

            out.write(frame)

            if progress_cb and frame_num % 10 == 0:
                progress_cb(frame_num, SIM_FRAMES, lane_counts, signals,
                            emergency, emergency_lane)

        out.release()

        if progress_cb:
            progress_cb(SIM_FRAMES, SIM_FRAMES, lane_counts, signals,
                        emergency, emergency_lane)

        return lane_counts
