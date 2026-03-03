# 🚦 NEXUS Traffic AI — Dynamic Flow Optimizer
### YOLOv8 · PyTorch · OpenCV · Flask · Emergency Green Corridor

---

## What It Does
- **Upload** a traffic video → YOLOv8 detects vehicles frame-by-frame
- **4 lanes** defined by ROI dividers; vehicles counted per lane
- **AI Signal Logic** (Webster's weighted scoring) picks the green lane dynamically
- **Emergency vehicles** (Ambulance / Fire Brigade) trigger instant green corridor
- **Simulation mode** — no video needed; synthetic animated traffic with emergency injection
- **Density graph** auto-generated with matplotlib

---

## Quick Start (Windows / Mac / Linux)

### Step 1 — Extract & Open

```
Unzip nexus-traffic-ai.zip
Open the traffic-ai/ folder in VS Code
```

### Step 2 — Create Virtual Environment

```bash
# In VS Code terminal (Ctrl + `)
cd traffic-ai

python -m venv venv

# Activate:
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install flask opencv-python numpy matplotlib Pillow werkzeug
```

**For YOLO (optional — system works without it):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

### Step 4 — Run

```bash
python app.py
```

Open browser: **http://localhost:5000**

---

## Usage

### Option A — Upload Video
1. Click **"Drop video here"** or drag & drop an MP4/AVI/MOV file
2. Click **PROCESS VIDEO**
3. Progress bar tracks detection
4. Result page shows annotated video + density graph

### Option B — Simulation Mode
1. Click **RUN SIMULATION** (no video needed)
2. Synthetic 4-lane road is generated with:
   - Random cars, buses, trucks, motorcycles
   - Ambulance injected at frame 150
   - AI signal decisions every frame
3. Watch the green corridor activate automatically

---

## Project Structure

```
traffic-ai/
│
├── app.py              ← Flask server, routes, background threads
├── detect.py           ← YOLOv8 detection, lane assignment, signal logic
├── simulate.py         ← Synthetic traffic generator (no video needed)
├── graph.py            ← Matplotlib density bar chart
├── train.py            ← YOLOv8 fine-tuning script
├── dataset.yaml        ← Training dataset config
├── requirements.txt
│
├── templates/
│   ├── index.html      ← Upload / Simulate page
│   └── result.html     ← Output video + stats + graph
│
├── static/
│   ├── output/         ← output.mp4 saved here
│   └── graphs/         ← density.png saved here
│
├── uploads/            ← Uploaded videos stored here
├── models/             ← YOLO .pt weights go here
│
└── dataset/
    ├── images/
    │   ├── train/      ← Training images
    │   └── val/        ← Validation images
    └── labels/
        ├── train/      ← YOLO .txt label files
        └── val/
```

---

## AI Signal Decision Logic

```python
def decide_signal(vehicle_counts, emergency_lane):
    # Emergency pre-emption (highest priority)
    if emergency_lane is not None:
        signals[emergency_lane] = "GREEN"   # instant corridor
        return signals, timings

    # Weighted scoring
    scores = {lane: count * priority[lane] for lane, count in vehicle_counts.items()}
    green_lane = max(scores, key=scores.get)

    # Dynamic timer: 10–60 seconds based on density ratio
    ratio     = counts[green_lane] / max_count
    green_sec = int(10 + ratio * 50)
    ...
```

---

## Custom Model Training

### 1. Prepare Dataset
Label images using [Roboflow](https://roboflow.com) or LabelImg.
Export in **YOLO format** to `dataset/images/` and `dataset/labels/`.

### 2. Train
```bash
python train.py --epochs 50 --batch 8 --imgsz 640 --device cpu
```

GPU (if available):
```bash
python train.py --epochs 100 --batch 16 --device 0
```

### 3. Use Trained Model
Copy `models/traffic_v1/weights/best.pt` to `models/yolov8n.pt`
and restart the app.

---

## Environment Details

| Component | Version |
|---|---|
| Python | 3.10+ |
| Flask | 3.0+ |
| PyTorch | 2.0+ (CPU) |
| YOLOv8 | ultralytics 8.2+ |
| OpenCV | 4.9+ |
| NumPy | 1.26+ |
| Matplotlib | 3.8+ |

---

## Troubleshooting

**`No module named cv2`**
```bash
pip install opencv-python
```

**`No module named ultralytics`**
```bash
pip install ultralytics
```
If that fails, the app runs in **mock mode** automatically — all features work except real YOLO detection.

**`python app.py` — port already in use**
```bash
python app.py   # edit app.py last line, change port=5000 to port=5001
```

**Video plays but no bounding boxes**
- YOLO model wasn't loaded → mock detector is active (random boxes shown)
- Install ultralytics + download weights to resolve

---

## Output Files

| File | Location | Description |
|---|---|---|
| `output.mp4` | `static/output/` | Annotated video with boxes + signal overlay |
| `density.png` | `static/graphs/` | Vehicle density bar chart per lane |
