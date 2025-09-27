# Dynamic ROI Vision Demo (YOLOv8)

This interactive desktop demo showcases four adaptive ROI modes with YOLOv8 (using your local yolov8n.pt):

1. Full fixed frame as ROI
2. Fixed frame with a fixed sub-ROI
3. Fixed frame with a movable ROI
4. Movable frame with a movable ROI

Features:
- Desktop UI (PySide6)
- Source: default webcam or sample video file
- Humans-only detection (COCO class person)
- CUDA preferred when available

## Setup

1) Create/activate a Python environment (recommended)

`powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
`

2) Install dependencies

`powershell
pip install -r requirements.txt
`

3) Place yolov8n.pt in the project root (already present per your note)

4) Run demo

`powershell
python run_demo.py
`

If you need a specific CUDA-enabled Torch build, follow the official PyTorch install instructions: https://pytorch.org/get-started/locally/

## Controls (initial)
- Mode selector (1–4)
- Source selector: Webcam (index) or Video File (browse)
- Start/Stop
- Confidence/IoU sliders

Further commits will add full ROI logic, tracking, overlays, and metrics.
