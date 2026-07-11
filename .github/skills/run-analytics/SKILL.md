# SKILL: Run the Analytics Pipeline

Start the real-time capture + inference loop on the built-in laptop webcam
(and screen capture for content segmentation, once sprint 4 lands).

## Prerequisites

- Python 3.12 venv activated
- Dependencies installed: `pip install -e .` (or `pip install -r requirements.txt`)
- Models downloaded: `python scripts/download_models.py`
- Webcam not held by another app (close Teams / Zoom / OBS)

## Run — dev mode (window with debug overlay)

```powershell
python -m gaze_analytics --device AUTO --preview
```

- Opens a window showing bounding boxes, head-pose arrows, attention badge.
- Frames are still discarded after processing — the preview is drawn on a
  transient copy that is dropped after `imshow`.
- Press `q` to quit.

## Run — headless (production style)

```powershell
python -m gaze_analytics --device AUTO --headless
```

- No preview window.
- Writes aggregate metrics to `data/metrics.sqlite` every 5 seconds.
- Use with the dashboard in a separate process.

## Choose the inference device explicitly

- `--device CPU`   — safest, works everywhere
- `--device GPU`   — Iris Xe iGPU (recommended for this laptop)
- `--device AUTO`  — OpenVINO picks best available; `GPU,CPU` preferred
- `--device NPU`   — only on Core Ultra (Meteor / Lunar Lake)

## Troubleshooting

- **"Camera index 0 failed to open"** → try `--camera 1` or check Windows
  camera privacy setting (Settings → Privacy → Camera → Desktop apps).
- **Low FPS on GPU** → confirm Intel Graphics driver is recent, and that
  OpenVINO detects the GPU: `python -c "from openvino.runtime import Core; print(Core().available_devices)"`.
- **Model not found** → re-run `python scripts/download_models.py`.
