# SKILL: Download OpenVINO Models

Fetch the pre-trained models from Open Model Zoo into `models/`.

## Prerequisites

- `openvino-dev` installed (comes from `requirements.txt`).
- ~150 MB of disk space.

## Run

```powershell
python scripts/download_models.py
```

## What it downloads

| Model | Precision | Size | Purpose |
|---|---|---|---|
| `person-detection-retail-0013` | INT8 + FP16 | ~3 MB | Count viewers |
| `face-detection-retail-0004`   | INT8 + FP16 | ~2 MB | Locate faces |
| `head-pose-estimation-adas-0001` | FP16 | ~4 MB | Attention gate |
| `gaze-estimation-adas-0002`    | FP16 | ~3 MB | Near-tier gaze |
| `age-gender-recognition-retail-0013` | FP16 | ~9 MB | Aggregate demographics |

## Layout after download

```
models/
├── intel/
│   ├── person-detection-retail-0013/
│   │   ├── FP16-INT8/*.xml *.bin
│   │   └── FP16/*.xml *.bin
│   ├── face-detection-retail-0004/...
│   ├── head-pose-estimation-adas-0001/...
│   ├── gaze-estimation-adas-0002/...
│   └── age-gender-recognition-retail-0013/...
```

The `models/` directory is gitignored — every developer runs the script once.

## Behind the scenes

The script uses `omz_downloader` from `openvino-dev`. Model URLs are pinned
by Open Model Zoo; if a download fails, you can fetch manually from
https://github.com/openvinotoolkit/open_model_zoo.
