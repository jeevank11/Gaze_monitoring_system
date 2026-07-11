---
applyTo: "**"
---

# Gaze Analytics — Global Copilot Instructions

You are helping build **privacy-preserving gaze analytics for digital signage**
on low-power Intel hardware (CPU + iGPU, no NPU on this laptop).

## Hard rules — never violate

1. **Frames are ephemeral.** No `cv2.imwrite`, `PIL.Image.save`, video writer,
   `numpy.save`, `pickle.dump`, or any other call that persists a raw camera
   frame or a raw screen capture. Grep for these before every commit.
2. **No face recognition, no embeddings, no ReID.** The tracker uses bbox
   geometry + Kalman motion only. IDs are `int` counters that die with the
   process.
3. **Only aggregates leave RAM.** SQLite stores counters, ratios, timestamps,
   pHash fingerprints (64-bit ints), and optional 160×90 content thumbnails.
   Never faces, never full frames, never per-person records.
4. **Everything runs on-device.** No cloud inference. No telemetry. No
   third-party API calls at runtime.
5. **Real-time budget: ≥15 FPS end-to-end at 720p** on Intel Core i7-1185G7
   (Iris Xe iGPU). Prefer OpenVINO `AUTO` device with `GPU` preferred.

## Coding style

- Python 3.12, type-hinted, `ruff`-clean, `pytest`-covered.
- Package layout is `src/gaze_analytics/...` — always use relative imports
  inside the package.
- Config lives in `gaze_analytics.config` (pydantic-settings, env-overridable).
- Use `rich` for CLI output, `typer` for the CLI entry point.
- Log with the stdlib `logging` module; never `print` in library code.
- Small pure functions; side-effects at the edges (capture, sink, UI).

## Model choices (Open Model Zoo)

- Person detection: `person-detection-retail-0013` INT8
- Face detection: `face-detection-retail-0004` INT8
- Head pose: `head-pose-estimation-adas-0001` FP16
- Gaze (near tier only): `gaze-estimation-adas-0002` FP16
- Age/gender (aggregate only): `age-gender-recognition-retail-0013` FP16

Downloaded via `scripts/download_models.py` into `models/` (gitignored).

## Testing

- Unit tests for pure logic (`phash`, `tracker`, `aggregator`).
- A privacy smoke test that greps the codebase for banned APIs.
- No test may open the real webcam — use fixture frames from `tests/fixtures/`.

## When suggesting code

- Prefer editing existing modules over creating new ones.
- Never introduce a dependency that isn't already in `pyproject.toml` without
  telling the user first.
- Never suggest adding a face-recognition, ReID, embedding, or cloud-inference
  library. If asked, refuse and explain the privacy constraint.
