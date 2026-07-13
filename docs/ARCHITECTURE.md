# Architecture

## Two capture sources, one pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                            One Python process                        │
│                                                                      │
│   ┌────────────┐        ┌────────────────────────────────────────┐   │
│   │  Webcam    │──▶──▶──│  Person detect (OpenVINO INT8, iGPU)   │   │
│   │  (~15 FPS) │        │  Face detect on person crops           │   │
│   │  BGR frame │        │  Head pose + Gaze (near tier)          │   │
│   │  discarded │        │  Age/Gender (attended faces only)      │   │
│   └────────────┘        └───────┬────────────────────────────────┘   │
│                                 ▼                                    │
│                       ┌───────────────────────┐                      │
│                       │  IoU + Kalman tracker │                      │
│                       │  (no embeddings)      │                      │
│                       └───────┬───────────────┘                      │
│                               ▼                                      │
│   ┌────────────┐        ┌───────────────────────┐                    │
│   │  Screen    │──pHash─▶  Content segmenter    │                    │
│   │  (~1 Hz)   │        │  (scene-change detect)│                    │
│   │  discarded │        └───────┬───────────────┘                    │
│   └────────────┘                ▼                                    │
│                       ┌───────────────────────┐                      │
│                       │  Aggregator (5 s win) │                      │
│                       │  viewers/attending/   │                      │
│                       │  dwell/gender/segment │                      │
│                       └───────┬───────────────┘                      │
│                               ▼                                      │
│                       ┌───────────────────────┐                      │
│                       │  SQLite  (aggregates  │                      │
│                       │  only, no frames)     │                      │
│                       └───────┬───────────────┘                      │
│                               ▼                                      │
│                       ┌───────────────────────┐                      │
│                       │  Streamlit dashboard  │                      │
│                       │  (separate process)   │                      │
│                       └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Attention decision

For each detected face per frame:

1. **Presence** → contributes to `viewers` count.
2. **Attention gate** → `|yaw| < 25°` and `|pitch| < 20°` → contributes to `attending`.
3. **Distance tier** (from face bbox height in pixels):
   - **Near** (bbox > 120 px tall, roughly < 1.5 m): also run gaze model,
     require gaze vector within `±15°` of screen normal.
   - **Far** (bbox < 60 px tall, roughly > 2.5 m): head-pose alone decides.
4. **Dwell time** → tracker assigns integer ID (RAM only, dies on process
   exit). Dwell = consecutive attending frames × frame period.
5. **Gender** → aggregate counter only. Never per-track.

## Content segmentation (no config required)

Every second, grab a screenshot of the primary monitor with `mss`, compute
a 64-bit perceptual hash. Compare Hamming distance vs. last hash:

- Distance < threshold (12 bits) → same content, continue current segment.
- Distance ≥ threshold **and** new hash stable for ≥3 seconds → new segment.
- New segment gets:
  - Auto-generated ID (`auto_YYYY-MM-DD_HH-MM-SS`)
  - Optional 160×90 thumbnail (privacy-safe: it's *your* screen, not a face)
  - Type tag: `ad`, `content`, or `unknown` (see YouTube badge detector)

## Data model (aggregates only)

```sql
metrics (id, ts, window_seconds, viewers, attending, avg_dwell_ms,
         male_count, female_count, segment_id, segment_type)
segments (id, started_at, ended_at, duration_ms, phash, thumbnail_b64,
          segment_type)
```

No per-person, no bbox, no image, no identity. Adding any of these fields
must be rejected in review.

## Sprint map

| # | Deliverable | Files touched | Exit criteria |
|---|---|---|---|
| 1 | Bootstrap + webcam capture | `capture/webcam.py`, `main.py`, `config.py` | `python -m gaze_analytics --preview` shows live feed at ≥15 FPS; `pytest -q` green |
| 2 | Face detection + overlay | `inference/face_detector.py`, `inference/openvino_runner.py` | Bounding boxes drawn; still ≥15 FPS |
| 3 | Head pose + attention gate | `inference/head_pose.py`, `engagement/attention.py` | Box turns green when looking at screen |
| 4 | Tracker + content segmenter | `tracker/iou_kalman.py`, `content/phash.py`, `content/segmenter.py`, `capture/screen.py` | Each face has a stable ID; screen changes create new segments |
| 5 | Age/gender + SQLite sink | `inference/age_gender.py`, `engagement/aggregator.py`, `storage/sqlite_sink.py` | Rows appearing every 5 s |
| 6 | Streamlit dashboard | `dashboard/📺_Gaze_Live.py` | Live tiles + trend chart + segments table |
| 7 | Privacy render + polish | `privacy/silhouette_render.py`, README, threat model | Silhouette-only demo mode works |
| 8 | Benchmark + Docker + exe + pitch | `scripts/benchmark.py`, `deploy/*`, `docs/PITCH.md`, `docs/BENCHMARKS.md` | Reproducible benchmark table; single-command deploy for each target |

## OpenVINO device selection

- Primary: `AUTO` with `PERFORMANCE_HINT=LATENCY` → OpenVINO picks best available.
- Preference order: `GPU` (Iris Xe) → `CPU`.
- NPU only enabled if `Core().available_devices` reports `NPU` (Core Ultra).
- All models are compiled once at startup; `InferRequest` reused per model.

## Threading model

Sprint 1: single-threaded capture + inference. Fine at 15 FPS on this hardware.
Sprint 4+: split into (capture thread) → (inference thread pool via OpenVINO
async API) → (aggregator thread) if we hit an FPS wall. Frame handoff is
zero-copy numpy views; no cross-process shared memory required.
