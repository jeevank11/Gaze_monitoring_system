# Gaze Analytics — Privacy-Preserving Signage Engagement

Real-time audience engagement analytics for digital signage, running entirely
on **low-power Intel CPU + iGPU** with **no face recognition, no cloud calls,
and no persisted frames**.

Built for **Intel Hackathon 2026 — Problem Statement 9: Privacy-Preserving
Gaze Analytics for Digital Signage**.

---

## What it does

Point a camera at the audience of any signage screen. The system reports:

- **Number of viewers** in the field of view
- **Viewing duration** (dwell time per attention session)
- **Estimated gender ratio** — aggregate only, never per-person
- **Audience engagement trends** — attention rate per auto-detected content segment

While playing anything on screen (YouTube, PPT, ad video, static image), the
pipeline **auto-segments the content** using perceptual hashing of the screen,
so metrics are tagged with *what* the audience was watching — without you
telling the system anything.

---

## Privacy guarantees

- **Frames are ephemeral.** Camera and screen frames live only in the current
  numpy buffer and are overwritten by the next grab. No `imwrite`, no video
  writer, no cache. Enforced by [`tests/test_privacy_smoke.py`](tests/test_privacy_smoke.py).
- **No face recognition.** No embeddings, no ReID, no identity storage. The
  tracker uses only bbox geometry + Kalman motion. IDs die with the process.
- **On-device only.** No cloud API. No telemetry. Everything runs in one
  Python process on the signage device.
- **Only aggregates persist.** SQLite stores counters, ratios, timestamps,
  content pHash fingerprints, and optional 160×90 content thumbnails —
  never faces, never full frames, never per-person records.

See [`docs/PRIVACY_THREAT_MODEL.md`](docs/PRIVACY_THREAT_MODEL.md) for the
full threat model.

---

## Hardware target

| | Requirement | This laptop |
|---|---|---|
| CPU | Intel x86_64, 4+ cores | ✅ Core i7-1185G7 (Tiger Lake) |
| iGPU | Intel HD/UHD/Iris | ✅ Iris Xe |
| NPU | Optional (Core Ultra) | ❌ — falls back to iGPU |
| RAM | 4 GB free | ✅ |
| Camera | Any UVC / built-in | ✅ built-in webcam |

**Target performance:** ≥15 FPS end-to-end at 720p, <2 TOPS peak, <15 W package
power. Pipeline scales down cleanly to Intel Atom and Compute Stick SKUs.

---

## Quick start

```powershell
# 1. Clone (or you already have this workspace open)
git clone https://github.com/Jeevank11/Gaze_monitoring_system.git
cd Gaze_monitoring_system

# 2. Python 3.12 venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install
pip install -e ".[dev]"

# 4. Fetch pretrained models (~150 MB, one-time)
python scripts/download_models.py

# 5. Run — dev preview
python -m gaze_analytics --preview

# 6. Run — production (headless) + dashboard
.\scripts\run_all.ps1
# open http://localhost:8501
```

---

## Architecture (one-glance)

```
Webcam ──▶ Person → Face → HeadPose → (Gaze) → AgeGender ──▶ Aggregator ──▶ SQLite
                                              │                    ▲
Screen ──▶ pHash → Segmenter (content ID) ────┘────────────────────┘
                                                                    │
                                                              Streamlit UI
```

Full walkthrough in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```
.github/
  agents/         — Copilot agent + persistent learnings
  instructions/   — always-loaded Copilot rules (privacy contract lives here)
  skills/         — reusable workflows (run, download models, benchmark, deploy)
  prompts/        — one-shot prompt templates
  workflows/      — CI (lint, test, release build)
src/gaze_analytics/
  capture/        — webcam + screen frame sources (frames discarded)
  inference/      — OpenVINO wrappers (person, face, head-pose, gaze, age-gender)
  tracker/        — IoU + Kalman tracker (no embeddings)
  content/        — pHash + scene segmenter (auto content tagging)
  engagement/     — attention gating, dwell, aggregation
  privacy/        — silhouette / bbox-only render for demo
  storage/        — SQLite sink (aggregates only)
dashboard/        — Streamlit UI
scripts/          — download models, run everything, benchmark
deploy/           — Docker, PyInstaller .exe, systemd services
docs/             — architecture, privacy, pitch, benchmarks
tests/            — pytest suite (including privacy grep)
```

---

## Sprints

1. ✅ **Sprint 1** — Bootstrap + webcam capture (frames discarded, FPS reported)
2. ✅ Sprint 2 — Face detection + debug overlay
3. ✅ Sprint 3 — Head-pose + attention gate
4. ✅ Sprint 4 — IoU/Kalman tracker + pHash content segmenter
5. ✅ Sprint 5 — Age/gender aggregator + SQLite sink
6. ✅ Sprint 6 — Streamlit dashboard (live)
7. ✅ Sprint 7 — Privacy-safe render mode + polish
8. ✅ Sprint 8 — Benchmark table + Docker + PyInstaller + pitch script

---

## Deployment

Three targets, one codebase. See [`.github/skills/deploy/SKILL.md`](.github/skills/deploy/SKILL.md).

| Target | Command |
|---|---|
| Local demo | `.\scripts\run_all.ps1` |
| Windows `.exe` | `.\deploy\windows\build_exe.ps1` |
| Docker | `docker compose -f deploy/docker/docker-compose.yml up` |
| Linux systemd | `sudo systemctl enable --now gaze-analytics gaze-dashboard` |
| GitHub Release | `git tag v0.1.0 && git push --tags` (CI builds artifacts) |

---

## Contributing

- Branch: `feature/hackathon-mvp` (rebase onto `develop`, then PR to `main`).
- Commit style: `sprint-N: <summary>` / `fix: <summary>` / `test: <summary>`.
- Run `ruff check src tests` and `pytest -q` before every commit.
- Anything face-recognition, ReID, embedding, or cloud-inference is
  automatically rejected — see [`.github/instructions/copilot-instructions.md`](.github/instructions/copilot-instructions.md).

## License

MIT — see [`LICENSE`](LICENSE).
