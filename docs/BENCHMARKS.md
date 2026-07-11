# Benchmarks

Measured with `scripts/benchmark.py --iterations 50` on synthetic 720p frames.
Numbers are **model inference latency only** — capture, tracker, aggregator,
and privacy render add ~1–2 ms combined per frame.

## Test rig

- **CPU:** Intel Core i7-1185G7 (Tiger Lake, 11th gen, 4C/8T)
- **iGPU:** Intel Iris Xe
- **NPU:** none
- **OS:** Windows 11
- **OpenVINO:** 2024.6
- **Frame source:** synthetic 720p RGB noise (benchmark is camera-free)

## Per-stage latency and end-to-end FPS

### Device: `CPU`

| Stage | Iterations | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|
| face-detection | 50 | 12.23 | 12.11 | 17.62 | 22.16 |
| head-pose | 50 | 2.53 | 2.52 | 4.20 | 4.36 |
| age-gender | 50 | 1.36 | 1.31 | 2.30 | 2.88 |

End-to-end estimate: **62.0 FPS** with 1 face, **41.9 FPS** with 3 faces.

### Device: `GPU`

| Stage | Iterations | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|
| face-detection | 50 | 8.46 | 8.24 | 13.14 | 17.00 |
| head-pose | 50 | 2.05 | 1.88 | 3.23 | 3.72 |
| age-gender | 50 | 1.13 | 1.09 | 1.55 | 1.67 |

End-to-end estimate: **85.9 FPS** with 1 face, **55.6 FPS** with 3 faces.

### Device: `AUTO`

| Stage | Iterations | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|
| face-detection | 50 | 8.94 | 7.33 | 22.63 | 55.37 |
| head-pose | 50 | 7.02 | 5.89 | 14.06 | 15.92 |
| age-gender | 50 | 2.34 | 2.02 | 3.86 | 7.16 |

End-to-end estimate: **54.6 FPS** with 1 face, **27.0 FPS** with 3 faces.

> `AUTO` shows a longer warm-up tail (visible in p99) because it profiles
> both CPU and GPU on first inferences before settling on GPU. For a
> pre-baked deployment prefer `--device GPU`; for portability keep `AUTO`.

## Real-time budget

Privacy contract target: **≥15 FPS end-to-end at 720p** on Intel Core
i7-1185G7 + Iris Xe. Achieved with substantial headroom on all three
devices, even with 3 simultaneous faces.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --device CPU  --iterations 50 --produce-table
.\.venv\Scripts\python.exe scripts/benchmark.py --device GPU  --iterations 50 --produce-table
.\.venv\Scripts\python.exe scripts/benchmark.py --device AUTO --iterations 50 --produce-table
```

Or append directly into this document:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --device CPU --iterations 100 --output docs/BENCHMARKS.md
```

The benchmark runs on synthetic frames only — it does **not** open the
webcam, does **not** write frames to disk, and does **not** hit the network.
