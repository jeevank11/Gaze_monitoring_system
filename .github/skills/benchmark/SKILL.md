# SKILL: Benchmark for the Pitch

Produce the numbers judges want to see: FPS, latency, power, per-device
comparison. Output goes into `docs/BENCHMARKS.md`.

## Prerequisites

- Analytics pipeline builds and runs.
- Windows PowerShell 5.1+ or 7.
- (Optional) `Intel Power Gadget` or `powermetrics` equivalent for Watts.

## Measure FPS per device

```powershell
python scripts/benchmark.py --device CPU  --seconds 30
python scripts/benchmark.py --device GPU  --seconds 30
python scripts/benchmark.py --device AUTO --seconds 30
```

Each run prints:

```
device=GPU  frames=487  fps=16.2  p50_latency=52 ms  p95_latency=71 ms
```

## Produce the pitch table

```powershell
python scripts/benchmark.py --produce-table >> docs/BENCHMARKS.md
```

Appends a Markdown table like:

| Device | Model Precision | Avg FPS | p95 Latency | Notes |
|---|---|---|---|---|
| CPU  | FP16-INT8 | 12.3 | 95 ms | Fallback |
| GPU  | FP16-INT8 | 16.2 | 71 ms | Preferred |
| AUTO | FP16-INT8 | 15.8 | 74 ms | Recommended |

## What the pitch needs

1. **One FPS number** that exceeds your real-time floor (≥15 FPS).
2. **One TOPS claim** — models total <2 TOPS peak on iGPU. Say so.
3. **One power number** — even a rough Watts figure from Task Manager /
   Intel Power Gadget beats no number. Frame it as "10–12 W package power
   during inference, vs. 45+ W for a discrete-GPU baseline."
4. **One privacy proof screenshot** — `dir data\*.jpg` returning empty,
   captured while the pipeline is running.
