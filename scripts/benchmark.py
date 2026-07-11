"""End-to-end benchmark for the gaze pipeline.

Runs each OpenVINO inference stage in isolation on a synthetic 720p frame
and reports per-model latency, then chains them together to estimate a
realistic end-to-end FPS. This lets us fill in ``docs/BENCHMARKS.md``
without any faces, cameras, or persisted frames — the benchmark is fully
synthetic and privacy-neutral.

Usage:
    python scripts/benchmark.py --device CPU --iterations 200
    python scripts/benchmark.py --device GPU --produce-table
    python scripts/benchmark.py --device AUTO --output docs/BENCHMARKS.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow running as a plain script (``python scripts/benchmark.py``).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from gaze_analytics.config import Settings  # noqa: E402
from gaze_analytics.inference import (  # noqa: E402
    AgeGenderEstimator,
    FaceDetector,
    HeadPoseEstimator,
)


@dataclass(frozen=True)
class StageResult:
    name: str
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def _time_stage(
    name: str,
    call,
    frame: np.ndarray,
    iterations: int,
    warmup: int = 5,
) -> StageResult:
    """Time ``call(frame)`` and return latency percentiles."""
    for _ in range(warmup):
        call(frame)
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        call(frame)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return StageResult(
        name=name,
        iterations=iterations,
        mean_ms=statistics.fmean(samples),
        p50_ms=samples[len(samples) // 2],
        p95_ms=samples[int(len(samples) * 0.95)],
        p99_ms=samples[int(len(samples) * 0.99)],
    )


def _synthetic_frame_720p(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)


def _synthetic_face_crop(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)


def run(device: str, iterations: int) -> dict[str, StageResult]:
    cfg = Settings(device=device)  # type: ignore[arg-type]
    frame = _synthetic_frame_720p()
    face_crop = _synthetic_face_crop()

    results: dict[str, StageResult] = {}

    face_det = FaceDetector(cfg)
    results["face-detection"] = _time_stage(
        "face-detection", face_det.detect, frame, iterations
    )

    head_pose = HeadPoseEstimator(cfg)
    results["head-pose"] = _time_stage(
        "head-pose", head_pose.estimate, face_crop, iterations
    )

    age_gender = AgeGenderEstimator(cfg)
    results["age-gender"] = _time_stage(
        "age-gender", age_gender.estimate, face_crop, iterations
    )
    return results


def e2e_fps_estimate(results: dict[str, StageResult], faces_per_frame: int) -> float:
    """Estimate end-to-end FPS = 1 / (face_det + N * (pose + age_gender))."""
    per_frame_ms = (
        results["face-detection"].mean_ms
        + faces_per_frame
        * (results["head-pose"].mean_ms + results["age-gender"].mean_ms)
    )
    return 1000.0 / per_frame_ms if per_frame_ms > 0 else 0.0


def format_table(device: str, results: dict[str, StageResult]) -> str:
    lines: list[str] = []
    lines.append(f"### Device: `{device}`")
    lines.append("")
    lines.append("| Stage | Iterations | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results.values():
        lines.append(
            f"| {r.name} | {r.iterations} | {r.mean_ms:.2f} | "
            f"{r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} |"
        )
    lines.append("")
    fps_1 = e2e_fps_estimate(results, faces_per_frame=1)
    fps_3 = e2e_fps_estimate(results, faces_per_frame=3)
    lines.append(
        f"End-to-end estimate: **{fps_1:.1f} FPS** with 1 face, "
        f"**{fps_3:.1f} FPS** with 3 faces."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the gaze pipeline.")
    parser.add_argument(
        "--device",
        default="AUTO",
        choices=["CPU", "GPU", "AUTO"],
        help="OpenVINO device",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Iterations per stage (default 100)",
    )
    parser.add_argument(
        "--produce-table",
        action="store_true",
        help="Print a markdown table suitable for pasting into docs/BENCHMARKS.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown table to this file (appended to)",
    )
    args = parser.parse_args()

    print(f"Benchmarking device={args.device} iterations={args.iterations} ...")
    results = run(args.device, args.iterations)

    if args.produce_table or args.output is not None:
        table = format_table(args.device, results)
        print("\n" + table + "\n")
        if args.output is not None:
            with args.output.open("a", encoding="utf-8") as fh:
                fh.write("\n\n" + table + "\n")
            print(f"Appended to {args.output}")
    else:
        for r in results.values():
            print(
                f"  {r.name:<20s} mean={r.mean_ms:6.2f} ms  "
                f"p50={r.p50_ms:6.2f}  p95={r.p95_ms:6.2f}  p99={r.p99_ms:6.2f}"
            )
        print(
            f"\nEnd-to-end estimate: {e2e_fps_estimate(results, 1):.1f} FPS (1 face), "
            f"{e2e_fps_estimate(results, 3):.1f} FPS (3 faces)"
        )


if __name__ == "__main__":
    main()

