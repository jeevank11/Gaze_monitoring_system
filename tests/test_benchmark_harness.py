"""Tests for the Sprint 8 benchmark harness.

Verifies that the benchmark runs on synthetic frames (no webcam),
produces sensible latency numbers, and formats a valid markdown table.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_PATH = _REPO_ROOT / "scripts" / "benchmark.py"


@pytest.fixture(scope="module")
def bench_module():
    """Load scripts/benchmark.py as a module."""
    spec = importlib.util.spec_from_file_location("bench_harness", _BENCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_frame_shape(bench_module):
    frame = bench_module._synthetic_frame_720p()
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype.name == "uint8"


def test_synthetic_face_crop_shape(bench_module):
    crop = bench_module._synthetic_face_crop()
    assert crop.shape == (128, 128, 3)


def test_time_stage_returns_percentiles(bench_module):
    import numpy as np

    calls = {"count": 0}

    def noop(_frame):
        calls["count"] += 1

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = bench_module._time_stage("noop", noop, frame, iterations=20, warmup=2)
    assert result.name == "noop"
    assert result.iterations == 20
    assert calls["count"] == 22  # warmup + iterations
    assert result.mean_ms >= 0.0
    assert result.p50_ms >= 0.0
    assert result.p95_ms >= result.p50_ms
    assert result.p99_ms >= result.p95_ms


def test_e2e_fps_estimate(bench_module):
    results = {
        "face-detection": bench_module.StageResult(
            "face-detection", 10, mean_ms=10.0, p50_ms=10.0, p95_ms=12.0, p99_ms=15.0
        ),
        "head-pose": bench_module.StageResult(
            "head-pose", 10, mean_ms=2.0, p50_ms=2.0, p95_ms=3.0, p99_ms=4.0
        ),
        "age-gender": bench_module.StageResult(
            "age-gender", 10, mean_ms=1.0, p50_ms=1.0, p95_ms=1.5, p99_ms=2.0
        ),
    }
    # 1 face: 10 + 1*(2+1) = 13 ms -> ~76.9 FPS
    fps1 = bench_module.e2e_fps_estimate(results, faces_per_frame=1)
    assert 76.0 < fps1 < 78.0
    # 3 faces: 10 + 3*(2+1) = 19 ms -> ~52.6 FPS
    fps3 = bench_module.e2e_fps_estimate(results, faces_per_frame=3)
    assert 52.0 < fps3 < 53.5


def test_format_table_contains_expected_columns(bench_module):
    results = {
        "face-detection": bench_module.StageResult(
            "face-detection", 5, mean_ms=8.0, p50_ms=8.0, p95_ms=10.0, p99_ms=12.0
        ),
        "head-pose": bench_module.StageResult(
            "head-pose", 5, mean_ms=2.0, p50_ms=2.0, p95_ms=3.0, p99_ms=4.0
        ),
        "age-gender": bench_module.StageResult(
            "age-gender", 5, mean_ms=1.0, p50_ms=1.0, p95_ms=1.5, p99_ms=2.0
        ),
    }
    table = bench_module.format_table("CPU", results)
    assert "### Device: `CPU`" in table
    assert "| Stage | Iterations | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |" in table
    assert "face-detection" in table
    assert "head-pose" in table
    assert "age-gender" in table
    assert "End-to-end estimate" in table
    assert "1 face" in table
    assert "3 faces" in table


def test_format_table_no_banned_persistence_terms(bench_module):
    """Privacy sanity: table must not mention persisted frames or images."""
    results = {
        "face-detection": bench_module.StageResult(
            "face-detection", 1, mean_ms=1.0, p50_ms=1.0, p95_ms=1.0, p99_ms=1.0
        ),
        "head-pose": bench_module.StageResult(
            "head-pose", 1, mean_ms=1.0, p50_ms=1.0, p95_ms=1.0, p99_ms=1.0
        ),
        "age-gender": bench_module.StageResult(
            "age-gender", 1, mean_ms=1.0, p50_ms=1.0, p95_ms=1.0, p99_ms=1.0
        ),
    }
    table = bench_module.format_table("CPU", results).lower()
    for banned in ("imwrite", "videowriter", "pickle", ".jpg saved", ".png saved"):
        assert banned not in table
