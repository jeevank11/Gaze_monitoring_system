"""Central configuration.

All tunables live here. Every value is overridable via environment variables
prefixed with ``GAZE_``, e.g. ``GAZE_CAMERA_INDEX=1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_sqlite_path() -> Path:
    """Return a default DB path that lives outside cloud-synced folders.

    OneDrive / Dropbox / etc. lock the SQLite file mid-write and cause
    "attempt to write a readonly database". We store under the user's local
    app-data dir instead, which sync tools do not touch. Overridable via
    ``GAZE_SQLITE_PATH``.
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(
            os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
        )
    return base / "gaze_analytics" / "metrics.sqlite"


def _default_ov_cache_dir() -> Path:
    """Local disk cache for OpenVINO compiled kernels.

    First run per model compiles GPU kernels (slow); subsequent runs load the
    cached blobs (fast). Overridable via ``GAZE_OV_CACHE_DIR``.
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(
            os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        )
    return base / "gaze_analytics" / "ov_cache"


class Settings(BaseSettings):
    """Runtime configuration for the gaze analytics pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="GAZE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Capture ------------------------------------------------------------
    camera_index: int = Field(default=0, description="OpenCV VideoCapture index")
    camera_backend: Literal["AUTO", "DSHOW", "MSMF", "V4L2"] = Field(
        default="AUTO", description="Camera backend; AUTO lets OpenCV pick"
    )
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 15

    # ---- Inference ----------------------------------------------------------
    device: Literal["AUTO", "CPU", "GPU", "NPU"] = Field(
        default="AUTO",
        description="OpenVINO inference device. AUTO picks best available.",
    )
    performance_hint: Literal["LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"] = "LATENCY"
    face_confidence_threshold: float = 0.7
    min_face_height_px: int = 80  # ignore faces smaller than this (filters screen artifacts)
    # Tiled (sliced) face detection: split the frame into a grid and run the
    # 300x300 SSD on each tile, then NMS. Dramatically improves small-face
    # recall for crowds; ~4x detector cost when grid=2 (2x2 tiles).
    face_tiled_detection: bool = False
    face_tile_grid: int = 2  # NxN tiles when face_tiled_detection is True
    face_tile_overlap: float = 0.15  # fractional overlap between tiles (avoids splitting faces on tile borders)

    # ---- Engagement thresholds ---------------------------------------------
    head_yaw_max_deg: float = 30.0
    head_pitch_max_deg: float = 25.0
    gaze_angle_max_deg: float = 15.0
    min_dwell_ms: int = 500
    # Hysteresis: a track must fail the attention check for this many
    # consecutive frames before being flipped to "away". Away→attending is
    # instant. Set to 1 to disable smoothing.
    attention_away_frames: int = 12

    # ---- Content segmentation -----------------------------------------------
    screen_capture_hz: float = 1.0
    phash_hamming_threshold: int = 12
    min_segment_seconds: int = 3
    keep_segment_thumbnail: bool = True
    thumbnail_size: tuple[int, int] = (160, 90)

    # ---- Aggregation & storage ---------------------------------------------
    aggregate_window_seconds: int = 5
    sqlite_path: Path = Field(default_factory=_default_sqlite_path)

    # ---- Runtime ------------------------------------------------------------
    preview: bool = False
    headless: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    models_dir: Path = REPO_ROOT / "models"
    ov_cache_dir: Path = Field(default_factory=_default_ov_cache_dir)
    # Optional local video file to use in place of the webcam (for testing).
    video_file: Path | None = None


settings = Settings()
