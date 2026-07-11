"""Head-pose estimator — wrapper around ``head-pose-estimation-adas-0001``.

Given a face crop, returns yaw / pitch / roll in degrees. The face crop is
only ever a numpy view into the caller's transient frame buffer — nothing
is persisted, nothing is copied out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import openvino as ov

from gaze_analytics.config import Settings, settings

log = logging.getLogger(__name__)

_MODEL_NAME = "head-pose-estimation-adas-0001"
_INPUT_H = 60
_INPUT_W = 60

# The model has three scalar outputs. Their friendly names in the IR are:
_YAW_LAYER = "angle_y_fc"
_PITCH_LAYER = "angle_p_fc"
_ROLL_LAYER = "angle_r_fc"

Precision = Literal["FP32", "FP16", "FP16-INT8"]


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees. All values relative to the camera axis."""

    yaw: float
    pitch: float
    roll: float


class HeadPoseEstimator:
    """OpenVINO wrapper. FP16 is the recommended precision for this model."""

    def __init__(
        self,
        cfg: Settings = settings,
        precision: Precision = "FP16",
    ) -> None:
        self._cfg = cfg
        model_path = self._resolve_model_path(cfg.models_dir, precision)
        log.info("Loading %s (%s) on device=%s", _MODEL_NAME, precision, cfg.device)
        core = ov.Core()
        model = core.read_model(str(model_path))
        self._compiled = core.compile_model(
            model,
            device_name=cfg.device,
            config={"PERFORMANCE_HINT": cfg.performance_hint},
        )
        self._input_key = self._compiled.input(0)
        # Cache output nodes by name so we don't hit the OVDict lookup each frame.
        self._yaw_key = self._compiled.output(_YAW_LAYER)
        self._pitch_key = self._compiled.output(_PITCH_LAYER)
        self._roll_key = self._compiled.output(_ROLL_LAYER)

    @staticmethod
    def _resolve_model_path(models_dir: Path, precision: Precision) -> Path:
        candidate = models_dir / "intel" / _MODEL_NAME / precision / f"{_MODEL_NAME}.xml"
        if candidate.exists():
            return candidate
        fallback = models_dir / "intel" / _MODEL_NAME / "FP16" / f"{_MODEL_NAME}.xml"
        if fallback.exists():
            log.warning("%s precision %s not found; falling back to FP16", _MODEL_NAME, precision)
            return fallback
        raise FileNotFoundError(
            f"Could not find {_MODEL_NAME} at {candidate}. "
            "Run `python scripts/download_models.py` first."
        )

    def estimate(self, face_crop_bgr: np.ndarray) -> HeadPose:
        """Run head-pose inference on a single face crop.

        The crop is read but not mutated or persisted.
        """
        resized = cv2.resize(face_crop_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        blob = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
        result = self._compiled({self._input_key: blob})
        return HeadPose(
            yaw=float(result[self._yaw_key].flatten()[0]),
            pitch=float(result[self._pitch_key].flatten()[0]),
            roll=float(result[self._roll_key].flatten()[0]),
        )
