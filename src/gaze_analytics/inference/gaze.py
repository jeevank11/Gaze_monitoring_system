"""Gaze estimator — wrapper around ``gaze-estimation-adas-0002``.

Given left-eye crop, right-eye crop, and head-pose angles, returns a 3D gaze
vector. The yaw/pitch of the gaze vector determines whether the viewer is
actually looking at the screen (not just facing it).

Inputs:
    - left_eye_image: [1, 3, 60, 60] NCHW FP32
    - right_eye_image: [1, 3, 60, 60] NCHW FP32
    - head_pose_angles: [1, 3] FP32 (yaw, pitch, roll in degrees)

Output:
    - gaze_vector: [1, 3] FP32 — unit direction vector (x, y, z)

Privacy note: only scalar angles are returned. No embeddings, no images stored.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import openvino as ov

from gaze_analytics.config import Settings, settings
from gaze_analytics.inference.head_pose import HeadPose

log = logging.getLogger(__name__)

_MODEL_NAME = "gaze-estimation-adas-0002"
_EYE_INPUT_H = 60
_EYE_INPUT_W = 60

Precision = Literal["FP32", "FP16", "FP16-INT8"]


@dataclass(frozen=True)
class GazeVector:
    """3D gaze direction + derived angles."""

    x: float
    y: float
    z: float
    yaw_deg: float   # horizontal gaze angle
    pitch_deg: float  # vertical gaze angle


def _extract_eye_crops(
    face_crop_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract left and right eye regions from a face crop using fixed ratios.

    Uses the anthropometric proportions of the face to estimate eye regions
    without a dedicated landmark model (keeps the pipeline fast).
    """
    h, w = face_crop_bgr.shape[:2]

    # Eye region vertical: ~25%-50% of face height
    y_top = int(h * 0.25)
    y_bot = int(h * 0.50)

    # Left eye: 10%-48% width; Right eye: 52%-90% width
    left_eye = face_crop_bgr[y_top:y_bot, int(w * 0.10):int(w * 0.48)]
    right_eye = face_crop_bgr[y_top:y_bot, int(w * 0.52):int(w * 0.90)]

    return left_eye, right_eye


class GazeEstimator:
    """OpenVINO wrapper for gaze-estimation-adas-0002."""

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
        # Map inputs by name
        self._left_eye_key = self._compiled.input("left_eye_image")
        self._right_eye_key = self._compiled.input("right_eye_image")
        self._head_pose_key = self._compiled.input("head_pose_angles")
        self._output_key = self._compiled.output(0)

    @staticmethod
    def _resolve_model_path(models_dir: Path, precision: Precision) -> Path:
        candidate = models_dir / "intel" / _MODEL_NAME / precision / f"{_MODEL_NAME}.xml"
        if candidate.exists():
            return candidate
        fallback = models_dir / "intel" / _MODEL_NAME / "FP16" / f"{_MODEL_NAME}.xml"
        if fallback.exists():
            log.warning(
                "%s precision %s not found; falling back to FP16", _MODEL_NAME, precision
            )
            return fallback
        raise FileNotFoundError(
            f"Could not find {_MODEL_NAME} at {candidate}. "
            "Run `python scripts/download_models.py` first."
        )

    def estimate(self, face_crop_bgr: np.ndarray, head_pose: HeadPose) -> GazeVector:
        """Run gaze estimation on a face crop with known head pose.

        Returns a GazeVector with 3D direction and yaw/pitch angles.
        """
        left_eye, right_eye = _extract_eye_crops(face_crop_bgr)

        # Ensure valid crops
        if left_eye.size == 0 or right_eye.size == 0:
            # Fallback: return a forward-looking gaze
            return GazeVector(x=0.0, y=0.0, z=-1.0, yaw_deg=0.0, pitch_deg=0.0)

        left_blob = self._preprocess_eye(left_eye)
        right_blob = self._preprocess_eye(right_eye)
        pose_blob = np.array(
            [[head_pose.yaw, head_pose.pitch, head_pose.roll]], dtype=np.float32
        )

        result = self._compiled(
            {
                self._left_eye_key: left_blob,
                self._right_eye_key: right_blob,
                self._head_pose_key: pose_blob,
            }
        )

        gaze_vec = result[self._output_key].flatten()
        x, y, z = float(gaze_vec[0]), float(gaze_vec[1]), float(gaze_vec[2])

        # Convert 3D vector to yaw/pitch angles
        gaze_yaw = math.degrees(math.atan2(x, -z))
        length_xz = math.sqrt(x * x + z * z)
        gaze_pitch = math.degrees(math.atan2(y, length_xz))

        return GazeVector(
            x=x, y=y, z=z,
            yaw_deg=gaze_yaw,
            pitch_deg=gaze_pitch,
        )

    @staticmethod
    def _preprocess_eye(eye_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            eye_bgr, (_EYE_INPUT_W, _EYE_INPUT_H), interpolation=cv2.INTER_LINEAR
        )
        return np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
