"""Face detector — thin OpenVINO wrapper around ``face-detection-retail-0004``.

Privacy contract (see .github/instructions/copilot-instructions.md):
    * We only ever look at pixels living in the caller's frame buffer.
    * We NEVER copy a face crop, save it, or send it anywhere. This module
      returns integer bounding-box coordinates only.
    * There is no embedding, no descriptor, no identity — just SSD scores.
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

# Model input spec for face-detection-retail-0004: NCHW, 1x3x300x300, BGR uint8.
_MODEL_NAME = "face-detection-retail-0004"
_INPUT_H = 300
_INPUT_W = 300

Precision = Literal["FP32", "FP16", "FP16-INT8"]


@dataclass(frozen=True)
class FaceBBox:
    """A single face detection in original-frame pixel coordinates.

    Only integers and a scalar score — no image data. Safe to log, aggregate,
    or persist as counts.
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    score: float

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


class FaceDetector:
    """SSD face detector running on OpenVINO ``AUTO`` (prefers iGPU on this box).

    Uses ``FP16-INT8`` by default for smallest footprint on Iris Xe; falls
    back to ``FP16`` if the INT8 IR is not present on disk.
    """

    def __init__(
        self,
        cfg: Settings = settings,
        precision: Precision = "FP16-INT8",
        conf_threshold: float = 0.6,
    ) -> None:
        self._cfg = cfg
        self._conf_threshold = conf_threshold
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
        self._output_key = self._compiled.output(0)

    @staticmethod
    def _resolve_model_path(models_dir: Path, precision: Precision) -> Path:
        """Return the model XML path, falling back precision → FP16 if needed."""
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

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBBox]:
        """Run one forward pass and return face bboxes above the score threshold.

        ``frame_bgr`` must be a BGR uint8 image (as produced by cv2). The array
        is read but never mutated, copied, or persisted.
        """
        h, w = frame_bgr.shape[:2]
        # cv2.resize returns a small (300x300x3) view we discard after inference.
        resized = cv2.resize(frame_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        # NHWC(uint8) -> NCHW(float32); model accepts float input, so this
        # is the cheapest conversion that satisfies the API.
        blob = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
        result = self._compiled({self._input_key: blob})
        # SSD output layout: [1, 1, N, 7] = [image_id, label, conf, x0, y0, x1, y1] normalised
        detections = result[self._output_key][0][0]
        faces: list[FaceBBox] = []
        for det in detections:
            score = float(det[2])
            if score < self._conf_threshold:
                continue
            xmin = int(np.clip(det[3], 0.0, 1.0) * w)
            ymin = int(np.clip(det[4], 0.0, 1.0) * h)
            xmax = int(np.clip(det[5], 0.0, 1.0) * w)
            ymax = int(np.clip(det[6], 0.0, 1.0) * h)
            if xmax <= xmin or ymax <= ymin:
                continue
            faces.append(FaceBBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, score=score))
        return faces


def draw_face_bboxes(frame_bgr: np.ndarray, faces: list[FaceBBox]) -> None:
    """In-place bbox overlay for the debug preview window.

    Draws only rectangles + score labels. No face pixels are copied out.
    """
    for f in faces:
        cv2.rectangle(frame_bgr, (f.xmin, f.ymin), (f.xmax, f.ymax), (0, 255, 0), 2)
        label = f"face {f.score:.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (f.xmin, max(0, f.ymin - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
