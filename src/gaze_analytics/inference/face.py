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
        conf_threshold: float | None = None,
    ) -> None:
        self._cfg = cfg
        self._conf_threshold = conf_threshold if conf_threshold is not None else cfg.face_confidence_threshold
        self._min_face_height = cfg.min_face_height_px
        self._tiled = bool(cfg.face_tiled_detection)
        self._tile_grid = max(1, int(cfg.face_tile_grid))
        self._tile_overlap = float(np.clip(cfg.face_tile_overlap, 0.0, 0.5))
        model_path = self._resolve_model_path(cfg.models_dir, precision)
        log.info("Loading %s (%s) on device=%s", _MODEL_NAME, precision, cfg.device)
        core = ov.Core()
        cfg.ov_cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cfg.ov_cache_dir)})
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
        """Run detection and return face bboxes above the score threshold.

        If ``cfg.face_tiled_detection`` is True, splits the frame into an
        NxN grid (with overlap) and runs the detector on each tile before
        merging results with NMS. This dramatically improves small-face
        recall for crowded scenes at the cost of NxN more inference passes.

        ``frame_bgr`` must be a BGR uint8 image (as produced by cv2). The array
        is read but never mutated, copied, or persisted.
        """
        h, w = frame_bgr.shape[:2]
        if not self._tiled or self._tile_grid <= 1:
            return self._detect_region(frame_bgr, 0, 0, w, h)

        # Tiled path: compute overlapping tile rectangles then merge results.
        grid = self._tile_grid
        overlap = self._tile_overlap
        tile_w = int(round(w / grid))
        tile_h = int(round(h / grid))
        pad_x = int(round(tile_w * overlap))
        pad_y = int(round(tile_h * overlap))

        all_faces: list[FaceBBox] = []
        for gy in range(grid):
            for gx in range(grid):
                x0 = max(0, gx * tile_w - pad_x)
                y0 = max(0, gy * tile_h - pad_y)
                x1 = min(w, (gx + 1) * tile_w + pad_x)
                y1 = min(h, (gy + 1) * tile_h + pad_y)
                if x1 - x0 < 32 or y1 - y0 < 32:
                    continue
                tile = frame_bgr[y0:y1, x0:x1]
                all_faces.extend(self._detect_region(tile, x0, y0, x1 - x0, y1 - y0))

        return _nms_faces(all_faces, iou_threshold=0.35)

    def _detect_region(
        self,
        region_bgr: np.ndarray,
        offset_x: int,
        offset_y: int,
        region_w: int,
        region_h: int,
    ) -> list[FaceBBox]:
        """Run one forward pass on a region and return bboxes in FULL-FRAME coords.

        ``offset_x/offset_y`` shift the local tile coordinates back to the
        original frame's coordinate system so callers can treat the result
        uniformly.
        """
        resized = cv2.resize(region_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        blob = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
        result = self._compiled({self._input_key: blob})
        # SSD output layout: [1, 1, N, 7] = [image_id, label, conf, x0, y0, x1, y1] normalised
        detections = result[self._output_key][0][0]
        faces: list[FaceBBox] = []
        for det in detections:
            score = float(det[2])
            if score < self._conf_threshold:
                continue
            xmin = int(np.clip(det[3], 0.0, 1.0) * region_w) + offset_x
            ymin = int(np.clip(det[4], 0.0, 1.0) * region_h) + offset_y
            xmax = int(np.clip(det[5], 0.0, 1.0) * region_w) + offset_x
            ymax = int(np.clip(det[6], 0.0, 1.0) * region_h) + offset_y
            if xmax <= xmin or ymax <= ymin:
                continue
            if (ymax - ymin) < self._min_face_height:
                continue
            faces.append(FaceBBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, score=score))
        return faces


def _nms_faces(faces: list[FaceBBox], iou_threshold: float = 0.35) -> list[FaceBBox]:
    """Greedy non-max suppression over FaceBBox list, highest-score first.

    Needed for tiled detection: overlapping tiles will re-detect the same
    face; NMS collapses duplicates. Pure geometry, no image data touched.
    """
    if len(faces) <= 1:
        return list(faces)
    ordered = sorted(faces, key=lambda f: f.score, reverse=True)
    kept: list[FaceBBox] = []
    for cand in ordered:
        dup = False
        for k in kept:
            xa1 = max(cand.xmin, k.xmin)
            ya1 = max(cand.ymin, k.ymin)
            xa2 = min(cand.xmax, k.xmax)
            ya2 = min(cand.ymax, k.ymax)
            inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
            if inter == 0:
                continue
            area_c = (cand.xmax - cand.xmin) * (cand.ymax - cand.ymin)
            area_k = (k.xmax - k.xmin) * (k.ymax - k.ymin)
            union = area_c + area_k - inter
            if union > 0 and inter / union >= iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


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
