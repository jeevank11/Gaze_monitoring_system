"""Age & gender estimator — wrapper around ``age-gender-recognition-retail-0013``.

Runs on a face crop and returns *aggregate-friendly* attributes: an estimated
age in years, and a gender label ``"M"``/``"F"`` with its confidence.

Privacy note: the caller MUST NOT persist per-face results. This module only
returns scalars — no embeddings, no biometric templates. Downstream code is
expected to fold results into rolling window counts and then discard them.
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

_MODEL_NAME = "age-gender-recognition-retail-0013"
_INPUT_H = 62
_INPUT_W = 62
_AGE_LAYER = "age_conv3"
_GENDER_LAYER = "prob"

Precision = Literal["FP32", "FP16", "FP16-INT8"]


@dataclass(frozen=True)
class AgeGender:
    """Coarse demographic estimate. Never combined with an identity."""

    age: float
    gender: Literal["M", "F"]
    gender_confidence: float


class AgeGenderEstimator:
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
        self._age_key = self._compiled.output(_AGE_LAYER)
        self._gender_key = self._compiled.output(_GENDER_LAYER)

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

    def estimate(self, face_crop_bgr: np.ndarray) -> AgeGender:
        """Run age/gender inference on a single face crop.

        The crop is read but not mutated or persisted.
        """
        resized = cv2.resize(face_crop_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        blob = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
        result = self._compiled({self._input_key: blob})
        age_years = float(result[self._age_key].flatten()[0]) * 100.0
        # Gender: [prob_female, prob_male].
        gender_probs = result[self._gender_key].flatten()
        prob_f = float(gender_probs[0])
        prob_m = float(gender_probs[1])
        if prob_m >= prob_f:
            label: Literal["M", "F"] = "M"
            conf = prob_m
        else:
            label = "F"
            conf = prob_f
        return AgeGender(age=age_years, gender=label, gender_confidence=conf)
