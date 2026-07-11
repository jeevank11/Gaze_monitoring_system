"""Tests for :mod:`gaze_analytics.inference.face`.

We do NOT load real OpenVINO models here. The tests double-out ``ov.Core``
so they run in the same lightweight environment as the rest of the suite
(no models/ folder required, no GPU needed).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gaze_analytics.config import Settings
from gaze_analytics.inference.face import (
    FaceBBox,
    FaceDetector,
    draw_face_bboxes,
)


def _make_fake_ov_core(detections: np.ndarray) -> MagicMock:
    """Build a fake ``ov.Core`` whose compiled model returns ``detections``.

    ``detections`` must be shape (N, 7): [image_id, label, conf, x0, y0, x1, y1].
    We wrap it as [1, 1, N, 7] to match the SSD output layout.
    """
    output_key = MagicMock(name="output_key")
    input_key = MagicMock(name="input_key")

    packed = detections.reshape(1, 1, -1, 7)
    compiled = MagicMock(name="compiled_model")
    compiled.input.return_value = input_key
    compiled.output.return_value = output_key
    compiled.side_effect = lambda _feed: {output_key: packed}

    core = MagicMock(name="Core")
    core.read_model.return_value = MagicMock(name="model")
    core.compile_model.return_value = compiled
    return core


@pytest.fixture(autouse=True)
def _mock_model_files(tmp_path, monkeypatch):
    """Create empty stand-in model IR so ``_resolve_model_path`` succeeds."""
    models_dir = tmp_path / "models"
    model_dir = models_dir / "intel" / "face-detection-retail-0004" / "FP16-INT8"
    model_dir.mkdir(parents=True)
    (model_dir / "face-detection-retail-0004.xml").write_text("<net/>")
    monkeypatch.setenv("GAZE_MODELS_DIR", str(models_dir))
    return models_dir


def test_detector_filters_by_confidence(_mock_model_files) -> None:
    # one strong, one weak
    detections = np.array(
        [
            [0, 1, 0.95, 0.10, 0.20, 0.30, 0.40],
            [0, 1, 0.30, 0.50, 0.60, 0.70, 0.80],
        ],
        dtype=np.float32,
    )
    core = _make_fake_ov_core(detections)
    with patch("gaze_analytics.inference.face.ov.Core", return_value=core):
        det = FaceDetector(cfg=Settings(models_dir=_mock_model_files), conf_threshold=0.6)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        faces = det.detect(frame)

    assert len(faces) == 1
    (face,) = faces
    assert face.score == pytest.approx(0.95, rel=1e-4)
    # 0.10*1280=128, 0.20*720=144, 0.30*1280=384, 0.40*720=288
    assert (face.xmin, face.ymin, face.xmax, face.ymax) == (128, 144, 384, 288)


def test_detector_clips_and_rejects_degenerate_boxes(_mock_model_files) -> None:
    """Out-of-range coords are clipped; degenerate boxes are dropped."""
    detections = np.array(
        [
            # coords outside [0,1] are clipped
            [0, 1, 0.9, -0.5, -0.5, 1.5, 1.5],
            # zero-area box after clipping is rejected
            [0, 1, 0.9, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    core = _make_fake_ov_core(detections)
    with patch("gaze_analytics.inference.face.ov.Core", return_value=core):
        det = FaceDetector(cfg=Settings(models_dir=_mock_model_files), conf_threshold=0.5)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        faces = det.detect(frame)

    assert len(faces) == 1
    assert (faces[0].xmin, faces[0].ymin, faces[0].xmax, faces[0].ymax) == (0, 0, 640, 360)


def test_detector_does_not_mutate_input_frame(_mock_model_files) -> None:
    detections = np.array([[0, 1, 0.9, 0.1, 0.1, 0.2, 0.2]], dtype=np.float32)
    core = _make_fake_ov_core(detections)
    with patch("gaze_analytics.inference.face.ov.Core", return_value=core):
        det = FaceDetector(cfg=Settings(models_dir=_mock_model_files))
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        snapshot = frame.copy()
        det.detect(frame)
    assert np.array_equal(frame, snapshot), "detect() must not mutate the input frame"


def test_resolve_model_path_falls_back_to_fp16(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    fp16_dir = models_dir / "intel" / "face-detection-retail-0004" / "FP16"
    fp16_dir.mkdir(parents=True)
    (fp16_dir / "face-detection-retail-0004.xml").write_text("<net/>")
    with patch(
        "gaze_analytics.inference.face.ov.Core",
        return_value=_make_fake_ov_core(np.zeros((0, 7), dtype=np.float32)),
    ):
        det = FaceDetector(cfg=Settings(models_dir=models_dir), precision="FP16-INT8")
    assert det is not None  # constructed without raising


def test_draw_face_bboxes_annotates_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bboxes = [FaceBBox(xmin=10, ymin=10, xmax=40, ymax=40, score=0.87)]
    draw_face_bboxes(frame, bboxes)
    # cv2.rectangle draws a green edge; at least one pixel should be non-black now.
    assert frame.any(), "expected the overlay to draw at least one pixel"
