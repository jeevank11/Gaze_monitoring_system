"""Tests for :mod:`gaze_analytics.inference.head_pose` and the attention gate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gaze_analytics.config import Settings
from gaze_analytics.engagement import is_attending
from gaze_analytics.inference.head_pose import HeadPose, HeadPoseEstimator


def _make_fake_ov_core(yaw: float, pitch: float, roll: float) -> MagicMock:
    """Fake ``ov.Core`` returning fixed yaw/pitch/roll."""
    yaw_key = MagicMock(name="yaw_key")
    pitch_key = MagicMock(name="pitch_key")
    roll_key = MagicMock(name="roll_key")
    input_key = MagicMock(name="input_key")

    compiled = MagicMock(name="compiled_model")
    compiled.input.return_value = input_key

    def _output(name):
        return {
            "angle_y_fc": yaw_key,
            "angle_p_fc": pitch_key,
            "angle_r_fc": roll_key,
        }[name]

    compiled.output.side_effect = _output
    compiled.side_effect = lambda _feed: {
        yaw_key: np.array([[yaw]], dtype=np.float32),
        pitch_key: np.array([[pitch]], dtype=np.float32),
        roll_key: np.array([[roll]], dtype=np.float32),
    }

    core = MagicMock(name="Core")
    core.read_model.return_value = MagicMock(name="model")
    core.compile_model.return_value = compiled
    return core


@pytest.fixture
def _model_dir(tmp_path):
    """Materialise an empty FP16 IR file so the path resolver succeeds."""
    d = tmp_path / "models" / "intel" / "head-pose-estimation-adas-0001" / "FP16"
    d.mkdir(parents=True)
    (d / "head-pose-estimation-adas-0001.xml").write_text("<net/>")
    return tmp_path / "models"


def test_estimator_returns_headpose_dataclass(_model_dir) -> None:
    core = _make_fake_ov_core(yaw=5.0, pitch=-3.0, roll=1.0)
    with patch("gaze_analytics.inference.head_pose.ov.Core", return_value=core):
        est = HeadPoseEstimator(cfg=Settings(models_dir=_model_dir))
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        pose = est.estimate(crop)
    assert isinstance(pose, HeadPose)
    assert pose.yaw == pytest.approx(5.0)
    assert pose.pitch == pytest.approx(-3.0)
    assert pose.roll == pytest.approx(1.0)


def test_estimator_does_not_mutate_input_crop(_model_dir) -> None:
    core = _make_fake_ov_core(yaw=0.0, pitch=0.0, roll=0.0)
    with patch("gaze_analytics.inference.head_pose.ov.Core", return_value=core):
        est = HeadPoseEstimator(cfg=Settings(models_dir=_model_dir))
        crop = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
        snap = crop.copy()
        est.estimate(crop)
    assert np.array_equal(crop, snap), "estimate() must not mutate the input crop"


@pytest.mark.parametrize(
    "yaw,pitch,expected",
    [
        (0.0, 0.0, True),
        (24.9, 19.9, True),
        (25.0, 20.0, True),
        (26.0, 0.0, False),
        (0.0, 21.0, False),
        (-30.0, 0.0, False),
        (0.0, -25.0, False),
    ],
)
def test_is_attending_yaw_and_pitch_bounds(yaw: float, pitch: float, expected: bool) -> None:
    pose = HeadPose(yaw=yaw, pitch=pitch, roll=0.0)
    cfg = Settings()  # defaults: yaw=25, pitch=20
    assert is_attending(pose, cfg) is expected


def test_is_attending_ignores_roll() -> None:
    """A tilted head still counts — roll is not part of the gate."""
    pose = HeadPose(yaw=0.0, pitch=0.0, roll=45.0)
    assert is_attending(pose, Settings()) is True
