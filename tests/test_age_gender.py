"""Tests for :mod:`gaze_analytics.inference.age_gender`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gaze_analytics.config import Settings
from gaze_analytics.inference.age_gender import AgeGender, AgeGenderEstimator


def _make_fake_ov_core(age_scaled: float, prob_female: float, prob_male: float) -> MagicMock:
    age_key = MagicMock(name="age_key")
    gender_key = MagicMock(name="gender_key")
    input_key = MagicMock(name="input_key")

    compiled = MagicMock(name="compiled_model")
    compiled.input.return_value = input_key

    def _output(name: str):
        return {"age_conv3": age_key, "prob": gender_key}[name]

    compiled.output.side_effect = _output
    compiled.side_effect = lambda _feed: {
        age_key: np.array([[[[age_scaled]]]], dtype=np.float32),
        gender_key: np.array([[[[prob_female]], [[prob_male]]]], dtype=np.float32),
    }

    core = MagicMock(name="Core")
    core.read_model.return_value = MagicMock(name="model")
    core.compile_model.return_value = compiled
    return core


@pytest.fixture
def _model_dir(tmp_path):
    d = tmp_path / "models" / "intel" / "age-gender-recognition-retail-0013" / "FP16"
    d.mkdir(parents=True)
    (d / "age-gender-recognition-retail-0013.xml").write_text("<net/>")
    return tmp_path / "models"


def test_estimator_scales_age_by_100(_model_dir) -> None:
    core = _make_fake_ov_core(age_scaled=0.42, prob_female=0.1, prob_male=0.9)
    with patch("gaze_analytics.inference.age_gender.ov.Core", return_value=core):
        est = AgeGenderEstimator(cfg=Settings(models_dir=_model_dir))
        result = est.estimate(np.zeros((80, 80, 3), dtype=np.uint8))
    assert isinstance(result, AgeGender)
    assert result.age == pytest.approx(42.0)
    assert result.gender == "M"
    assert result.gender_confidence == pytest.approx(0.9)


def test_estimator_picks_female_when_prob_higher(_model_dir) -> None:
    core = _make_fake_ov_core(age_scaled=0.28, prob_female=0.8, prob_male=0.2)
    with patch("gaze_analytics.inference.age_gender.ov.Core", return_value=core):
        est = AgeGenderEstimator(cfg=Settings(models_dir=_model_dir))
        result = est.estimate(np.zeros((80, 80, 3), dtype=np.uint8))
    assert result.gender == "F"
    assert result.age == pytest.approx(28.0)
    assert result.gender_confidence == pytest.approx(0.8)


def test_estimator_does_not_mutate_input_crop(_model_dir) -> None:
    core = _make_fake_ov_core(age_scaled=0.3, prob_female=0.5, prob_male=0.5)
    with patch("gaze_analytics.inference.age_gender.ov.Core", return_value=core):
        est = AgeGenderEstimator(cfg=Settings(models_dir=_model_dir))
        crop = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
        snap = crop.copy()
        est.estimate(crop)
    assert np.array_equal(crop, snap)
