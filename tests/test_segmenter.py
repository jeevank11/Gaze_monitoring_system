"""Tests for the pHash-based content segmenter."""

from __future__ import annotations

import numpy as np

from gaze_analytics.config import Settings
from gaze_analytics.content.segmenter import ContentSegmenter


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (240, 320)) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _gradient(seed: int, size: tuple[int, int] = (240, 320)) -> np.ndarray:
    h, w = size
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return np.stack([base, base, base], axis=-1)


def test_first_frame_creates_segment_zero() -> None:
    seg = ContentSegmenter(cfg=Settings()).update(_solid((10, 20, 30)), timestamp=0.0)
    assert seg.segment_id == 0
    assert seg.first_seen_ts == 0.0
    assert seg.last_seen_ts == 0.0


def test_same_content_keeps_same_segment_and_extends_last_seen() -> None:
    segmenter = ContentSegmenter(cfg=Settings())
    img = _gradient(seed=1)
    a = segmenter.update(img, timestamp=0.0)
    b = segmenter.update(img.copy(), timestamp=1.0)
    c = segmenter.update(img.copy(), timestamp=2.5)
    assert a.segment_id == b.segment_id == c.segment_id == 0
    assert c.first_seen_ts == 0.0
    assert c.last_seen_ts == 2.5
    assert c.duration_seconds == 2.5


def test_content_change_creates_new_segment() -> None:
    segmenter = ContentSegmenter(cfg=Settings())
    first = segmenter.update(_gradient(seed=1), timestamp=0.0)
    second = segmenter.update(_gradient(seed=999), timestamp=1.0)
    assert first.segment_id == 0
    assert second.segment_id == 1
    assert second.first_seen_ts == 1.0


def test_thumbnail_shape_matches_config_when_enabled() -> None:
    cfg = Settings(keep_segment_thumbnail=True, thumbnail_size=(160, 90))
    seg = ContentSegmenter(cfg=cfg).update(_gradient(seed=7), timestamp=0.0)
    assert seg.thumbnail is not None
    # cv2.resize takes (w, h); the returned array is (h, w, c).
    assert seg.thumbnail.shape == (90, 160, 3)


def test_thumbnail_omitted_when_disabled() -> None:
    cfg = Settings(keep_segment_thumbnail=False)
    seg = ContentSegmenter(cfg=cfg).update(_gradient(seed=7), timestamp=0.0)
    assert seg.thumbnail is None
