"""Tests for :mod:`gaze_analytics.privacy.render`.

These are the strictest tests in the suite: for a demo we absolutely must
guarantee that every non-``off`` mode has physically overwritten the face
pixels in the frame buffer. If any pixel inside the bbox still matches the
original, the mode has failed its contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from gaze_analytics.inference import FaceBBox
from gaze_analytics.privacy import apply_privacy_mask


def _distinctive_frame(seed: int = 42) -> np.ndarray:
    """Return a high-frequency frame that has no chance of matching a mask by luck."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)


def _face() -> FaceBBox:
    return FaceBBox(xmin=100, ymin=80, xmax=200, ymax=180, score=0.95)


def _region(frame: np.ndarray, bbox: FaceBBox) -> np.ndarray:
    return frame[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax]


def test_off_mode_leaves_frame_unchanged() -> None:
    frame = _distinctive_frame()
    snap = frame.copy()
    apply_privacy_mask(frame, [_face()], mode="off")
    assert np.array_equal(frame, snap)


def test_no_faces_is_a_no_op() -> None:
    frame = _distinctive_frame()
    snap = frame.copy()
    apply_privacy_mask(frame, [], mode="blur")
    assert np.array_equal(frame, snap)


def test_blur_mode_changes_every_pixel_in_face_region() -> None:
    frame = _distinctive_frame()
    face = _face()
    original_region = _region(frame, face).copy()
    apply_privacy_mask(frame, [face], mode="blur")
    new_region = _region(frame, face)
    # A heavy Gaussian blur must change virtually all pixels in the region.
    diff = (new_region != original_region).any(axis=2)
    changed_ratio = diff.mean()
    assert changed_ratio > 0.99, f"blur left {(1 - changed_ratio):.1%} of pixels intact"


def test_blur_mode_leaves_area_outside_bbox_untouched() -> None:
    frame = _distinctive_frame()
    face = _face()
    snap = frame.copy()
    apply_privacy_mask(frame, [face], mode="blur")
    # Everything above the bbox must be pixel-identical.
    assert np.array_equal(frame[: face.ymin], snap[: face.ymin])
    # Same for the strip below.
    assert np.array_equal(frame[face.ymax :], snap[face.ymax :])


def test_pixelate_mode_collapses_region_to_a_low_frequency_mosaic() -> None:
    frame = _distinctive_frame()
    face = _face()
    region_size = (face.ymax - face.ymin) * (face.xmax - face.xmin)
    apply_privacy_mask(frame, [face], mode="pixelate")
    region = _region(frame, face)
    # Pixelate downsamples to ~blocks x blocks then upsamples with NEAREST, so
    # the region contains at most ~blocks^2 distinct colours. A random 100x100
    # region normally has ~10,000 unique colours; if pixelate is working, this
    # must drop by orders of magnitude.
    flattened = region.reshape(-1, 3)
    unique_colours = np.unique(flattened, axis=0).shape[0]
    assert unique_colours <= 200, (
        f"pixelated region should collapse to <= 200 unique colours, got {unique_colours}"
    )
    assert unique_colours < region_size / 20, (
        "pixelate did not meaningfully reduce colour count "
        f"({unique_colours} vs region size {region_size})"
    )


def test_silhouette_mode_overwrites_region_with_solid_fill() -> None:
    frame = _distinctive_frame()
    face = _face()
    apply_privacy_mask(frame, [face], mode="silhouette")
    region = _region(frame, face)
    # After the outline is drawn, the interior should be almost entirely the
    # solid fill colour. Look at the centre pixel — it must be the fill.
    cy, cx = region.shape[0] // 2, region.shape[1] // 2
    assert tuple(region[cy, cx]) == (60, 60, 60)


def test_bbox_clipped_to_frame_bounds() -> None:
    frame = _distinctive_frame()
    # bbox that partially extends off the frame — must not raise.
    face = FaceBBox(xmin=-20, ymin=-30, xmax=50, ymax=60, score=0.9)
    apply_privacy_mask(frame, [face], mode="blur")


def test_multiple_faces_all_masked() -> None:
    frame = _distinctive_frame()
    a = FaceBBox(xmin=10, ymin=10, xmax=60, ymax=60, score=0.9)
    b = FaceBBox(xmin=200, ymin=150, xmax=260, ymax=200, score=0.9)
    orig_a = _region(frame, a).copy()
    orig_b = _region(frame, b).copy()
    apply_privacy_mask(frame, [a, b], mode="silhouette")
    assert not np.array_equal(_region(frame, a), orig_a)
    assert not np.array_equal(_region(frame, b), orig_b)


@pytest.mark.parametrize("mode", ["blur", "pixelate", "silhouette"])
def test_face_region_never_retains_original_pixels(mode: str) -> None:
    """The hard privacy invariant: no mode leaves the raw pixels visible."""
    frame = _distinctive_frame(seed=7)
    face = _face()
    original_region = _region(frame, face).copy()
    apply_privacy_mask(frame, [face], mode=mode)  # type: ignore[arg-type]
    new_region = _region(frame, face)
    # The number of pixels that are byte-for-byte identical must be tiny.
    identical = (new_region == original_region).all(axis=2).mean()
    assert identical < 0.02, (
        f"mode {mode} left {identical:.2%} of original pixels visible"
    )
