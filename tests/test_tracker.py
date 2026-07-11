"""Tests for the IoU + Kalman face tracker.

The tracker only sees integer bboxes — no image data required. We do use
real filterpy internally (it's a pure-Python library, cheap enough for CI).
"""

from __future__ import annotations

import pytest

from gaze_analytics.inference.face import FaceBBox
from gaze_analytics.tracker import IoUTracker
from gaze_analytics.tracker.kalman_box import KalmanBoxTracker


@pytest.fixture(autouse=True)
def _reset_track_ids() -> None:
    KalmanBoxTracker._reset_ids_for_tests()


def _bbox(x1: int, y1: int, x2: int, y2: int, score: float = 0.9) -> FaceBBox:
    return FaceBBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2, score=score)


def test_single_face_keeps_stable_id_across_frames() -> None:
    tracker = IoUTracker(iou_threshold=0.2, max_misses=5, min_hits=1)
    ts = 0.0
    ids: list[int] = []
    # A face that shifts a few pixels each frame — still ~unity IoU with prior.
    for step in range(6):
        det = _bbox(100 + step, 100 + step, 200 + step, 200 + step)
        tracks = tracker.update([det], timestamp=ts + step * 0.1)
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1, f"track ID should be stable, got {ids}"


def test_new_id_for_disjoint_detection() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_misses=5, min_hits=1)
    a = tracker.update([_bbox(0, 0, 100, 100)], timestamp=0.0)
    # A completely disjoint box — must get a new track id.
    b = tracker.update(
        [_bbox(0, 0, 100, 100), _bbox(500, 500, 600, 600)],
        timestamp=0.1,
    )
    assert len(a) == 1
    assert len(b) == 2
    ids_first = {t.track_id for t in a}
    ids_second = {t.track_id for t in b}
    assert ids_first.issubset(ids_second)
    assert len(ids_second - ids_first) == 1


def test_track_dies_after_max_misses() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_misses=3, min_hits=1)
    tracker.update([_bbox(0, 0, 100, 100)], timestamp=0.0)
    # Miss for max_misses + 1 frames.
    for step in range(1, 6):
        tracks = tracker.update([], timestamp=0.1 * step)
    # After ≥ max_misses+1 misses, the track should no longer be emitted.
    assert tracks == []


def test_dwell_seconds_computed_from_first_to_last_seen() -> None:
    tracker = IoUTracker(iou_threshold=0.2, max_misses=5, min_hits=1)
    ts_start = 100.0
    tracker.update([_bbox(0, 0, 100, 100)], timestamp=ts_start)
    tracks = tracker.update([_bbox(1, 1, 101, 101)], timestamp=ts_start + 1.5)
    assert len(tracks) == 1
    assert tracks[0].dwell_seconds == pytest.approx(1.5, rel=1e-3)


def test_tracker_never_stores_pixels() -> None:
    """The tracker must not accept or expose any image data."""
    tracker = IoUTracker()
    tracker.update([_bbox(0, 0, 100, 100)], timestamp=0.0)
    for trk in tracker._trackers:
        state_attrs = set(vars(trk))
        for banned in ("pixels", "image", "crop", "embedding", "descriptor"):
            assert banned not in state_attrs, f"tracker leaked {banned}"
