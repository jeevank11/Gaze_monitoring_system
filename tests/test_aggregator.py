"""Tests for the rolling-window aggregator."""

from __future__ import annotations

from gaze_analytics.config import Settings
from gaze_analytics.engagement import RollingAggregator, WindowMetrics
from gaze_analytics.inference.face import FaceBBox
from gaze_analytics.tracker import TrackedFace


def _track(track_id: int, dwell_seconds: float = 0.0) -> TrackedFace:
    return TrackedFace(
        track_id=track_id,
        bbox=FaceBBox(xmin=0, ymin=0, xmax=100, ymax=100, score=0.9),
        hits=1,
        age=1,
        first_seen_ts=0.0,
        last_seen_ts=dwell_seconds,
    )


def test_flush_only_emits_after_window_seconds() -> None:
    cfg = Settings(aggregate_window_seconds=5)
    agg = RollingAggregator(cfg=cfg)
    agg.record([_track(1)], attending_ids=set(), gender_by_id={}, timestamp=0.0)
    # 4.9s later — still inside window.
    assert agg.flush_if_due(timestamp=4.9) is None
    # 5.0s later — window closes.
    row = agg.flush_if_due(timestamp=5.0)
    assert isinstance(row, WindowMetrics)
    assert row.window_seconds == 5


def test_max_viewers_and_attending_are_maxima_over_window() -> None:
    cfg = Settings(aggregate_window_seconds=2)
    agg = RollingAggregator(cfg=cfg)
    agg.record([_track(1)], {1}, {}, timestamp=0.0)
    agg.record([_track(1), _track(2)], {1, 2}, {}, timestamp=0.5)
    agg.record([_track(1)], {1}, {}, timestamp=1.0)
    row = agg.flush_if_due(timestamp=2.0)
    assert row is not None
    assert row.viewers == 2
    assert row.attending == 2


def test_avg_dwell_ms_uses_per_track_max_dwell() -> None:
    cfg = Settings(aggregate_window_seconds=1)
    agg = RollingAggregator(cfg=cfg)
    agg.record([_track(1, dwell_seconds=0.5), _track(2, dwell_seconds=1.5)], set(), {}, 0.0)
    row = agg.flush_if_due(timestamp=1.0)
    assert row is not None
    # (500 + 1500) / 2 = 1000
    assert row.avg_dwell_ms == 1000


def test_gender_counts_use_last_observed_label_per_track() -> None:
    cfg = Settings(aggregate_window_seconds=1)
    agg = RollingAggregator(cfg=cfg)
    agg.record([_track(1), _track(2)], set(), {1: "M", 2: "F"}, 0.0)
    # Re-observe track 1 as F — last write wins for that track only.
    agg.record([_track(1), _track(2)], set(), {1: "F"}, 0.5)
    row = agg.flush_if_due(timestamp=1.0)
    assert row is not None
    assert row.male_count == 0
    assert row.female_count == 2


def test_flush_resets_state_between_windows() -> None:
    cfg = Settings(aggregate_window_seconds=1)
    agg = RollingAggregator(cfg=cfg)
    agg.record([_track(1), _track(2)], {1, 2}, {}, 0.0)
    first = agg.flush_if_due(timestamp=1.0)
    assert first is not None and first.viewers == 2

    # New window starts on the next record; the old maxima must not carry over.
    agg.record([_track(3)], set(), {}, 1.0)
    second = agg.flush_if_due(timestamp=2.0)
    assert second is not None
    assert second.viewers == 1
    assert second.attending == 0


def test_flush_before_any_record_returns_none() -> None:
    cfg = Settings(aggregate_window_seconds=1)
    agg = RollingAggregator(cfg=cfg)
    assert agg.flush_if_due(timestamp=5.0) is None
    assert agg.flush(timestamp=5.0) is None


def test_segment_id_is_passed_through() -> None:
    cfg = Settings(aggregate_window_seconds=1)
    agg = RollingAggregator(cfg=cfg)
    agg.record([_track(1)], set(), {}, 0.0)
    row = agg.flush_if_due(timestamp=1.0, segment_id="seg-42")
    assert row is not None and row.segment_id == "seg-42"
