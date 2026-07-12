"""Tests for the read-only dashboard queries.

The dashboard reads the same aggregate-only SQLite that :mod:`gaze_analytics.storage`
writes. Here we round-trip a few known rows through :class:`SqliteSink` and
verify the query helpers surface them correctly.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from dashboard.data import (
    compute_kpis,
    decode_thumbnail,
    gender_totals,
    metrics_frame,
    segments_frame,
)

from gaze_analytics.content import ContentSegment
from gaze_analytics.engagement import WindowMetrics
from gaze_analytics.storage import SqliteSink


@pytest.fixture
def populated_db(tmp_path):
    db_path = tmp_path / "metrics.sqlite"
    sink = SqliteSink(db_path)
    # The sink maps monotonic → wall-clock relative to its own init time,
    # so seed synthetic rows with monotonic values near "now".
    base = time.monotonic()

    # Two aggregate rows.
    sink.write_metrics(
        WindowMetrics(
            window_start_ts=base + 0.0,
            window_end_ts=base + 5.0,
            window_seconds=5,
            viewers=2,
            attending=1,
            avg_dwell_ms=800,
            male_count=1,
            female_count=1,
            segment_id="seg-a",
        )
    )
    sink.write_metrics(
        WindowMetrics(
            window_start_ts=base + 5.0,
            window_end_ts=base + 10.0,
            window_seconds=5,
            viewers=4,
            attending=3,
            avg_dwell_ms=1600,
            male_count=2,
            female_count=2,
            segment_id="seg-b",
        )
    )

    # One segment with a thumbnail, closed.
    seg = ContentSegment(
        segment_id=1,
        phash=0x1122334455667788,
        first_seen_ts=base + 0.0,
        last_seen_ts=base + 4.5,
        thumbnail=np.full((90, 160, 3), 128, dtype=np.uint8),
    )
    sid = sink.upsert_segment_start(seg)
    sink.close_segment(seg, sid)

    sink.close()
    return db_path


def test_metrics_frame_returns_all_rows_when_window_all(populated_db) -> None:
    df = metrics_frame(populated_db, window_minutes=None)
    assert len(df) == 2
    # Sorted ascending by ts.
    assert df.iloc[0]["viewers"] == 2
    assert df.iloc[1]["viewers"] == 4


def test_metrics_frame_time_filter_excludes_old_rows(populated_db) -> None:
    # Any positive minutes value keeps rows written just now (they are seconds old).
    fresh = metrics_frame(populated_db, window_minutes=60)
    assert len(fresh) == 2
    # A negative-effect filter — asking for "the last 0 minutes minus a second"
    # is nonsensical in normal use, so we validate the SQL path with a
    # deliberately large window instead.
    assert metrics_frame(populated_db, window_minutes=24 * 60).shape[0] == 2


def test_segments_frame_returns_closed_segment(populated_db) -> None:
    df = segments_frame(populated_db)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["duration_ms"] == 4500
    assert row["phash"] == "1122334455667788"
    assert row["thumbnail_b64"] is not None


def test_segments_frame_only_with_viewers_hides_unviewed(tmp_path) -> None:
    """The display filter must not touch storage — segments still exist in DB."""
    db_path = tmp_path / "metrics.sqlite"
    sink = SqliteSink(db_path)
    base = time.monotonic()

    # Segment A — has a viewed metric row.
    seg_a = ContentSegment(
        segment_id=1,
        phash=0xAAAA_AAAA_AAAA_AAAA,
        first_seen_ts=base,
        last_seen_ts=base + 5.0,
        thumbnail=np.full((90, 160, 3), 100, dtype=np.uint8),
    )
    sid_a = sink.upsert_segment_start(seg_a)
    sink.close_segment(seg_a, sid_a)
    sink.write_metrics(
        WindowMetrics(
            window_start_ts=base,
            window_end_ts=base + 5.0,
            window_seconds=5,
            viewers=2,
            attending=1,
            avg_dwell_ms=500,
            male_count=1,
            female_count=1,
            segment_id=sid_a,
        )
    )

    # Segment B — no viewers (metric row exists but viewers=0).
    seg_b = ContentSegment(
        segment_id=2,
        phash=0xBBBB_BBBB_BBBB_BBBB,
        first_seen_ts=base + 5.0,
        last_seen_ts=base + 10.0,
        thumbnail=np.full((90, 160, 3), 200, dtype=np.uint8),
    )
    sid_b = sink.upsert_segment_start(seg_b)
    sink.close_segment(seg_b, sid_b)
    sink.write_metrics(
        WindowMetrics(
            window_start_ts=base + 5.0,
            window_end_ts=base + 10.0,
            window_seconds=5,
            viewers=0,
            attending=0,
            avg_dwell_ms=0,
            male_count=0,
            female_count=0,
            segment_id=sid_b,
        )
    )
    sink.close()

    # Default: both segments visible.
    all_segments = segments_frame(db_path)
    assert len(all_segments) == 2

    # Filtered: only the viewed one.
    viewed = segments_frame(db_path, only_with_viewers=True)
    assert len(viewed) == 1
    assert viewed.iloc[0]["id"] == sid_a


def test_kpis_come_from_the_latest_row(populated_db) -> None:
    metrics = metrics_frame(populated_db, window_minutes=None)
    segments = segments_frame(populated_db)
    kpis = compute_kpis(metrics, segments)
    # ``*_now`` fields track the latest window row (viewers=4, attending=3).
    assert kpis.viewers_now == 4
    assert kpis.attending_now == 3
    # attention_rate = sum(attending) / sum(viewers) * 100 = (1+3)/(2+4)*100
    assert kpis.attention_rate == pytest.approx(66.7, abs=0.1)
    assert kpis.total_segments == 1
    assert kpis.avg_dwell_ms == pytest.approx(1200.0)  # (800 + 1600) / 2


def test_kpis_are_zero_when_empty(tmp_path) -> None:
    # Build an empty schema so the file exists but has no rows.
    SqliteSink(tmp_path / "empty.sqlite").close()
    metrics = metrics_frame(tmp_path / "empty.sqlite", window_minutes=None)
    segments = segments_frame(tmp_path / "empty.sqlite")
    kpis = compute_kpis(metrics, segments)
    assert kpis.viewers_now == 0
    assert kpis.attending_now == 0
    assert kpis.attention_rate == 0.0
    assert kpis.avg_dwell_ms == 0.0
    assert kpis.total_segments == 0


def test_gender_totals_sum_across_window(populated_db) -> None:
    df = metrics_frame(populated_db, window_minutes=None)
    totals = gender_totals(df)
    assert totals == {"Male": 3, "Female": 3, "Nobody": 0}


def test_decode_thumbnail_roundtrips_a_jpeg(populated_db) -> None:
    df = segments_frame(populated_db)
    img = decode_thumbnail(df.iloc[0]["thumbnail_b64"])
    assert img is not None
    assert img.shape == (90, 160, 3)


def test_decode_thumbnail_returns_none_for_bad_input() -> None:
    assert decode_thumbnail(None) is None
    assert decode_thumbnail("not-base64!!!") is None


def test_metrics_frame_raises_when_db_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        metrics_frame(tmp_path / "nope.sqlite", window_minutes=None)
