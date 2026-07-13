"""Tests for the business-insights aggregate queries."""

from __future__ import annotations

import time

import numpy as np
import pytest
from dashboard.business import (
    all_segments,
    bottom_by_attention,
    classify_quadrants,
    gaze_kpis,
    insights_kpis,
    top_by_attention_time,
    top_by_impressions,
)

from gaze_analytics.content import ContentSegment
from gaze_analytics.engagement import WindowMetrics
from gaze_analytics.storage import SqliteSink


def _seed_segment(
    sink: SqliteSink,
    seg_num: int,
    phash: int,
    base: float,
    offset: float,
) -> str:
    seg = ContentSegment(
        segment_id=seg_num,
        phash=phash,
        first_seen_ts=base + offset,
        last_seen_ts=base + offset + 5.0,
        thumbnail=np.full((90, 160, 3), 100 + seg_num, dtype=np.uint8),
    )
    sid = sink.upsert_segment_start(seg)
    sink.close_segment(seg, sid)
    return sid


@pytest.fixture
def leaderboard_db(tmp_path):
    """Three segments with intentionally distinct reach / attention profiles."""
    db_path = tmp_path / "metrics.sqlite"
    sink = SqliteSink(db_path)
    base = time.monotonic()

    # Segment A: high reach (viewers 5x4=20), medium attention (2x4=8 -> 40%).
    sid_a = _seed_segment(sink, 1, 0xA1A1_A1A1_A1A1_A1A1, base, 0.0)
    for i in range(4):
        sink.write_metrics(
            WindowMetrics(
                window_start_ts=base + i * 5.0,
                window_end_ts=base + (i + 1) * 5.0,
                window_seconds=5,
                viewers=5,
                attending=2,
                avg_dwell_ms=500,
                male_count=2,
                female_count=3,
                segment_id=sid_a,
            )
        )

    # Segment B: medium reach (viewers 4+4=8), very low attention (0+1=1 -> 12.5%).
    sid_b = _seed_segment(sink, 2, 0xB2B2_B2B2_B2B2_B2B2, base, 20.0)
    for i in range(2):
        sink.write_metrics(
            WindowMetrics(
                window_start_ts=base + 20.0 + i * 5.0,
                window_end_ts=base + 25.0 + i * 5.0,
                window_seconds=5,
                viewers=4,
                attending=0 if i == 0 else 1,
                avg_dwell_ms=200,
                male_count=2,
                female_count=2,
                segment_id=sid_b,
            )
        )

    # Segment C: low reach (viewers=3), perfect attention (100%). Should be
    # excluded from "Least Attention" by the min_viewers guard.
    sid_c = _seed_segment(sink, 3, 0xC3C3_C3C3_C3C3_C3C3, base, 40.0)
    sink.write_metrics(
        WindowMetrics(
            window_start_ts=base + 40.0,
            window_end_ts=base + 45.0,
            window_seconds=5,
            viewers=3,
            attending=3,
            avg_dwell_ms=1000,
            male_count=1,
            female_count=2,
            segment_id=sid_c,
        )
    )

    sink.close()
    return db_path, sid_a, sid_b, sid_c


def test_top_by_impressions_ranks_by_total_viewers(leaderboard_db) -> None:
    db_path, sid_a, sid_b, sid_c = leaderboard_db
    df = top_by_impressions(db_path, window_minutes=None, limit=5)
    assert list(df["segment_id"]) == [sid_a, sid_b, sid_c]
    assert int(df.iloc[0]["impressions"]) == 20  # 4 windows x 5 viewers
    assert df.iloc[0]["attention_pct"] == pytest.approx(40.0)  # 8 / 20


def test_bottom_by_attention_puts_worst_first(leaderboard_db) -> None:
    db_path, sid_a, sid_b, _sid_c = leaderboard_db
    df = bottom_by_attention(db_path, window_minutes=None, limit=5, min_viewers=5)
    # C excluded (only 3 viewers). B (12.5%) beats A (40%) as "worst".
    assert list(df["segment_id"]) == [sid_b, sid_a]
    assert df.iloc[0]["attention_pct"] == pytest.approx(12.5)


def test_bottom_by_attention_respects_min_viewers_guard(leaderboard_db) -> None:
    db_path, sid_a, _sid_b, _sid_c = leaderboard_db
    # Only A clears a tight threshold — B and C are dropped.
    df = bottom_by_attention(db_path, window_minutes=None, limit=5, min_viewers=15)
    assert list(df["segment_id"]) == [sid_a]


def test_top_by_attention_time_ranks_by_attending_seconds(leaderboard_db) -> None:
    db_path, sid_a, sid_b, sid_c = leaderboard_db
    # attention_seconds = SUM(attending * window_seconds)
    #   A: 4 windows * 2 attending * 5s = 40
    #   B: (0 + 1) attending * 5s       =  5
    #   C: 1 window  * 3 attending * 5s = 15
    # Ranking: A > C > B
    df = top_by_attention_time(db_path, window_minutes=None, limit=5)
    assert list(df["segment_id"]) == [sid_a, sid_c, sid_b]
    assert int(df.iloc[0]["attention_seconds"]) == 40
    assert int(df.iloc[1]["attention_seconds"]) == 15
    assert int(df.iloc[2]["attention_seconds"]) == 5


def test_empty_db_returns_empty_frames(tmp_path) -> None:
    db_path = tmp_path / "empty.sqlite"
    SqliteSink(db_path).close()
    assert top_by_impressions(db_path, window_minutes=None).empty
    assert bottom_by_attention(db_path, window_minutes=None).empty
    assert top_by_attention_time(db_path, window_minutes=None).empty


def test_missing_db_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        top_by_impressions(tmp_path / "nope.sqlite")
    with pytest.raises(FileNotFoundError):
        bottom_by_attention(tmp_path / "nope.sqlite")
    with pytest.raises(FileNotFoundError):
        top_by_attention_time(tmp_path / "nope.sqlite")


# ---- New helpers (KPI strip + scatter + quadrants) ----------------------


def test_all_segments_returns_every_segment_meeting_threshold(leaderboard_db) -> None:
    db_path, sid_a, sid_b, sid_c = leaderboard_db
    # min_viewers=1 keeps all three; ordering is impressions DESC.
    df = all_segments(db_path, window_minutes=None, min_viewers=1)
    assert list(df["segment_id"]) == [sid_a, sid_b, sid_c]
    assert set(df.columns) >= {
        "segment_id", "impressions", "attention_pct",
        "duration_ms", "thumbnail_b64", "started_at",
    }


def test_classify_quadrants_labels_relative_to_median(leaderboard_db) -> None:
    db_path, sid_a, sid_b, sid_c = leaderboard_db
    df = classify_quadrants(all_segments(db_path, window_minutes=None))
    labels = dict(zip(df["segment_id"], df["quadrant"], strict=True))
    # A: high impressions (20), medium attn (40%) — above median on impr.
    # B: medium impressions (8), low attn (12.5%).
    # C: low impressions (3), high attn (100%).
    # Medians: impr=8, attn=40. Classification is ``>= median`` for high.
    assert labels[sid_a] == "Winner"   # imp>=median, attn>=median
    assert labels[sid_b] == "Kill"     # imp>=median (=8), attn<median (12.5)
    assert labels[sid_c] == "Niche"    # imp<median, attn>=median


def test_classify_quadrants_handles_empty_frame() -> None:
    import pandas as pd

    empty_df = pd.DataFrame(
        columns=["segment_id", "impressions", "attention_pct", "duration_ms"]
    )
    out = classify_quadrants(empty_df)
    assert out.empty
    assert "quadrant" in out.columns


def test_insights_kpis_summarises_the_current_window(leaderboard_db) -> None:
    db_path, *_ = leaderboard_db
    kpis = insights_kpis(db_path, window_minutes=None)
    # Sums across all metric rows: viewers = 20 + 8 + 3 = 31; attending = 8 + 1 + 3 = 12.
    assert kpis.impressions == 31
    assert kpis.engaged_impressions == 12
    assert kpis.attention_pct == pytest.approx(round(100 * 12 / 31, 1))
    assert kpis.unique_segments == 3
    # No prior window available for "All time" — deltas should be None.
    assert kpis.impressions_delta is None
    assert kpis.attention_delta_pp is None


def test_insights_kpis_computes_prior_window_delta(leaderboard_db) -> None:
    db_path, *_ = leaderboard_db
    # Any positive window keeps the just-written rows in "current" and leaves
    # the prior window empty, so deltas equal the current values.
    kpis = insights_kpis(db_path, window_minutes=60)
    assert kpis.impressions_delta == kpis.impressions
    assert kpis.engaged_delta == kpis.engaged_impressions
    assert kpis.segments_delta == kpis.unique_segments


def test_insights_kpis_empty_db_is_all_zeros(tmp_path) -> None:
    db_path = tmp_path / "empty.sqlite"
    SqliteSink(db_path).close()
    kpis = insights_kpis(db_path, window_minutes=60)
    assert kpis.impressions == 0
    assert kpis.attention_pct == 0.0
    assert kpis.impressions_delta == 0  # prior window also empty


def test_insights_kpis_missing_db_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        insights_kpis(tmp_path / "nope.sqlite")


# ---- Gaze Insights KPIs -------------------------------------------------


def test_gaze_kpis_summarises_audience_totals(leaderboard_db) -> None:
    db_path, *_ = leaderboard_db
    kpis = gaze_kpis(db_path, window_minutes=None)
    # Totals across the seven metric rows the fixture writes:
    # viewers  = 4x5 + 2x4 + 1x3 = 31
    # attending = 4x2 + 0 + 1 + 3 = 12
    # male     = 4x2 + 2x2 + 1 = 13
    # female   = 4x3 + 2x2 + 2 = 18
    # peak     = max(5, 4, 3) = 5
    # attention_time = SUM(attending * 5s) = 12 * 5 = 60 seconds
    assert kpis.total_viewers == 31
    assert kpis.peak_concurrent == 5
    assert kpis.male_count == 13
    assert kpis.female_count == 18
    assert kpis.attention_pct == pytest.approx(round(100 * 12 / 31, 1))
    assert kpis.attention_time_seconds == 60
    # "All time" window has no prior period, so all deltas are None.
    assert kpis.total_viewers_delta is None
    assert kpis.peak_concurrent_delta is None
    assert kpis.attention_delta_pp is None
    assert kpis.dwell_delta_ms is None
    assert kpis.attention_time_delta_s is None


def test_gaze_kpis_reports_deltas_when_window_bounded(leaderboard_db) -> None:
    db_path, *_ = leaderboard_db
    # Any positive window keeps the just-written rows in "current" and leaves
    # the prior window empty, so deltas equal the current totals.
    kpis = gaze_kpis(db_path, window_minutes=60)
    assert kpis.total_viewers_delta == kpis.total_viewers
    assert kpis.peak_concurrent_delta == kpis.peak_concurrent
    assert kpis.attention_time_delta_s == kpis.attention_time_seconds


def test_gaze_kpis_empty_db_is_all_zeros(tmp_path) -> None:
    db_path = tmp_path / "empty.sqlite"
    SqliteSink(db_path).close()
    kpis = gaze_kpis(db_path, window_minutes=60)
    assert kpis.total_viewers == 0
    assert kpis.peak_concurrent == 0
    assert kpis.male_count == 0
    assert kpis.female_count == 0
    assert kpis.attention_pct == 0.0
    assert kpis.attention_time_seconds == 0
    # Prior window is also empty, so deltas resolve to 0 (not None).
    assert kpis.total_viewers_delta == 0
    assert kpis.peak_concurrent_delta == 0
    assert kpis.attention_time_delta_s == 0


def test_gaze_kpis_missing_db_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        gaze_kpis(tmp_path / "nope.sqlite")
