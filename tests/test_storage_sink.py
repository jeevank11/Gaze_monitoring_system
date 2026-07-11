"""Tests for :mod:`gaze_analytics.storage.sink`.

Uses a fresh SQLite DB in ``tmp_path`` for each test — never touches the
real ``data/metrics.sqlite``.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from gaze_analytics.content import ContentSegment
from gaze_analytics.engagement import WindowMetrics
from gaze_analytics.storage import SqliteSink


@pytest.fixture
def sink(tmp_path):
    s = SqliteSink(tmp_path / "test.sqlite")
    yield s
    s.close()


def _metrics() -> WindowMetrics:
    return WindowMetrics(
        window_start_ts=0.0,
        window_end_ts=5.0,
        window_seconds=5,
        viewers=3,
        attending=2,
        avg_dwell_ms=1200,
        male_count=1,
        female_count=2,
        segment_id="seg-1",
    )


def test_schema_creates_metrics_and_segments_tables(tmp_path) -> None:
    SqliteSink(tmp_path / "s.sqlite").close()
    conn = sqlite3.connect(str(tmp_path / "s.sqlite"))
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "metrics" in names
    assert "segments" in names
    conn.close()


def test_write_metrics_persists_all_aggregate_fields(sink) -> None:
    sink.write_metrics(_metrics())
    conn = sqlite3.connect(str(sink._path))
    row = conn.execute(
        "SELECT window_seconds, viewers, attending, avg_dwell_ms, "
        "male_count, female_count, segment_id FROM metrics"
    ).fetchone()
    conn.close()
    assert row == (5, 3, 2, 1200, 1, 2, "seg-1")


def test_metrics_table_has_no_identity_columns(sink) -> None:
    """The schema must never grow a per-person / bbox / image column."""
    conn = sqlite3.connect(str(sink._path))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(metrics)")}
    conn.close()
    banned = {"person_id", "track_id", "face_id", "bbox", "image", "embedding"}
    assert cols.isdisjoint(banned), f"forbidden columns present: {cols & banned}"


def test_upsert_and_close_segment(sink) -> None:
    seg = ContentSegment(
        segment_id=7,
        phash=0xDEADBEEFCAFEBABE,
        first_seen_ts=0.0,
        last_seen_ts=3.5,
        thumbnail=np.zeros((90, 160, 3), dtype=np.uint8),
    )
    sid = sink.upsert_segment_start(seg)
    # Second call with the same seg is idempotent (INSERT OR IGNORE).
    sink.upsert_segment_start(seg)

    sink.close_segment(seg, sid)

    conn = sqlite3.connect(str(sink._path))
    rows = conn.execute(
        "SELECT id, phash, duration_ms, thumbnail_b64 IS NOT NULL FROM segments"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    row_id, phash_hex, duration_ms, has_thumb = rows[0]
    assert row_id == sid
    assert phash_hex == f"{seg.phash:016x}"
    assert duration_ms == 3500
    assert has_thumb == 1
