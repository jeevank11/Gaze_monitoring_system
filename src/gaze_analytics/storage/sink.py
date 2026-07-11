"""SQLite sink for aggregate metrics — sprint 5.

The schema (:file:`schema.sql`) is intentionally aggregate-only: counters,
timestamps, and pHash fingerprints. There is no column for identity, bbox,
or image data, and any migration adding one MUST be rejected.
"""

from __future__ import annotations

import base64
import logging
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from gaze_analytics.content import ContentSegment
from gaze_analytics.engagement import WindowMetrics

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _iso_utc(ts_monotonic: float, ref_monotonic: float, ref_wall: datetime) -> str:
    """Map a ``time.monotonic()`` value to a wall-clock ISO-8601 UTC string."""
    delta = ts_monotonic - ref_monotonic
    wall = ref_wall + timedelta(seconds=delta)
    return wall.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _thumbnail_to_b64(thumbnail: np.ndarray | None) -> str | None:
    if thumbnail is None:
        return None
    ok, buf = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


class SqliteSink:
    """Aggregate-only SQLite writer. One row per closed window; one row per segment."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        self._conn.commit()
        # Reference point for translating monotonic clock to wall clock.
        self._ref_monotonic = time.monotonic()
        self._ref_wall = datetime.now(tz=UTC)
        log.info("SQLite sink ready at %s", db_path)

    # -- public API -----------------------------------------------------------

    def write_metrics(self, metrics: WindowMetrics) -> None:
        """Insert one aggregate row. Called on every window flush."""
        ts_iso = _iso_utc(metrics.window_end_ts, self._ref_monotonic, self._ref_wall)
        self._conn.execute(
            """
            INSERT INTO metrics (
                ts, window_seconds, viewers, attending, avg_dwell_ms,
                male_count, female_count, segment_id, segment_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_iso,
                metrics.window_seconds,
                metrics.viewers,
                metrics.attending,
                metrics.avg_dwell_ms,
                metrics.male_count,
                metrics.female_count,
                metrics.segment_id,
                None,
            ),
        )
        self._conn.commit()

    def upsert_segment_start(self, segment: ContentSegment) -> str:
        """Insert (or ignore) a new segment row. Returns the wall-clock segment id."""
        segment_id = self._segment_id(segment)
        started_iso = _iso_utc(segment.first_seen_ts, self._ref_monotonic, self._ref_wall)
        thumb_b64 = _thumbnail_to_b64(segment.thumbnail)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO segments (
                id, started_at, ended_at, duration_ms,
                phash, thumbnail_b64, segment_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                started_iso,
                None,
                None,
                f"{segment.phash:016x}",
                thumb_b64,
                None,
            ),
        )
        self._conn.commit()
        return segment_id

    def close_segment(self, segment: ContentSegment, segment_id: str) -> None:
        """Fill in ``ended_at`` and ``duration_ms`` when a segment finishes."""
        ended_iso = _iso_utc(segment.last_seen_ts, self._ref_monotonic, self._ref_wall)
        duration_ms = int(segment.duration_seconds * 1000)
        self._conn.execute(
            "UPDATE segments SET ended_at = ?, duration_ms = ? WHERE id = ?",
            (ended_iso, duration_ms, segment_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- internals ------------------------------------------------------------

    def _segment_id(self, segment: ContentSegment) -> str:
        started_iso = _iso_utc(segment.first_seen_ts, self._ref_monotonic, self._ref_wall)
        # 'auto_2026-07-10T11-03-24.123Z' style; strip colons for filesystems.
        safe = started_iso.replace(":", "-")
        return f"auto_{safe}_{segment.segment_id:04d}"
