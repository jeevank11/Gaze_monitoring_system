"""Read-only helpers that query the aggregate-only SQLite database.

Kept in a plain module (not inside the Streamlit page) so the queries can be
unit-tested and reused from any UI. The dashboard imports these — it never
runs raw SQL on its own.
"""

from __future__ import annotations

import base64
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Kpis:
    """Snapshot of the top-line numbers shown in the header tiles."""

    viewers_now: int
    attending_now: int
    avg_dwell_ms: float
    total_segments: int
    total_rows: int


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"metrics database not found at {db_path}. "
            "Run `python -m gaze_analytics run --sink` to populate it."
        )
    return sqlite3.connect(str(db_path))


def metrics_frame(db_path: Path, window_minutes: int | None = 60) -> pd.DataFrame:
    """Return recent aggregate rows, most-recent last (chart-friendly)."""
    with closing(_connect(db_path)) as conn:
        query = "SELECT ts, window_seconds, viewers, attending, avg_dwell_ms, " \
                "male_count, female_count, segment_id FROM metrics"
        params: tuple = ()
        if window_minutes is not None:
            cutoff_dt = datetime.now(tz=UTC) - timedelta(minutes=window_minutes)
            cutoff = (
                cutoff_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            )
            query += " WHERE ts >= ?"
            params = (cutoff,)
        query += " ORDER BY ts ASC"
        df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def segments_frame(db_path: Path, limit: int = 20) -> pd.DataFrame:
    """Return the ``limit`` most recent content segments."""
    with closing(_connect(db_path)) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, started_at, ended_at, duration_ms, phash,
                   thumbnail_b64, segment_type
            FROM segments
            ORDER BY started_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    if not df.empty:
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
        df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True, errors="coerce")
    return df


def compute_kpis(metrics: pd.DataFrame, segments: pd.DataFrame) -> Kpis:
    """Derive the four header tiles from the two frames."""
    if metrics.empty:
        return Kpis(0, 0, 0.0, len(segments), 0)
    latest = metrics.iloc[-1]
    return Kpis(
        viewers_now=int(latest["viewers"]),
        attending_now=int(latest["attending"]),
        avg_dwell_ms=float(metrics["avg_dwell_ms"].tail(60).mean()),
        total_segments=len(segments),
        total_rows=len(metrics),
    )


def gender_totals(metrics: pd.DataFrame) -> dict[str, int]:
    """Aggregate gender counts across the visible window (privacy-safe totals)."""
    if metrics.empty:
        return {"M": 0, "F": 0}
    return {
        "M": int(metrics["male_count"].sum()),
        "F": int(metrics["female_count"].sum()),
    }


def decode_thumbnail(thumbnail_b64: str | None) -> np.ndarray | None:
    """Decode a base64 JPEG thumbnail into a numpy RGB array for ``st.image``."""
    if thumbnail_b64 is None:
        return None
    try:
        buf = base64.b64decode(thumbnail_b64.encode("ascii"))
    except (ValueError, TypeError):
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
