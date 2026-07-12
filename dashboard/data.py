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
    attention_rate: float  # percentage 0-100
    avg_dwell_ms: float
    total_segments: int


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
        # Convert to local timezone so dashboard charts match the user's clock
        local_tz = datetime.now().astimezone().tzinfo
        df["ts"] = df["ts"].dt.tz_convert(local_tz)
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
        local_tz = datetime.now().astimezone().tzinfo
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
        df["started_at"] = df["started_at"].dt.tz_convert(local_tz)
        df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True, errors="coerce")
        df["ended_at"] = df["ended_at"].dt.tz_convert(local_tz)
    return df


def compute_kpis(metrics: pd.DataFrame, segments: pd.DataFrame) -> Kpis:
    """Derive the header tiles from the metrics in the selected window."""
    if metrics.empty:
        return Kpis(0, 0, 0.0, 0.0, len(segments))
    latest = metrics.iloc[-1]
    total_viewers = int(metrics["viewers"].sum())
    total_attending = int(metrics["attending"].sum())
    attention_rate = (total_attending / total_viewers * 100) if total_viewers > 0 else 0.0
    return Kpis(
        viewers_now=int(latest["viewers"]),
        attending_now=int(latest["attending"]),
        attention_rate=round(attention_rate, 1),
        avg_dwell_ms=float(metrics["avg_dwell_ms"].mean()),
        total_segments=len(segments),
    )


def gender_totals(metrics: pd.DataFrame) -> dict[str, int]:
    """Aggregate gender counts across the visible window (privacy-safe totals).

    Windows where no viewers were detected are counted as 'Nobody' rather
    than being silently ignored — this avoids inflating Male/Female ratios.
    """
    if metrics.empty:
        return {"Male": 0, "Female": 0, "Nobody": 0}
    nobody = int((metrics["viewers"] == 0).sum())
    return {
        "Male": int(metrics["male_count"].sum()),
        "Female": int(metrics["female_count"].sum()),
        "Nobody": nobody,
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


def content_effectiveness(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute per-segment engagement scores.

    For each segment_id, computes:
    - total_windows: number of aggregate windows during the segment
    - windows_with_viewers: windows where at least 1 viewer was present
    - total_attending: sum of attending counts
    - total_viewers: sum of viewer counts
    - attention_rate: attending / viewers (0-100%), the key effectiveness metric
    - avg_dwell_ms: average dwell time across the segment

    Returns a DataFrame sorted by attention_rate descending (best content first).
    """
    if metrics.empty or "segment_id" not in metrics.columns:
        return pd.DataFrame(
            columns=[
                "segment_id", "total_windows", "windows_with_viewers",
                "total_attending", "total_viewers", "attention_rate", "avg_dwell_ms",
            ]
        )

    # Drop rows with no segment_id
    with_seg = metrics[metrics["segment_id"].notna()].copy()
    if with_seg.empty:
        return pd.DataFrame(
            columns=[
                "segment_id", "total_windows", "windows_with_viewers",
                "total_attending", "total_viewers", "attention_rate", "avg_dwell_ms",
            ]
        )

    grouped = with_seg.groupby("segment_id").agg(
        total_windows=("viewers", "count"),
        windows_with_viewers=("viewers", lambda x: (x > 0).sum()),
        total_attending=("attending", "sum"),
        total_viewers=("viewers", "sum"),
        avg_dwell_ms=("avg_dwell_ms", "mean"),
    ).reset_index()

    grouped["attention_rate"] = (
        grouped["total_attending"] / grouped["total_viewers"].replace(0, float("nan")) * 100
    ).fillna(0).round(1)

    return grouped.sort_values("attention_rate", ascending=False).reset_index(drop=True)
