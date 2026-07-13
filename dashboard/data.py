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
    peak_viewers: int  # max viewers observed in any single window (current run / selection)
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


def _latest_run_start(db_path: Path) -> str | None:
    """Return the started_at timestamp of the most recent pipeline run, or None."""
    with closing(_connect(db_path)) as conn:
        try:
            row = conn.execute(
                "SELECT started_at FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None  # runs table doesn't exist yet


def metrics_frame(
    db_path: Path,
    window_minutes: int | None = 60,
    current_run_only: bool = False,
) -> pd.DataFrame:
    """Return recent aggregate rows, most-recent last (chart-friendly)."""
    with closing(_connect(db_path)) as conn:
        query = "SELECT ts, window_seconds, viewers, attending, avg_dwell_ms, " \
                "male_count, female_count, segment_id FROM metrics"
        params: tuple = ()
        if current_run_only:
            run_start = _latest_run_start(db_path)
            if run_start:
                query += " WHERE ts >= ?"
                params = (run_start,)
        elif window_minutes is not None:
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


def segments_frame(
    db_path: Path,
    limit: int = 20,
    only_with_viewers: bool = False,
) -> pd.DataFrame:
    """Return the ``limit`` most recent content segments.

    When ``only_with_viewers`` is True, filter to segments that overlap at
    least one aggregate window with ``viewers > 0``. The underlying tables
    stay unchanged — this is a presentation filter only.
    """
    with closing(_connect(db_path)) as conn:
        if only_with_viewers:
            query = """
                SELECT id, started_at, ended_at, duration_ms, phash,
                       thumbnail_b64, segment_type
                FROM segments
                WHERE id IN (
                    SELECT DISTINCT segment_id FROM metrics
                    WHERE viewers > 0 AND segment_id IS NOT NULL
                )
                ORDER BY started_at DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT id, started_at, ended_at, duration_ms, phash,
                       thumbnail_b64, segment_type
                FROM segments
                ORDER BY started_at DESC
                LIMIT ?
            """
        df = pd.read_sql_query(query, conn, params=(limit,))
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
        return Kpis(0, 0, 0, 0.0, 0.0, len(segments))
    latest = metrics.iloc[-1]
    total_viewers = int(metrics["viewers"].sum())
    total_attending = int(metrics["attending"].sum())
    attention_rate = (total_attending / total_viewers * 100) if total_viewers > 0 else 0.0
    return Kpis(
        viewers_now=int(latest["viewers"]),
        attending_now=int(latest["attending"]),
        peak_viewers=int(metrics["viewers"].max()),
        attention_rate=round(attention_rate, 1),
        avg_dwell_ms=float(metrics["avg_dwell_ms"].mean()),
        total_segments=len(segments),
    )


def gender_totals(metrics: pd.DataFrame) -> dict[str, int]:
    """Cumulative Male/Female counts across the selected window.

    Each row contributes the number of *tracks* whose majority-voted gender was
    Male / Female during that aggregate window. Windows with no viewers are
    ignored — they belong to occupancy, not demographics. Empty-room padding
    used to be reported as ``Nobody`` but it dominated the pie and made the
    Male/Female split look static, so it was removed. Use ``latest_gender``
    for a real-time snapshot.
    """
    if metrics.empty:
        return {"Male": 0, "Female": 0}
    return {
        "Male": int(metrics["male_count"].sum()),
        "Female": int(metrics["female_count"].sum()),
    }


def latest_gender(metrics: pd.DataFrame) -> dict[str, int | str | None]:
    """Return the Male/Female counts from the most recent window that saw someone.

    Falls back to the latest row if none had viewers. The ``ts`` field lets the
    UI show *when* the snapshot was taken so the user can gauge freshness.
    """
    if metrics.empty:
        return {"Male": 0, "Female": 0, "ts": None}
    non_empty = metrics[metrics["viewers"] > 0]
    row = non_empty.iloc[-1] if not non_empty.empty else metrics.iloc[-1]
    ts = row.get("ts")
    ts_iso = None
    if ts is not None and pd.notna(ts):
        ts_iso = pd.Timestamp(ts).isoformat()
    return {
        "Male": int(row["male_count"]),
        "Female": int(row["female_count"]),
        "ts": ts_iso,
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
    - avg_viewers: average viewers per window (not summed)
    - avg_attending: average attending per window
    - attention_rate: attending / viewers (0-100%), the key effectiveness metric
    - avg_dwell_ms: average dwell time across the segment
    - local_time: human-friendly local timestamp extracted from segment_id

    Returns a DataFrame sorted by attention_rate descending (best content first).
    """
    if metrics.empty or "segment_id" not in metrics.columns:
        return pd.DataFrame(
            columns=[
                "segment_id", "local_time", "total_windows",
                "avg_viewers", "avg_attending", "attention_rate", "avg_dwell_ms",
            ]
        )

    # Drop rows with no segment_id
    with_seg = metrics[metrics["segment_id"].notna()].copy()
    if with_seg.empty:
        return pd.DataFrame(
            columns=[
                "segment_id", "local_time", "total_windows",
                "avg_viewers", "avg_attending", "attention_rate", "avg_dwell_ms",
            ]
        )

    grouped = with_seg.groupby("segment_id").agg(
        total_windows=("viewers", "count"),
        avg_viewers=("viewers", "mean"),
        avg_attending=("attending", "mean"),
        total_attending=("attending", "sum"),
        total_viewers=("viewers", "sum"),
        avg_dwell_ms=("avg_dwell_ms", "mean"),
    ).reset_index()

    grouped["avg_viewers"] = grouped["avg_viewers"].round(1)
    grouped["avg_attending"] = grouped["avg_attending"].round(1)
    grouped["attention_rate"] = (
        grouped["total_attending"] / grouped["total_viewers"].replace(0, float("nan")) * 100
    ).fillna(0).round(1)

    # Convert segment_id UTC timestamp to local time for display
    local_tz = datetime.now().astimezone().tzinfo

    def _segment_to_local_time(seg_id: str) -> str:
        try:
            # Extract UTC timestamp from: auto_2026-07-12T10-14-12.245Z_0000
            parts = seg_id.split("_", 1)[1].rsplit("_", 1)[0]
            # Convert dashes back to colons for ISO parsing: T10-14-12 → T10:14:12
            iso_str = parts[:10] + "T" + parts[11:].replace("-", ":")
            utc_dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            local_dt = utc_dt.astimezone(local_tz)
            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            return seg_id

    grouped["local_time"] = grouped["segment_id"].apply(_segment_to_local_time)

    # Drop intermediate columns
    grouped = grouped.drop(columns=["total_attending", "total_viewers"])

    return grouped.sort_values("local_time", ascending=False).reset_index(drop=True)
