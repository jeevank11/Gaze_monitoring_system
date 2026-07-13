"""Aggregate business-insight queries over the aggregate-only SQLite.

Everything here is a SELECT on ``metrics`` + ``segments``. No raw frames,
no per-person data — the schema is aggregate-only by contract (see
``docs/PRIVACY_THREAT_MODEL.md``).

Entry points used by the dashboard pages:

- :func:`gaze_kpis` — five audience-centric tiles (viewers, peak, dwell,
  attention, gender split) with same-length prior-window deltas.
- :func:`insights_kpis` — five executive-summary tiles with
  same-length prior-window deltas (legacy Business Insights view).
- :func:`all_segments` — every segment in the window (feeds the
  reach-vs-attention scatter and the master table).
- :func:`classify_quadrants` — tag each segment as Winner / Kill /
  Niche / Fill relative to the median split of the current window.
- :func:`top_by_impressions` / :func:`bottom_by_attention` /
  :func:`top_by_attention_time` — ranked-list helpers used by the
  Content leaderboard on the Gaze Insights page.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class InsightsKpis:
    """Top-of-page executive tiles for the Business Insights view.

    ``*_delta`` fields are ``None`` when there is no prior window to compare
    against (e.g. the user picked "All time" or the DB is fresh). Deltas are
    always computed against a prior window of *the same length* ending at
    the start of the current one — so "Last 24 h" compares to the 24 h
    immediately before that.
    """

    impressions: int
    engaged_impressions: int
    attention_pct: float
    unique_segments: int
    avg_dwell_ms: float
    impressions_delta: int | None
    engaged_delta: int | None
    attention_delta_pp: float | None
    segments_delta: int | None
    dwell_delta_ms: float | None


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"metrics database not found at {db_path}. "
            "Run the pipeline with `--sink` to populate it."
        )
    return sqlite3.connect(str(db_path))


def _time_filter(window_minutes: int | None) -> tuple[str, tuple]:
    """Return an ``AND m.ts >= ?`` fragment (or empty) and its params."""
    if window_minutes is None:
        return "", ()
    cutoff = (
        (datetime.now(tz=UTC) - timedelta(minutes=window_minutes))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return "AND m.ts >= ?", (cutoff,)


def _segment_stats(
    db_path: Path,
    *,
    window_minutes: int | None,
    order_by: str,
    limit: int,
    min_viewers: int | None = None,
) -> pd.DataFrame:
    """Shared aggregation query behind both leaderboards.

    ``order_by`` is a fixed literal from the two callers below (never user
    input) — safe to inline into the SQL text.
    """
    where_ts, ts_params = _time_filter(window_minutes)
    if min_viewers is not None:
        having = "HAVING SUM(m.viewers) >= ?"
        having_params: tuple = (int(min_viewers),)
    else:
        having = ""
        having_params = ()

    # ``order_by`` and ``having`` are hardcoded fragments controlled by the two
    # callers below (never user input); the time filter and min_viewers are
    # bound as SQL parameters. Safe from injection — silence ruff's heuristic.
    query = f"""
        SELECT s.id                                        AS segment_id,
               s.started_at                                AS started_at,
               s.ended_at                                  AS ended_at,
               s.duration_ms                               AS duration_ms,
               s.thumbnail_b64                             AS thumbnail_b64,
               SUM(m.viewers)                              AS impressions,
               SUM(m.attending)                            AS engaged_impressions,
               COALESCE(SUM(m.attending * m.window_seconds), 0)
                                                          AS attention_seconds,
               ROUND(
                   100.0 * SUM(m.attending)
                   / NULLIF(SUM(m.viewers), 0), 1
               )                                           AS attention_pct
        FROM   segments s
        JOIN   metrics  m ON m.segment_id = s.id
        WHERE  1=1 {where_ts}
        GROUP BY s.id
        {having}
        ORDER BY {order_by}
        LIMIT ?
    """  # noqa: S608 - order_by/having are internal literals, not user input
    params = (*ts_params, *having_params, limit)
    with closing(_connect(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
        df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True, errors="coerce")
    return df


def top_by_impressions(
    db_path: Path,
    window_minutes: int | None = 24 * 60,
    limit: int = 5,
) -> pd.DataFrame:
    """Top segments by total viewer-impressions in the window.

    Ties broken by ``engaged_impressions`` so the one that actually earned
    attention wins.
    """
    return _segment_stats(
        db_path,
        window_minutes=window_minutes,
        order_by="impressions DESC, engaged_impressions DESC",
        limit=limit,
    )


def top_by_attention_time(
    db_path: Path,
    window_minutes: int | None = 24 * 60,
    limit: int = 5,
    min_viewers: int = 1,
) -> pd.DataFrame:
    """Top segments by attention time (``SUM(attending * window_seconds)``).

    This is the "which content earned the most eyes-on time?" leaderboard.
    Ties broken by impressions so higher-reach content wins on a tie.
    """
    return _segment_stats(
        db_path,
        window_minutes=window_minutes,
        order_by="attention_seconds DESC, impressions DESC",
        limit=limit,
        min_viewers=min_viewers,
    )


def bottom_by_attention(
    db_path: Path,
    window_minutes: int | None = 24 * 60,
    limit: int = 5,
    min_viewers: int = 10,
) -> pd.DataFrame:
    """Segments with the lowest attention rate; excludes low-sample noise.

    A segment must have at least ``min_viewers`` cumulative viewer-samples
    across the window to be eligible — otherwise a 5-second window with a
    single distracted person would always "win" as worst content.
    """
    return _segment_stats(
        db_path,
        window_minutes=window_minutes,
        order_by="attention_pct ASC, impressions DESC",
        limit=limit,
        min_viewers=min_viewers,
    )


def all_segments(
    db_path: Path,
    window_minutes: int | None = 24 * 60,
    min_viewers: int = 1,
    limit: int = 1000,
) -> pd.DataFrame:
    """All segments in the window with impressions / attention aggregates.

    Feeds both the reach-vs-attention scatter and the master table. The
    ``min_viewers`` guard filters out segments that never had a single
    viewer-sample — those are noise on the scatter plot.
    """
    return _segment_stats(
        db_path,
        window_minutes=window_minutes,
        order_by="impressions DESC",
        limit=limit,
        min_viewers=min_viewers,
    )


def classify_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each segment with one of four quadrants relative to the medians.

    Returns a *copy* of ``df`` with a new ``quadrant`` column taking one of
    ``"Winner"``, ``"Kill"``, ``"Niche"``, ``"Fill"``. Splits are computed
    on the median of ``impressions`` and ``attention_pct`` within the
    supplied frame — so the labels are relative to the current window,
    not absolute thresholds.
    """
    out = df.copy()
    if out.empty:
        out["quadrant"] = pd.Series(dtype="object")
        return out

    impressions = out["impressions"].astype(float)
    attention = out["attention_pct"].astype(float).fillna(0.0)
    impressions_median = impressions.median()
    attention_median = attention.median()

    def _label(imp: float, attn: float) -> str:
        high_imp = imp >= impressions_median
        high_attn = attn >= attention_median
        if high_imp and high_attn:
            return "Winner"
        if high_imp and not high_attn:
            return "Kill"
        if not high_imp and high_attn:
            return "Niche"
        return "Fill"

    out["quadrant"] = [
        _label(i, a) for i, a in zip(impressions, attention, strict=True)
    ]
    return out


QUADRANT_COLORS: dict[str, str] = {
    "Winner": "#2ca02c",  # green
    "Kill": "#d62728",  # red
    "Niche": "#1f77b4",  # blue
    "Fill": "#7f7f7f",  # gray
}
QUADRANT_ICONS: dict[str, str] = {
    "Winner": "🏆 Winner",
    "Kill": "⚠️ Kill",
    "Niche": "💎 Niche",
    "Fill": "▫️ Fill",
}


def insights_kpis(
    db_path: Path,
    window_minutes: int | None = 24 * 60,
) -> InsightsKpis:
    """Executive-summary tiles for the top of the Business Insights page.

    Runs one aggregate over the current window and, when the window is
    bounded, a second aggregate over the equal-length prior window to
    compute deltas.
    """
    with closing(_connect(db_path)) as conn:
        current = _kpi_aggregate(conn, window_minutes=window_minutes)
        if window_minutes is None:
            prior = None
        else:
            prior = _kpi_aggregate(
                conn,
                window_minutes=window_minutes,
                offset_minutes=window_minutes,
            )

    def _delta_int(cur: int, prv: int | None) -> int | None:
        return None if prv is None else int(cur - prv)

    def _delta_float(cur: float, prv: float | None) -> float | None:
        return None if prv is None else float(cur - prv)

    return InsightsKpis(
        impressions=current["impressions"],
        engaged_impressions=current["engaged_impressions"],
        attention_pct=current["attention_pct"],
        unique_segments=current["unique_segments"],
        avg_dwell_ms=current["avg_dwell_ms"],
        impressions_delta=_delta_int(
            current["impressions"], prior["impressions"] if prior else None
        ),
        engaged_delta=_delta_int(
            current["engaged_impressions"],
            prior["engaged_impressions"] if prior else None,
        ),
        attention_delta_pp=_delta_float(
            current["attention_pct"], prior["attention_pct"] if prior else None
        ),
        segments_delta=_delta_int(
            current["unique_segments"],
            prior["unique_segments"] if prior else None,
        ),
        dwell_delta_ms=_delta_float(
            current["avg_dwell_ms"], prior["avg_dwell_ms"] if prior else None
        ),
    )


def _kpi_aggregate(
    conn: sqlite3.Connection,
    *,
    window_minutes: int | None,
    offset_minutes: int = 0,
) -> dict:
    """Run one KPI aggregation query. ``offset_minutes`` shifts the window
    back in time (used to compute the prior period for deltas)."""
    now = datetime.now(tz=UTC)
    where = ""
    params: tuple = ()
    if window_minutes is not None:
        end = now - timedelta(minutes=offset_minutes)
        start = end - timedelta(minutes=window_minutes)
        where = "WHERE ts >= ? AND ts < ?"
        params = (_iso(start), _iso(end))

    query = f"""
        SELECT COALESCE(SUM(viewers), 0)                       AS impressions,
               COALESCE(SUM(attending), 0)                     AS engaged_impressions,
               COUNT(DISTINCT segment_id)                      AS unique_segments,
               COALESCE(AVG(NULLIF(avg_dwell_ms, 0)), 0.0)     AS avg_dwell_ms,
               COALESCE(MAX(viewers), 0)                       AS peak_concurrent,
               COALESCE(SUM(male_count), 0)                    AS male_count,
               COALESCE(SUM(female_count), 0)                  AS female_count,
               COALESCE(SUM(attending * window_seconds), 0)    AS attention_seconds
        FROM   metrics
        {where}
    """  # noqa: S608 - `where` is a fixed literal, params are bound
    row = conn.execute(query, params).fetchone()
    impressions = int(row[0] or 0)
    engaged = int(row[1] or 0)
    attention_pct = (100.0 * engaged / impressions) if impressions > 0 else 0.0
    return {
        "impressions": impressions,
        "engaged_impressions": engaged,
        "unique_segments": int(row[2] or 0),
        "avg_dwell_ms": float(row[3] or 0.0),
        "attention_pct": round(attention_pct, 1),
        "peak_concurrent": int(row[4] or 0),
        "male_count": int(row[5] or 0),
        "female_count": int(row[6] or 0),
        "attention_seconds": int(row[7] or 0),
    }


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Gaze Insights — audience-centric tiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GazeKpis:
    """Top-of-page audience tiles for the Gaze Insights view.

    ``*_delta`` fields follow the same convention as :class:`InsightsKpis`:
    ``None`` when no prior window is available, otherwise the signed change
    versus the equal-length prior window.

    ``attention_time_seconds`` is ``SUM(attending * window_seconds)`` — the
    total number of viewer-seconds where at least one person was actually
    attending the screen. It's the closest metric we have to the standard
    OOH-advertising "eyes-on" currency.
    """

    total_viewers: int
    peak_concurrent: int
    avg_dwell_ms: float
    attention_pct: float
    male_count: int
    female_count: int
    attention_time_seconds: int
    total_viewers_delta: int | None
    peak_concurrent_delta: int | None
    dwell_delta_ms: float | None
    attention_delta_pp: float | None
    attention_time_delta_s: int | None


def gaze_kpis(
    db_path: Path,
    window_minutes: int | None = 24 * 60,
) -> GazeKpis:
    """Audience-centric KPI tiles for the Gaze Insights page.

    Reuses :func:`_kpi_aggregate` for the current window and, when the
    window is bounded, the equal-length prior window (for deltas).
    """
    with closing(_connect(db_path)) as conn:
        current = _kpi_aggregate(conn, window_minutes=window_minutes)
        if window_minutes is None:
            prior = None
        else:
            prior = _kpi_aggregate(
                conn,
                window_minutes=window_minutes,
                offset_minutes=window_minutes,
            )

    def _di(cur: int, prv: int | None) -> int | None:
        return None if prv is None else int(cur - prv)

    def _df(cur: float, prv: float | None) -> float | None:
        return None if prv is None else float(cur - prv)

    return GazeKpis(
        total_viewers=current["impressions"],
        peak_concurrent=current["peak_concurrent"],
        avg_dwell_ms=current["avg_dwell_ms"],
        attention_pct=current["attention_pct"],
        male_count=current["male_count"],
        female_count=current["female_count"],
        attention_time_seconds=current["attention_seconds"],
        total_viewers_delta=_di(
            current["impressions"], prior["impressions"] if prior else None
        ),
        peak_concurrent_delta=_di(
            current["peak_concurrent"],
            prior["peak_concurrent"] if prior else None,
        ),
        dwell_delta_ms=_df(
            current["avg_dwell_ms"], prior["avg_dwell_ms"] if prior else None
        ),
        attention_delta_pp=_df(
            current["attention_pct"], prior["attention_pct"] if prior else None
        ),
        attention_time_delta_s=_di(
            current["attention_seconds"],
            prior["attention_seconds"] if prior else None,
        ),
    )
