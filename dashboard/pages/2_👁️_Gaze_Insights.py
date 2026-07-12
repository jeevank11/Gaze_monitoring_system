"""Gaze Insights — audience-centric view.

Three layers, all sourced from aggregate SQL only:

1. **Audience KPI strip** — total viewers, peak concurrent, avg dwell,
   attention rate, M/F split. Each tile shows a delta vs the equal-length
   prior window.
2. **Audience engagement trend** — one time-series chart with viewers and
   attending lines. Auto-rebinned to keep the chart legible at every zoom
   level (5-second raw → 1-minute → 15-minute → 1-hour depending on the
   window length).
3. **Gender split** — donut chart of M/F/Nobody counts.

All numbers come from aggregate rows only. No faces, no frames, no
per-person records.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from dashboard.business import gaze_kpis, top_by_attention_time  # noqa: E402
from dashboard.data import metrics_frame  # noqa: E402
from dashboard.theme import inject_theme  # noqa: E402

from gaze_analytics.config import settings  # noqa: E402

st.set_page_config(
    page_title="Gaze Insights",
    page_icon="👁️",
    layout="wide",
)

# Shared stylesheet — keeps typography and sidebar styling identical across
# every page in the multipage app (see dashboard/theme.py).
inject_theme()

_LOCAL_TZ = datetime.now().astimezone().tzinfo


def _fmt_duration_ms(ms: float | int | None) -> str:
    """Human-readable dwell duration.

    Chooses the unit that keeps the number readable:
    seconds under 1 minute, minutes under 1 hour, hours above.
    """
    if ms is None or pd.isna(ms):
        return "—"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:,.1f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:,.1f} min"
    return f"{minutes / 60.0:,.1f} h"


def _short_segment_id(sid: str) -> str:
    if not isinstance(sid, str):
        return str(sid)
    tail = sid.rsplit("_", 1)[-1]
    return f"#{tail}" if tail.isdigit() else sid


# ---- Cached wrappers so rerender-heavy widgets don't refetch --------------


@st.cache_data(ttl=30)
def _cached_kpis(db_path: Path, window_minutes: int | None):
    return gaze_kpis(db_path, window_minutes=window_minutes)


@st.cache_data(ttl=30)
def _cached_metrics(db_path: Path, window_minutes: int | None) -> pd.DataFrame:
    return metrics_frame(db_path, window_minutes=window_minutes)


@st.cache_data(ttl=30)
def _cached_leaderboard(
    db_path: Path, window_minutes: int | None, limit: int
) -> pd.DataFrame:
    return top_by_attention_time(
        db_path, window_minutes=window_minutes, limit=limit
    )


# ---- Header ---------------------------------------------------------------

st.title("👁️ Gaze Insights")
st.caption(
    "Who is watching, for how long, and how their attention flows. "
    "All numbers come from aggregate rows only — no faces, no frames."
)

# ---- Sidebar --------------------------------------------------------------

st.sidebar.header("Insights window")
window_label = st.sidebar.selectbox(
    "Time window",
    options=["Last 1 hour", "Last 24 hours", "Last 7 days", "All time"],
    index=1,
)
_WINDOW_MAP: dict[str, int | None] = {
    "Last 1 hour": 60,
    "Last 24 hours": 24 * 60,
    "Last 7 days": 7 * 24 * 60,
    "All time": None,
}
window_minutes = _WINDOW_MAP[window_label]

db_path = Path(
    st.sidebar.text_input("SQLite path", value=str(settings.sqlite_path))
)

st.sidebar.markdown(
    "**Privacy contract**  \n"
    "- No frames written to disk  \n"
    "- No face recognition, no embeddings  \n"
    "- Only aggregate counters + phash + thumbnail leave the pipeline"
)

if not db_path.exists():
    st.warning(
        f"Metrics database not found at `{db_path}`. "
        "Start the pipeline from the **Live Ops** page first."
    )
    st.stop()

# ---- Fetch ----------------------------------------------------------------

kpis = _cached_kpis(db_path, window_minutes)
metrics_df = _cached_metrics(db_path, window_minutes)
leaderboard_df = _cached_leaderboard(db_path, window_minutes, limit=5)

# ---- Layer 1: Audience KPI strip ------------------------------------------


def _fmt_pct_change(cur: int, delta: int | None) -> str | None:
    if delta is None:
        return None
    prior = cur - delta
    if prior == 0:
        return f"{'+' if delta >= 0 else ''}{delta:,} vs 0"
    return f"{100.0 * delta / prior:+.1f}%"


def _fmt_int_delta(delta: int | None) -> str | None:
    return None if delta is None else f"{delta:+,d}"


def _fmt_seconds_delta(delta_ms: float | None) -> str | None:
    if delta_ms is None:
        return None
    if abs(delta_ms) < 1000:
        return f"{delta_ms:+.0f} ms"
    return f"{delta_ms / 1000.0:+.1f} s"


def _fmt_pp_delta(delta_pp: float | None) -> str | None:
    return None if delta_pp is None else f"{delta_pp:+.1f} pp"


def _fmt_attention_time(seconds: int | float | None) -> str:
    """Human-readable duration: `52 min`, `3.2 h`, `2d 4h`."""
    if seconds is None or pd.isna(seconds):
        return "—"
    secs = int(seconds)
    if secs < 60:
        return f"{secs} s"
    if secs < 3600:
        return f"{secs // 60} min"
    if secs < 86400:
        return f"{secs / 3600:.1f} h"
    days = secs // 86400
    hours = (secs % 86400) // 3600
    return f"{days}d {hours}h"


def _fmt_attention_time_delta(delta_s: int | None) -> str | None:
    if delta_s is None:
        return None
    if abs(delta_s) < 60:
        return f"{delta_s:+d} s"
    if abs(delta_s) < 3600:
        return f"{delta_s / 60:+.0f} min"
    return f"{delta_s / 3600:+.1f} h"


def _fmt_ratio(male: int, female: int) -> str:
    total = male + female
    if total == 0:
        return "—"
    m_pct = 100.0 * male / total
    f_pct = 100.0 * female / total
    return f"{m_pct:.0f}% / {f_pct:.0f}%"


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Impressions",
    f"{kpis.total_viewers:,}",
    delta=_fmt_pct_change(kpis.total_viewers, kpis.total_viewers_delta),
)
c2.metric(
    "Attention time",
    _fmt_attention_time(kpis.attention_time_seconds),
    delta=_fmt_attention_time_delta(kpis.attention_time_delta_s),
)
c3.metric(
    "Avg viewing duration",
    _fmt_duration_ms(kpis.avg_dwell_ms),
    delta=_fmt_seconds_delta(kpis.dwell_delta_ms),
)
c4.metric(
    "Attention rate",
    f"{kpis.attention_pct:.1f}%",
    delta=_fmt_pp_delta(kpis.attention_delta_pp),
)
c5.metric(
    "M / F ratio",
    _fmt_ratio(kpis.male_count, kpis.female_count),
    delta=f"{kpis.male_count:,} M · {kpis.female_count:,} F",
    delta_color="off",
)

_peak_delta_txt = _fmt_int_delta(kpis.peak_concurrent_delta)
_peak_delta_suffix = f" ({_peak_delta_txt} vs prior)" if _peak_delta_txt else ""
st.caption(
    "**Impressions** — viewer-samples across 5 s windows (not unique people).  \n"
    "**Attention time** — seconds where at least one person was actively "
    "looking at the screen.  \n"
    "**Avg viewing duration** — mean time a single viewer stayed in frame "
    "before leaving.  \n"
    "**Attention rate** — share of viewer-samples classified as attending "
    "(engaged ÷ impressions).  \n"
    f"**Peak** — most viewers seen at once: **{kpis.peak_concurrent:,}**"
    f"{_peak_delta_suffix}."
)

st.caption(
    f"Window: **{window_label}** · "
    "deltas compare to the equal-length prior window."
)

st.divider()

# ---- Layer 2: Audience engagement trend -----------------------------------

st.subheader("Audience engagement trend")

if metrics_df.empty:
    st.info(
        "No aggregate rows in this window yet. "
        "Start the pipeline from the **Live Ops** page and wait for a window "
        "to close (5 seconds by default)."
    )
else:
    # Auto-rebin so the chart stays legible at every zoom level.
    if window_minutes is None:
        rebin = "1h"
    elif window_minutes <= 60:
        rebin = None  # raw 5-second bins
    elif window_minutes <= 24 * 60:
        rebin = "1min"
    elif window_minutes <= 7 * 24 * 60:
        rebin = "15min"
    else:
        rebin = "1h"

    trend = metrics_df.copy()
    trend["ts"] = pd.to_datetime(trend["ts"], utc=True, errors="coerce")
    trend["ts"] = trend["ts"].dt.tz_convert(_LOCAL_TZ).dt.tz_localize(None)

    if rebin is not None:
        trend = (
            trend.set_index("ts")[["viewers", "attending"]]
            .resample(rebin)
            .sum()
            .reset_index()
        )
    else:
        trend = trend[["ts", "viewers", "attending"]]

    long_df = trend.melt(
        id_vars="ts",
        value_vars=["viewers", "attending"],
        var_name="series",
        value_name="count",
    )
    long_df["series"] = long_df["series"].map(
        {"viewers": "Viewers", "attending": "Attending"}
    )

    fig = px.line(
        long_df,
        x="ts",
        y="count",
        color="series",
        color_discrete_map={"Viewers": "#1f77b4", "Attending": "#2ca02c"},
        labels={"ts": "Time", "count": "People", "series": ""},
    )
    fig.update_traces(mode="lines", line={"width": 2})
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        height=360,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.05, "x": 0},
    )
    st.plotly_chart(fig, width="stretch")

    # Peak / quiet moments (only meaningful when we have a few bins).
    if len(trend) >= 3 and trend["viewers"].sum() > 0:
        peak_row = trend.loc[trend["viewers"].idxmax()]
        peak_time = peak_row["ts"]
        peak_val = int(peak_row["viewers"])
        st.caption(
            f"Peak: **{peak_val}** viewers around "
            f"{peak_time:%b %d, %H:%M}"
            + (f" (rebinned to {rebin})" if rebin else " (raw 5 s windows)")
        )

st.divider()

# ---- Layer 2b: Top content by attention time ------------------------------

st.subheader("Top content by attention time")

if leaderboard_df.empty:
    st.caption("No segments with attention yet.")
else:
    lb = leaderboard_df.copy()
    lb["attention_time"] = lb["attention_seconds"].fillna(0).astype(int).apply(
        _fmt_attention_time
    )
    # ImageColumn needs a URL / data-URL, not a decoded ndarray. Wrap the
    # already-base64 JPEG blob straight from the DB.
    lb["thumbnail"] = lb["thumbnail_b64"].apply(
        lambda s: f"data:image/jpeg;base64,{s}" if isinstance(s, str) and s else None
    )
    lb["segment"] = lb["segment_id"].apply(_short_segment_id)
    lb["started_at"] = pd.to_datetime(
        lb["started_at"], utc=True, errors="coerce"
    ).dt.tz_convert(_LOCAL_TZ).dt.tz_localize(None)
    lb["rank"] = range(1, len(lb) + 1)
    lb = lb[
        [
            "rank",
            "thumbnail",
            "segment",
            "attention_time",
            "impressions",
            "attention_pct",
            "started_at",
        ]
    ]
    st.dataframe(
        lb,
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "thumbnail": st.column_config.ImageColumn(
                "Preview", width="small"
            ),
            "segment": st.column_config.TextColumn("Segment"),
            "attention_time": st.column_config.TextColumn(
                "Attention time",
                help="SUM(attending * window_seconds) for this segment.",
            ),
            "impressions": st.column_config.ProgressColumn(
                "Impressions",
                format="%d",
                min_value=0,
                max_value=int(lb["impressions"].max() or 1),
            ),
            "attention_pct": st.column_config.ProgressColumn(
                "Attention rate",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "started_at": st.column_config.DatetimeColumn(
                "Started at",
                format="MMM D, HH:mm",
            ),
        },
    )
    st.caption(
        "Ranked by attention time (seconds where a viewer was actively "
        "looking at the screen). Ties broken by impressions."
    )

st.divider()

# ---- Layer 3: Gender split ------------------------------------------------

st.subheader("Gender split")
if metrics_df.empty:
    st.caption("No data yet.")
else:
    male_total = int(metrics_df["male_count"].sum())
    female_total = int(metrics_df["female_count"].sum())
    # "Nobody" windows: rows where viewers == 0. Kept as a category so the
    # ratios aren't inflated by silent stretches.
    nobody_total = int((metrics_df["viewers"] == 0).sum())

    if male_total + female_total + nobody_total == 0:
        st.caption("No detections in this window.")
    else:
        donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Male", "Female", "Nobody in frame"],
                    values=[male_total, female_total, nobody_total],
                    hole=0.55,
                    marker={"colors": ["#4c78a8", "#e45756", "#bab0ac"]},
                    textinfo="label+percent",
                    textposition="outside",
                    sort=False,
                )
            ]
        )
        donut.update_layout(
            margin={"l": 20, "r": 20, "t": 10, "b": 10},
            height=320,
            showlegend=False,
        )
        st.plotly_chart(donut, width="stretch")
        st.caption(
            "Estimated by the on-device age/gender model. "
            "No identity is stored."
        )
