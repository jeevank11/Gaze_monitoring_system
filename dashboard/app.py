"""Streamlit dashboard for gaze analytics (Sprint 6).

Reads aggregate rows from ``data/metrics.sqlite`` and shows:

- KPI tiles (viewers now, attending now, rolling avg dwell, segments, total rows)
- Trend chart of viewers vs. attending over the selected window
- Demographic totals across the window (aggregate M / F counts)
- Content segments table with in-memory thumbnails

Every metric shown here comes from **aggregate rows** — no faces, no bboxes,
no frames. That's the privacy contract; this UI just renders it.
"""

from __future__ import annotations

import time
from pathlib import Path

import plotly.express as px
import streamlit as st
from dashboard.data import (
    Kpis,
    compute_kpis,
    decode_thumbnail,
    gender_totals,
    metrics_frame,
    segments_frame,
)

from gaze_analytics.config import settings

st.set_page_config(
    page_title="Gaze Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)

# ---- Sidebar --------------------------------------------------------------

st.sidebar.header("Data source")
db_path_str = st.sidebar.text_input(
    "SQLite path", value=str(settings.sqlite_path)
)
window_label = st.sidebar.selectbox(
    "Window",
    options=["Last 5 min", "Last 15 min", "Last 1 hour", "Last 24 hours", "All time"],
    index=2,
)
_WINDOW_MAP: dict[str, int | None] = {
    "Last 5 min": 5,
    "Last 15 min": 15,
    "Last 1 hour": 60,
    "Last 24 hours": 24 * 60,
    "All time": None,
}
window_minutes = _WINDOW_MAP[window_label]

auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

st.sidebar.markdown(
    "**Privacy contract**  \n"
    "- No frames written to disk  \n"
    "- No face recognition, no embeddings  \n"
    "- SQLite stores counters + phash only"
)

# ---- Header ---------------------------------------------------------------

st.title("Gaze Analytics — Signage Engagement")
st.caption("Privacy-preserving audience analytics on Intel CPU/iGPU.")

db_path = Path(db_path_str)
if not db_path.exists():
    st.warning(
        f"Metrics database not found at `{db_path}`. "
        "Start the pipeline with `python -m gaze_analytics run --sink` "
        "to begin writing aggregate rows."
    )
    st.stop()

metrics_df = metrics_frame(db_path, window_minutes=window_minutes)
segments_df = segments_frame(db_path, limit=20)
kpis: Kpis = compute_kpis(metrics_df, segments_df)

# ---- KPI tiles ------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Viewers now", kpis.viewers_now)
col2.metric("Attending now", kpis.attending_now)
col3.metric("Avg dwell (ms)", f"{kpis.avg_dwell_ms:,.0f}")
col4.metric("Segments (recent)", kpis.total_segments)
col5.metric("Rows in window", kpis.total_rows)

st.divider()

# ---- Trend chart ----------------------------------------------------------

st.subheader("Viewers vs. attending")
if metrics_df.empty:
    st.info(
        "No aggregate rows in the selected window yet. "
        "Start the pipeline and wait for a window to close."
    )
else:
    trend = metrics_df.melt(
        id_vars="ts",
        value_vars=["viewers", "attending"],
        var_name="series",
        value_name="count",
    )
    fig = px.line(
        trend,
        x="ts",
        y="count",
        color="series",
        markers=True,
        title=None,
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 10, "b": 20}, height=320)
    st.plotly_chart(fig, use_container_width=True)

# ---- Demographics ---------------------------------------------------------

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Demographics (aggregate)")
    gt = gender_totals(metrics_df)
    if gt["M"] + gt["F"] == 0:
        st.caption("No demographic samples yet. Run with `--age-gender` to enable.")
    else:
        gender_fig = px.pie(
            names=["Male", "Female"],
            values=[gt["M"], gt["F"]],
            hole=0.55,
        )
        gender_fig.update_layout(
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            height=280,
            showlegend=True,
        )
        st.plotly_chart(gender_fig, use_container_width=True)

with col_b:
    st.subheader("Dwell time (ms)")
    if metrics_df.empty:
        st.caption("No dwell data yet.")
    else:
        dwell_fig = px.area(metrics_df, x="ts", y="avg_dwell_ms")
        dwell_fig.update_layout(
            margin={"l": 20, "r": 20, "t": 10, "b": 20}, height=280
        )
        st.plotly_chart(dwell_fig, use_container_width=True)

st.divider()

# ---- Content segments -----------------------------------------------------

st.subheader("Recent content segments")
if segments_df.empty:
    st.caption("No content segments yet. Run with `--segment-content` to enable.")
else:
    for _, row in segments_df.iterrows():
        thumb_col, meta_col = st.columns([1, 4])
        with thumb_col:
            img = decode_thumbnail(row.get("thumbnail_b64"))
            if img is not None:
                st.image(img, width=160)
            else:
                st.caption("(no thumbnail)")
        with meta_col:
            started = row["started_at"]
            ended = row["ended_at"]
            dur_ms = row["duration_ms"]
            st.markdown(
                f"**`{row['id']}`**  \n"
                f"started: {started}  \n"
                f"ended: {ended if ended is not None else '_(in progress)_'}  \n"
                f"duration: {dur_ms if dur_ms is not None else '—'} ms  \n"
                f"phash: `{row['phash']}`"
            )

st.divider()
st.caption(
    "See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) and "
    "[`docs/PRIVACY_THREAT_MODEL.md`](../docs/PRIVACY_THREAT_MODEL.md)."
)

if auto_refresh:
    time.sleep(5)
    st.rerun()
