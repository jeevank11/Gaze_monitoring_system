"""Streamlit dashboard for gaze analytics.

Two planes on a single page:

- **Control plane** (sidebar): start/stop the pipeline as a detached subprocess,
  toggle module flags, pick device + privacy mode. Config persists across
  Streamlit reruns via a pidfile so the dashboard can be restarted without
  killing an in-flight pipeline.
- **Data plane** (main area): KPI tiles, trend charts, demographics, content
  segments, plus a status card and log tail.

Every metric shown here comes from **aggregate rows** in the SQLite database —
no faces, no bboxes, no frames. That's the privacy contract; this UI just
renders it. The subprocess helpers in :mod:`dashboard.process` see PIDs and
log lines only.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Ensure the repo root is on sys.path when Streamlit launches this script
# directly (streamlit prepends the script's directory, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from dashboard.data import (  # noqa: E402
    Kpis,
    compute_kpis,
    content_effectiveness,
    decode_thumbnail,
    gender_totals,
    metrics_frame,
    segments_frame,
)
from dashboard.process import (  # noqa: E402
    DEFAULT_LOG_PATH,
    DEFAULT_PID_PATH,
    PipelineConfig,
    cli_preview,
    format_uptime,
    start,
    status,
    stop,
    tail_log,
    uptime_seconds,
)
from streamlit_autorefresh import st_autorefresh  # noqa: E402

from gaze_analytics.config import settings  # noqa: E402

st.set_page_config(
    page_title="Gaze Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)


# ---- Session state defaults ----------------------------------------------

def _init_state() -> None:
    if "pipeline_cfg" not in st.session_state:
        st.session_state.pipeline_cfg = PipelineConfig()
    if "last_action" not in st.session_state:
        st.session_state.last_action = ""


_init_state()

pipeline_status = status(DEFAULT_PID_PATH)
# If a pipeline was launched previously, prefer its live config over any
# session default so the sidebar reflects reality.
if pipeline_status.running and pipeline_status.config is not None:
    st.session_state.pipeline_cfg = pipeline_status.config

current_cfg: PipelineConfig = st.session_state.pipeline_cfg

# ---- Sidebar : Control plane ---------------------------------------------

st.sidebar.header("Pipeline control")

if pipeline_status.running:
    st.sidebar.success(f"Running — PID {pipeline_status.pid}")
else:
    st.sidebar.info("Stopped")

start_col, stop_col = st.sidebar.columns(2)
start_disabled = pipeline_status.running
stop_disabled = not pipeline_status.running

if start_col.button("▶ Start", disabled=start_disabled, width="stretch"):
    new_status = start(current_cfg)
    st.session_state.last_action = (
        f"Started PID {new_status.pid}" if new_status.running else "Start failed"
    )
    st.rerun()

if stop_col.button("⏹ Stop", disabled=stop_disabled, width="stretch"):
    stopped = stop()
    st.session_state.last_action = "Stopped pipeline" if stopped else "Nothing to stop"
    st.rerun()

if st.session_state.last_action:
    st.sidebar.caption(st.session_state.last_action)

st.sidebar.divider()
st.sidebar.subheader("Configuration")

_config_disabled = pipeline_status.running  # can't reconfigure while running

device = st.sidebar.selectbox(
    "Device",
    options=["AUTO", "CPU", "GPU"],
    index=["AUTO", "CPU", "GPU"].index(current_cfg.device),
    disabled=_config_disabled,
    help="OpenVINO device. AUTO picks the best available.",
)
privacy_mode = st.sidebar.selectbox(
    "Privacy mode",
    options=["blur", "pixelate", "silhouette", "off"],
    index=["blur", "pixelate", "silhouette", "off"].index(current_cfg.privacy_mode),
    disabled=_config_disabled,
    help="How the preview window masks faces. Aggregates are unaffected.",
)
log_level = st.sidebar.selectbox(
    "Log level",
    options=["DEBUG", "INFO", "WARNING", "ERROR"],
    index=["DEBUG", "INFO", "WARNING", "ERROR"].index(current_cfg.log_level),
    disabled=_config_disabled,
)

st.sidebar.markdown("**Modules**")
detect_faces = st.sidebar.checkbox(
    "Detect faces", value=current_cfg.detect_faces, disabled=_config_disabled
)
head_pose = st.sidebar.checkbox(
    "Head pose + attention", value=current_cfg.head_pose, disabled=_config_disabled
)
track_flag = st.sidebar.checkbox(
    "Track (IoU + Kalman)", value=current_cfg.track, disabled=_config_disabled
)
segment_content = st.sidebar.checkbox(
    "Segment screen content (pHash)",
    value=current_cfg.segment_content,
    disabled=_config_disabled,
)
age_gender = st.sidebar.checkbox(
    "Age / gender aggregate", value=current_cfg.age_gender, disabled=_config_disabled
)
gaze_flag = st.sidebar.checkbox(
    "Gaze estimation (precise attention)", value=current_cfg.gaze, disabled=_config_disabled
)
sink = st.sidebar.checkbox(
    "SQLite aggregate sink", value=current_cfg.sink, disabled=_config_disabled
)
preview = st.sidebar.checkbox(
    "Preview window (OS-native)",
    value=current_cfg.preview,
    disabled=_config_disabled,
    help="Opens a local OpenCV window showing the privacy-masked feed.",
)

# Snapshot the sidebar into a new immutable config for the next Start click.
if not pipeline_status.running:
    st.session_state.pipeline_cfg = replace(
        current_cfg,
        device=device,  # type: ignore[arg-type]
        privacy_mode=privacy_mode,  # type: ignore[arg-type]
        log_level=log_level,
        detect_faces=detect_faces,
        head_pose=head_pose,
        track=track_flag,
        segment_content=segment_content,
        age_gender=age_gender,
        gaze=gaze_flag,
        sink=sink,
        preview=preview,
    )

with st.sidebar.expander("Equivalent CLI command", expanded=False):
    st.code(cli_preview(st.session_state.pipeline_cfg), language="bash")

st.sidebar.divider()
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

auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto_refresh:
    st_autorefresh(interval=5000, key="_auto_refresh")

st.sidebar.markdown(
    "**Privacy contract**  \n"
    "- No frames written to disk  \n"
    "- No face recognition, no embeddings  \n"
    "- SQLite stores counters + phash only"
)

# ---- Header ---------------------------------------------------------------

st.title("Gaze Analytics — Signage Engagement")
st.caption("Privacy-preserving audience analytics on Intel CPU/iGPU.")

# ---- Status card ----------------------------------------------------------

status_cols = st.columns(4)
status_cols[0].metric(
    "Pipeline",
    "Running" if pipeline_status.running else "Stopped",
)
status_cols[1].metric("PID", pipeline_status.pid or "—")
status_cols[2].metric(
    "Uptime", format_uptime(uptime_seconds(pipeline_status)) if pipeline_status.running else "—"
)
status_cols[3].metric(
    "Device",
    (pipeline_status.config.device if pipeline_status.config else current_cfg.device),
)

st.divider()

# ---- Data plane -----------------------------------------------------------

db_path = Path(db_path_str)
if not db_path.exists():
    st.warning(
        f"Metrics database not found at `{db_path}`. "
        "Start the pipeline from the sidebar (with **SQLite aggregate sink** enabled) "
        "to begin writing aggregate rows."
    )
    _render_charts = False
else:
    _render_charts = True

if _render_charts:
    metrics_df = metrics_frame(db_path, window_minutes=window_minutes)
    segments_df = segments_frame(db_path, limit=20)
    kpis: Kpis = compute_kpis(metrics_df, segments_df)

    # ---- KPI tiles --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Viewers Now", kpis.viewers_now)
    col2.metric("Attending Now", kpis.attending_now)
    col3.metric("Attention Rate", f"{kpis.attention_rate}%")
    col4.metric("Avg Dwell (ms)", f"{kpis.avg_dwell_ms:,.0f}")
    col5.metric("Segments", kpis.total_segments)

    st.divider()

    # ---- Trend chart ------------------------------------------------------

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
        st.plotly_chart(fig, width="stretch")

    # ---- Demographics -----------------------------------------------------

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Demographics (aggregate)")
        gt = gender_totals(metrics_df)
        if gt["Male"] + gt["Female"] + gt["Nobody"] == 0:
            st.caption("No demographic samples yet. Run with age/gender enabled.")
        else:
            labels = []
            values = []
            colors = []
            _color_map = {"Male": "#636EFA", "Female": "#EF553B", "Nobody": "#CCCCCC"}
            for label in ("Male", "Female", "Nobody"):
                if gt[label] > 0:
                    labels.append(label)
                    values.append(gt[label])
                    colors.append(_color_map[label])
            gender_fig = px.pie(
                names=labels,
                values=values,
                hole=0.55,
                color_discrete_sequence=colors,
            )
            gender_fig.update_layout(
                margin={"l": 20, "r": 20, "t": 10, "b": 20},
                height=280,
                showlegend=True,
            )
            st.plotly_chart(gender_fig, width="stretch")

    with col_b:
        st.subheader("Dwell time (ms)")
        if metrics_df.empty:
            st.caption("No dwell data yet.")
        else:
            dwell_fig = px.area(metrics_df, x="ts", y="avg_dwell_ms")
            dwell_fig.update_layout(
                margin={"l": 20, "r": 20, "t": 10, "b": 20}, height=280
            )
            st.plotly_chart(dwell_fig, width="stretch")

    st.divider()

    # ---- Content Effectiveness Leaderboard --------------------------------

    st.subheader("Content Effectiveness Score")
    effectiveness_df = content_effectiveness(metrics_df)
    if effectiveness_df.empty:
        st.caption(
            "No segment data yet. Run with *Segment screen content* and "
            "*SQLite aggregate sink* enabled."
        )
    else:
        # Show bar chart of attention rates
        eff_chart = effectiveness_df.head(10).copy()
        eff_chart["label"] = eff_chart["local_time"].str[-8:]  # show just HH:MM:SS
        fig_eff = px.bar(
            eff_chart,
            x="label",
            y="attention_rate",
            color="attention_rate",
            color_continuous_scale=["#EF553B", "#FECB52", "#00CC96"],
            range_color=[0, 100],
            labels={"attention_rate": "Attention %", "label": "Content Segment"},
        )
        fig_eff.update_layout(
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            height=300,
            xaxis_tickangle=-30,
            showlegend=False,
        )
        st.plotly_chart(fig_eff, width="stretch")

        # Detailed table
        with st.expander("Detailed scores", expanded=False):
            display_df = effectiveness_df[[
                "local_time", "attention_rate", "avg_viewers",
                "avg_attending", "avg_dwell_ms", "total_windows",
            ]].rename(columns={
                "local_time": "Content Started",
                "attention_rate": "Attention %",
                "avg_viewers": "Avg Viewers",
                "avg_attending": "Avg Attending",
                "avg_dwell_ms": "Avg Dwell (ms)",
                "total_windows": "Duration (windows)",
            })
            st.dataframe(display_df, width="stretch", hide_index=True)

    st.divider()

    # ---- Content segments -------------------------------------------------

    st.subheader("Recent content segments")
    if segments_df.empty:
        st.caption("No content segments yet. Enable *Segment screen content* to populate.")
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

# ---- Log tail -------------------------------------------------------------

with st.expander("Pipeline log (last 50 lines)", expanded=False):
    log_text = tail_log(DEFAULT_LOG_PATH, lines=50)
    if not log_text:
        st.caption(f"No log file yet at `{DEFAULT_LOG_PATH}`.")
    else:
        st.caption(
            f"Tailing `{DEFAULT_LOG_PATH}` "
            f"(refreshed {datetime.now().strftime('%H:%M:%S')})"
        )
        st.code(log_text, language="text")

st.caption(
    "See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) and "
    "[`docs/PRIVACY_THREAT_MODEL.md`](../docs/PRIVACY_THREAT_MODEL.md)."
)

