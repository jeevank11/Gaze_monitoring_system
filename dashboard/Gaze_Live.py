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
from dataclasses import fields, replace
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
    latest_gender,
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
from dashboard.theme import inject_theme  # noqa: E402
from streamlit_autorefresh import st_autorefresh  # noqa: E402

from gaze_analytics.capture.screen import list_monitors  # noqa: E402
from gaze_analytics.config import settings  # noqa: E402

st.set_page_config(
    page_title="Gaze Live",
    page_icon="📺",
    layout="wide",
)

# ---- Global theme / typography ------------------------------------------
# Shared across all pages (see dashboard/theme.py) so the sidebar nav,
# metric tiles and typography stay identical when the user switches pages.
inject_theme()


# ---- Session state defaults ----------------------------------------------

def _init_state() -> None:
    # Session state can hold a stale PipelineConfig from a previous run of
    # the dashboard (before new fields were added). If the stored instance is
    # missing any field the current dataclass declares, replace it with a
    # fresh default so widget rendering doesn't AttributeError.
    stored = st.session_state.get("pipeline_cfg")
    expected_fields = {f.name for f in fields(PipelineConfig)}
    if (
        stored is None
        or not isinstance(stored, PipelineConfig)
        or any(not hasattr(stored, name) for name in expected_fields)
    ):
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
st.sidebar.subheader("Presets")
_preset_disabled = pipeline_status.running
_p1, _p2 = st.sidebar.columns(2)
if _p1.button(
    "👤 Signage",
    disabled=_preset_disabled,
    width="stretch",
    help=(
        "Near-tier viewer analytics: a handful of people close to the screen. "
        "Runs the full stack (head pose + gaze + age/gender) for rich per-viewer "
        "attention. Min face height 80 px so posters/TVs behind the camera are ignored."
    ),
):
    st.session_state.pipeline_cfg = replace(
        current_cfg,
        device="AUTO",
        detect_faces=True,
        head_pose=True,
        gaze=True,
        age_gender=True,
        track=True,
        segment_content=True,
        sink=True,
        preview=True,
        min_face_height_px=80,
        face_confidence_threshold=0.7,
        tiled_detection=False,
        tile_grid=2,
    )
    st.session_state.last_action = "Preset: Signage (near-tier, few viewers)"
    st.rerun()

if _p2.button(
    "👥 Crowd",
    disabled=_preset_disabled,
    width="stretch",
    help=(
        "Aggregate analytics for 20+ people in frame. Turns OFF gaze and "
        "age/gender (too expensive per face) and lowers the min face height "
        "to 25 px so distant heads still count. Head-pose attention is kept."
    ),
):
    st.session_state.pipeline_cfg = replace(
        current_cfg,
        device="GPU",
        detect_faces=True,
        head_pose=True,
        gaze=False,
        age_gender=False,
        track=True,
        segment_content=True,
        sink=True,
        preview=True,
        min_face_height_px=25,
        face_confidence_threshold=0.4,
        tiled_detection=True,
        tile_grid=2,
    )
    st.session_state.last_action = "Preset: Crowd (aggregate, many viewers)"
    st.rerun()

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

st.sidebar.markdown("**Input source**")
# Enumerate physical monitors so the user can pick which display to
# screen-capture for content segmentation. mss returns index 0 = "all
# monitors combined" (usually not useful); indices 1..N are real displays.
_monitors = list_monitors()
_physical_monitors = _monitors[1:] if len(_monitors) > 1 else []
if _physical_monitors:
    _monitor_options = list(range(1, len(_physical_monitors) + 1))
    _monitor_labels = {
        i: f"Monitor {i}  \u2013  {m['width']}\u00d7{m['height']} @ ({m['left']}, {m['top']})"
        for i, m in zip(_monitor_options, _physical_monitors)
    }
    _current_monitor = int(current_cfg.monitor_index)
    if _current_monitor not in _monitor_options:
        _current_monitor = _monitor_options[0]
    monitor_index = st.sidebar.selectbox(
        "Screen to capture",
        options=_monitor_options,
        index=_monitor_options.index(_current_monitor),
        format_func=lambda i: _monitor_labels[i],
        disabled=_config_disabled,
        help=(
            "Which physical display to screen-capture for content segmentation. "
            "Change this if you're running the pipeline on a laptop but the ad "
            "content plays on an external monitor."
        ),
    )
else:
    monitor_index = int(current_cfg.monitor_index)
    st.sidebar.caption("Screen to capture: no monitors detected")

video_file_input = st.sidebar.text_input(
    "Video file (optional)",
    value=current_cfg.video_file or "",
    disabled=_config_disabled,
    help="Path to a local video file to use in place of the webcam. Leave blank to use the camera.",
)
# Users often paste paths wrapped in quotes (Windows "Copy as path" adds them);
# strip surrounding quotes/whitespace so OpenCV gets the raw filename.
_video_file = video_file_input.strip().strip('"').strip("'") or None

min_face_height_px = st.sidebar.slider(
    "Min face height (px)",
    min_value=5,
    max_value=200,
    value=int(current_cfg.min_face_height_px),
    step=5,
    disabled=_config_disabled,
    help=(
        "Faces smaller than this are ignored. Default 80 is safe for live "
        "signage (rejects faces on nearby screens). Drop to ~30 when testing "
        "with recorded videos where people appear small."
    ),
)

face_confidence_threshold = st.sidebar.slider(
    "Face detector confidence",
    min_value=0.20,
    max_value=0.95,
    value=float(current_cfg.face_confidence_threshold),
    step=0.05,
    disabled=_config_disabled,
    help=(
        "Confidence floor for face detections. Default 0.70 gives clean single-viewer "
        "tracking. Drop to ~0.40 for crowds so more small/partial faces get through "
        "(at the cost of a few false positives)."
    ),
)

tiled_detection = st.sidebar.checkbox(
    "Tiled face detection (crowd mode)",
    value=bool(current_cfg.tiled_detection),
    disabled=_config_disabled,
    help=(
        "Split the frame into an NxN grid and run the face detector on each tile. "
        "Roughly doubles the count of small faces detected in crowds. Costs ~NxN "
        "more detector time per frame (still real-time in crowd mode)."
    ),
)
tile_grid = st.sidebar.slider(
    "Tile grid (NxN)",
    min_value=2,
    max_value=4,
    value=int(current_cfg.tile_grid),
    step=1,
    disabled=_config_disabled or not tiled_detection,
    help="2 = 2x2 = 4 tiles (recommended). 3 = 3x3 = 9 tiles (denser crowds, ~2x slower).",
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
        video_file=_video_file,
        monitor_index=int(monitor_index),
        min_face_height_px=min_face_height_px,
        face_confidence_threshold=face_confidence_threshold,
        tiled_detection=tiled_detection,
        tile_grid=tile_grid,
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
    options=["Current run", "Last 5 min", "Last 15 min", "Last 30 min", "Last 1 hour", "All time"],
    index=0,
)
_WINDOW_MAP: dict[str, int | None] = {
    "Current run": None,
    "Last 5 min": 5,
    "Last 15 min": 15,
    "Last 30 min": 30,
    "Last 1 hour": 60,
    "All time": None,
}
_current_run_only = window_label == "Current run"
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

# ---- Danger zone: wipe SQLite -------------------------------------------

with st.sidebar.expander("Danger zone", expanded=False):
    _wipe_disabled = pipeline_status.running
    if _wipe_disabled:
        st.caption("Stop the pipeline before clearing data.")
    _confirm_wipe = st.checkbox(
        "Yes, delete all metrics and segments",
        value=False,
        disabled=_wipe_disabled,
        key="_confirm_wipe",
    )
    if st.button(
        "🗑 Clear all data",
        disabled=_wipe_disabled or not _confirm_wipe,
        width="stretch",
    ):
        _wipe_path = Path(db_path_str)
        if not _wipe_path.exists():
            st.warning(f"Nothing to clear — no database at `{_wipe_path}`.")
        else:
            import sqlite3 as _sqlite3
            try:
                with _sqlite3.connect(str(_wipe_path)) as _conn:
                    _conn.execute("DELETE FROM metrics")
                    _conn.execute("DELETE FROM segments")
                    _conn.commit()
                st.success("Cleared all metrics and segments.")
                st.rerun()
            except _sqlite3.OperationalError as err:
                st.error(f"Could not clear database: {err}")

# ---- Header ---------------------------------------------------------------

st.markdown(
    """
    <div class="ga-hero">
      <div class="ga-badge">�</div>
      <div>
        <h1>Gaze Analytics — Signage Engagement</h1>
        <div class="ga-sub">Privacy-preserving audience analytics on Intel CPU / iGPU</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    # All data follows the window selection
    metrics_df = metrics_frame(
        db_path,
        window_minutes=window_minutes,
        current_run_only=_current_run_only,
    )
    segments_df = segments_frame(db_path, limit=50)
    kpis: Kpis = compute_kpis(metrics_df, segments_df)

    # ---- KPI tiles --------------------------------------------------------

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Viewers Now", kpis.viewers_now)
    col2.metric("Attending Now", kpis.attending_now)
    col3.metric("Peak Viewers", kpis.peak_viewers, help="Maximum viewers seen in any single window during the current selection.")
    col4.metric("Attention Rate", f"{kpis.attention_rate}%")
    col5.metric("Avg Dwell (ms)", f"{kpis.avg_dwell_ms:,.0f}")
    col6.metric("Segments", kpis.total_segments)

    st.divider()

    # ---- Trend chart ------------------------------------------------------

    st.subheader("Viewers vs. Attending")
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
        st.subheader("Demographics")
        gt = gender_totals(metrics_df)
        latest = latest_gender(metrics_df)
        # Live snapshot row — shows the current gender split as percentages
        # so it doesn't feel stuck on stale cumulative numbers.
        _snap_total = latest["Male"] + latest["Female"]
        _snap_male_pct = (
            f"{latest['Male'] / _snap_total * 100:.0f}%" if _snap_total else "—"
        )
        _snap_female_pct = (
            f"{latest['Female'] / _snap_total * 100:.0f}%" if _snap_total else "—"
        )
        snap_m, snap_f = st.columns(2)
        snap_m.metric("Male (now)", _snap_male_pct)
        snap_f.metric("Female (now)", _snap_female_pct)

        total_samples = gt["Male"] + gt["Female"]
        if total_samples == 0:
            st.caption(
                "No demographic samples yet. Ensure age/gender is enabled and"
                " someone is on camera."
            )
        else:
            labels = []
            values = []
            colors = []
            _color_map = {"Male": "#636EFA", "Female": "#EF553B"}
            for label in ("Male", "Female"):
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
                height=240,
                showlegend=True,
            )
            st.plotly_chart(gender_fig, width="stretch")
            st.caption(
                f"Cumulative across {total_samples} gender-observed track"
                f"{'s' if total_samples != 1 else ''} in this window."
            )

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
            ]].copy()
            display_df["avg_dwell_ms"] = display_df["avg_dwell_ms"].round(0).astype(int)
            display_df = display_df.rename(columns={
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

