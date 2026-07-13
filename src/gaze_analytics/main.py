"""CLI entry point for the gaze analytics pipeline.

Sprint 1: capture. Sprint 2: face detection. Sprint 3: head-pose + attention.
Sprint 4: IoU + Kalman tracker + pHash content segmenter.
Sprint 5: age/gender aggregate inference + rolling aggregator + SQLite sink.
Sprint 7: privacy-safe render modes for the preview window.

Run with:
    python -m gaze_analytics run --preview --segment-content --age-gender --sink
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Annotated

import cv2
import typer
from rich.console import Console
from rich.logging import RichHandler

from gaze_analytics import __version__
from gaze_analytics.capture.screen import ScreenGrabber
from gaze_analytics.capture.webcam import iter_frames
from gaze_analytics.config import settings
from gaze_analytics.content import ContentSegment, ContentSegmenter
from gaze_analytics.engagement import AttentionSmoother, RollingAggregator, is_attending
from gaze_analytics.inference import (
    AgeGenderEstimator,
    FaceBBox,
    FaceDetector,
    GazeEstimator,
    GazeVector,
    HeadPose,
    HeadPoseEstimator,
)
from gaze_analytics.privacy import PrivacyMode, apply_privacy_mask
from gaze_analytics.storage import SqliteSink
from gaze_analytics.tracker import IoUTracker, TrackedFace

console = Console()
app = typer.Typer(add_completion=False, help="Privacy-preserving gaze analytics.")

_COLOR_ATTENDING = (0, 200, 0)  # green
_COLOR_DISTRACTED = (0, 200, 220)  # amber
_COLOR_NO_POSE = (128, 128, 128)  # grey


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _draw_face_annotation(
    frame_bgr,
    bbox: FaceBBox,
    pose: HeadPose | None,
    attending: bool,
    track_id: int | None,
    dwell_seconds: float | None,
) -> None:
    """In-place bbox + pose label overlay. No pixels leave the frame buffer."""
    active = _COLOR_ATTENDING if attending else _COLOR_DISTRACTED
    color = _COLOR_NO_POSE if pose is None else active
    cv2.rectangle(frame_bgr, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color, 2)

    parts: list[str] = []
    if track_id is not None:
        parts.append(f"#{track_id}")
    if pose is None:
        parts.append(f"face {bbox.score:.2f}")
    else:
        state = "ATTENDING" if attending else "away"
        parts.append(f"{state} y={pose.yaw:+.0f} p={pose.pitch:+.0f}")
    if dwell_seconds is not None and dwell_seconds >= 0.1:
        parts.append(f"{dwell_seconds:.1f}s")
    label = " ".join(parts)

    cv2.putText(
        frame_bgr,
        label,
        (bbox.xmin, max(0, bbox.ymin - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _match_track_to_detection(
    tracks: list[TrackedFace],
    bbox: FaceBBox,
) -> TrackedFace | None:
    """Best-IoU match between the emitted track list and a detection bbox."""
    best: TrackedFace | None = None
    best_iou = 0.0
    for trk in tracks:
        tb = trk.bbox
        xa1 = max(tb.xmin, bbox.xmin)
        ya1 = max(tb.ymin, bbox.ymin)
        xa2 = min(tb.xmax, bbox.xmax)
        ya2 = min(tb.ymax, bbox.ymax)
        inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        if inter == 0:
            continue
        area_t = (tb.xmax - tb.xmin) * (tb.ymax - tb.ymin)
        area_d = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
        iou = inter / float(area_t + area_d - inter)
        if iou > best_iou:
            best_iou = iou
            best = trk
    return best


@app.command()
def run(
    device: Annotated[str, typer.Option(help="OpenVINO device: AUTO, CPU, GPU, NPU")] = "AUTO",
    camera: Annotated[int, typer.Option(help="Webcam index")] = 0,
    video: Annotated[
        Path | None,
        typer.Option(help="Local video file to use in place of the webcam (for testing)"),
    ] = None,
    preview: Annotated[bool, typer.Option(help="Show debug preview window")] = False,
    headless: Annotated[bool, typer.Option(help="No window; log FPS periodically")] = False,
    detect_faces: Annotated[
        bool, typer.Option(help="Run face detection (Sprint 2)")
    ] = True,
    head_pose: Annotated[
        bool, typer.Option(help="Run head-pose + attention gating (Sprint 3)")
    ] = True,
    track: Annotated[
        bool, typer.Option(help="Run IoU + Kalman tracker for stable IDs (Sprint 4)")
    ] = True,
    segment_content: Annotated[
        bool, typer.Option(help="Also grab the screen and detect content changes (Sprint 4)")
    ] = False,
    age_gender: Annotated[
        bool, typer.Option(help="Run age/gender aggregate inference (Sprint 5)")
    ] = False,
    gaze: Annotated[
        bool, typer.Option(help="Run gaze estimation for precise attention (near-tier)")
    ] = False,
    sink: Annotated[
        bool, typer.Option(help="Write aggregate rows to SQLite (Sprint 5)")
    ] = False,
    privacy_mode: Annotated[
        str,
        typer.Option(
            help="Preview mask: off | blur | pixelate | silhouette (Sprint 7)",
        ),
    ] = "blur",
    min_face_height: Annotated[
        int,
        typer.Option(
            help=(
                "Minimum face height in pixels to accept as a viewer. "
                "Default 80 (signage-safe); lower to 30-40 for recorded-video testing."
            ),
        ),
    ] = 80,
    face_confidence: Annotated[
        float,
        typer.Option(
            help=(
                "Face-detector confidence floor (0.0-1.0). Default 0.7 (single viewer). "
                "Lower to ~0.4 for crowds so more small/partial faces pass through."
            ),
        ),
    ] = 0.7,
    tiled_detection: Annotated[
        bool,
        typer.Option(
            help=(
                "Slice the frame into an NxN grid and run the face detector on each tile "
                "before merging with NMS. Greatly improves small-face recall in crowds "
                "at ~NxN detector cost."
            ),
        ),
    ] = False,
    tile_grid: Annotated[
        int,
        typer.Option(
            help="Grid size for --tiled-detection (e.g. 2 = 2x2 tiles, 3 = 3x3).",
        ),
    ] = 2,
    log_level: Annotated[str, typer.Option(help="DEBUG / INFO / WARNING / ERROR")] = "INFO",
) -> None:
    """Start the capture + inference loop."""
    _configure_logging(log_level)
    log = logging.getLogger("gaze_analytics")

    # On Windows, the dashboard's Stop button delivers CTRL_BREAK_EVENT, which
    # Python surfaces as SIGBREAK — not SIGINT — so we install a handler that
    # raises KeyboardInterrupt to trigger the same clean shutdown path as Ctrl+C.
    if hasattr(signal, "SIGBREAK"):
        def _on_break(_signum: int, _frame: object) -> None:  # pragma: no cover
            raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK, _on_break)  # type: ignore[attr-defined]

    if privacy_mode not in {"off", "blur", "pixelate", "silhouette"}:
        raise typer.BadParameter(
            f"privacy_mode must be one of off|blur|pixelate|silhouette, got {privacy_mode!r}"
        )
    privacy: PrivacyMode = privacy_mode  # type: ignore[assignment]

    settings.device = device  # type: ignore[assignment]
    settings.camera_index = camera
    settings.video_file = video
    settings.preview = preview
    settings.headless = headless
    settings.min_face_height_px = min_face_height
    settings.face_confidence_threshold = face_confidence
    settings.face_tiled_detection = tiled_detection
    settings.face_tile_grid = max(1, int(tile_grid))

    log.info("gaze-analytics v%s starting", __version__)
    log.info(
        "config: device=%s camera=%d preview=%s headless=%s faces=%s "
        "head_pose=%s track=%s segment_content=%s age_gender=%s sink=%s "
        "privacy_mode=%s",
        settings.device,
        settings.camera_index,
        settings.preview,
        settings.headless,
        detect_faces,
        head_pose,
        track,
        segment_content,
        age_gender,
        sink,
        privacy,
    )

    face_detector: FaceDetector | None = FaceDetector(settings) if detect_faces else None
    pose_estimator: HeadPoseEstimator | None = (
        HeadPoseEstimator(settings) if (detect_faces and head_pose) else None
    )
    gaze_estimator: GazeEstimator | None = (
        GazeEstimator(settings) if (detect_faces and head_pose and gaze) else None
    )
    tracker: IoUTracker | None = IoUTracker() if (detect_faces and track) else None
    segmenter: ContentSegmenter | None = ContentSegmenter() if segment_content else None
    screen: ScreenGrabber | None = ScreenGrabber() if segment_content else None
    age_gender_estimator: AgeGenderEstimator | None = (
        AgeGenderEstimator(settings) if (detect_faces and age_gender) else None
    )
    aggregator: RollingAggregator | None = RollingAggregator(settings) if sink else None
    db_sink: SqliteSink | None = SqliteSink(settings.sqlite_path) if sink else None
    attention_smoother = AttentionSmoother(settings.attention_away_frames)

    # Screen capture cadence: one grab every N webcam frames.
    screen_period = (
        max(1, int(settings.target_fps / max(0.1, settings.screen_capture_hz)))
        if segment_content
        else 0
    )
    last_segment_id: int | None = None
    active_segment_id: str | None = None
    active_segment: ContentSegment | None = None

    frames_seen = 0
    faces_last = 0
    attending_last = 0
    tracks_last = 0
    started = time.monotonic()
    last_report = started

    try:
        for frame in iter_frames(settings):
            frames_seen += 1
            now = time.monotonic()

            faces: list[FaceBBox] = []
            tracks: list[TrackedFace] = []
            attending_count = 0
            attending_ids: set[int] = set()
            gender_by_id: dict[int, str] = {}

            if face_detector is not None:
                faces = face_detector.detect(frame.pixels)
                if tracker is not None:
                    tracks = tracker.update(faces, timestamp=frame.timestamp_monotonic)

                # Pass 1: run inference on the raw frame (pose + age/gender need
                # unmodified face pixels). Cache per-face results for pass 2.
                per_face_state: list[
                    tuple[FaceBBox, HeadPose | None, bool, TrackedFace | None]
                ] = []
                for f in faces:
                    pose: HeadPose | None = None
                    gaze_vec: GazeVector | None = None
                    attends = False
                    crop = frame.pixels[f.ymin : f.ymax, f.xmin : f.xmax]
                    if pose_estimator is not None and crop.size > 0:
                        pose = pose_estimator.estimate(crop)
                        # Near-tier: if gaze model is loaded, refine attention
                        if gaze_estimator is not None:
                            gaze_vec = gaze_estimator.estimate(crop, pose)
                        attends = is_attending(pose, settings, gaze=gaze_vec)

                    matched = _match_track_to_detection(tracks, f) if tracker else None
                    # Apply per-track hysteresis so a brief head turn doesn't flip to "away".
                    if matched is not None:
                        attends = attention_smoother.update(matched.track_id, attends)
                    if attends:
                        attending_count += 1
                        if matched is not None:
                            attending_ids.add(matched.track_id)

                    if (
                        age_gender_estimator is not None
                        and matched is not None
                        and crop.size > 0
                    ):
                        ag = age_gender_estimator.estimate(crop)
                        gender_by_id[matched.track_id] = ag.gender

                    per_face_state.append((f, pose, attends, matched))

                # Pass 2: apply the privacy mask (mutates the frame buffer) and
                # then draw the debug overlay on top of the now-safe pixels.
                if preview:
                    apply_privacy_mask(frame.pixels, faces, privacy)
                    for f, pose, attends, matched in per_face_state:
                        _draw_face_annotation(
                            frame.pixels,
                            f,
                            pose,
                            attends,
                            track_id=matched.track_id if matched else None,
                            dwell_seconds=(matched.dwell_seconds if matched else None),
                        )

                faces_last = len(faces)
                attending_last = attending_count
                tracks_last = len(tracks)
                # Drop smoother state for tracks that disappeared this frame.
                attention_smoother.gc({t.track_id for t in tracks})

            if (
                segmenter is not None
                and screen is not None
                and screen_period > 0
                and frames_seen % screen_period == 0
            ):
                screen_frame = screen.grab()
                if screen_frame is None:
                    # Transient GDI failure — skip segmentation this iteration.
                    # ScreenGrabber already logged a warning.
                    pass
                else:
                    seg: ContentSegment = segmenter.update(
                        screen_frame,
                        timestamp=frame.timestamp_monotonic,
                    )
                    if seg.segment_id != last_segment_id:
                        log.info(
                            "content segment #%d started phash=%016x t=%.2fs",
                            seg.segment_id,
                            seg.phash,
                            seg.first_seen_ts,
                        )
                        # Close previous segment in the sink, start the new one.
                        if db_sink is not None:
                            if active_segment is not None and active_segment_id is not None:
                                db_sink.close_segment(active_segment, active_segment_id)
                            active_segment_id = db_sink.upsert_segment_start(seg)
                        active_segment = seg
                        last_segment_id = seg.segment_id
                    elif active_segment is not None:
                        # Keep the reference fresh so close_segment picks up the final ts.
                        active_segment = seg

            if aggregator is not None:
                aggregator.record(
                    tracks,
                    attending_ids,
                    gender_by_id,  # type: ignore[arg-type]
                    timestamp=frame.timestamp_monotonic,
                    face_count=len(faces),
                )
                row = aggregator.flush_if_due(
                    timestamp=frame.timestamp_monotonic,
                    segment_id=active_segment_id,
                )
                if row is not None and db_sink is not None:
                    db_sink.write_metrics(row)
                    log.info(
                        "aggregate window closed: viewers=%d attending=%d "
                        "avg_dwell_ms=%d M=%d F=%d segment=%s",
                        row.viewers,
                        row.attending,
                        row.avg_dwell_ms,
                        row.male_count,
                        row.female_count,
                        row.segment_id,
                    )

            if preview:
                cv2.imshow("gaze-analytics (Sprint 8: release)", frame.pixels)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if now - last_report >= 1.0:
                fps = frames_seen / (now - started)
                log.info(
                    "frames=%d elapsed=%.1fs fps=%.1f faces=%d tracks=%d attending=%d",
                    frames_seen,
                    now - started,
                    fps,
                    faces_last,
                    tracks_last,
                    attending_last,
                )
                last_report = now
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        # Flush any partial window before shutting the sink.
        if aggregator is not None and db_sink is not None:
            final = aggregator.flush(timestamp=time.monotonic(), segment_id=active_segment_id)
            if final is not None:
                db_sink.write_metrics(final)
        if (
            db_sink is not None
            and active_segment is not None
            and active_segment_id is not None
        ):
            db_sink.close_segment(active_segment, active_segment_id)
        if db_sink is not None:
            db_sink.close()
        if preview:
            cv2.destroyAllWindows()
        if screen is not None:
            screen.close()
        elapsed = time.monotonic() - started
        fps = frames_seen / elapsed if elapsed > 0 else 0.0
        log.info(
            "Shutdown clean: frames=%d elapsed=%.1fs avg_fps=%.2f",
            frames_seen,
            elapsed,
            fps,
        )


@app.command()
def version() -> None:
    """Print the version and exit."""
    console.print(f"gaze-analytics [bold cyan]v{__version__}[/]")


if __name__ == "__main__":
    app()
