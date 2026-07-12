"""Subprocess lifecycle helpers for the dashboard control plane.

The dashboard spawns the gaze-analytics pipeline as a separate process,
persists its PID + config to disk so we can adopt an already-running pipeline
after the Streamlit server restarts, and tails its log file for status.

This module never sees frames. It only manages PIDs, exit codes, and the
line-oriented log file written by the pipeline's stdlib logger.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PID_PATH = REPO_ROOT / "data" / "pipeline.pid"
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "pipeline.log"

Device = Literal["AUTO", "CPU", "GPU"]
PrivacyMode = Literal["off", "blur", "pixelate", "silhouette"]

# strip ANSI escape sequences from rich-formatted log lines
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class PipelineConfig:
    """User-facing pipeline configuration; mirrors the ``run`` CLI flags."""

    device: Device = "AUTO"
    privacy_mode: PrivacyMode = "blur"
    detect_faces: bool = True
    head_pose: bool = True
    track: bool = True
    segment_content: bool = True
    age_gender: bool = True
    gaze: bool = True
    sink: bool = True
    preview: bool = True
    log_level: str = "INFO"
    # Optional local video file to feed in place of the webcam (for testing).
    video_file: str | None = None
    # Minimum face height in pixels to accept as a viewer. Lower this when
    # testing with recorded videos where faces appear smaller than the 80 px
    # signage default (see Settings.min_face_height_px).
    min_face_height_px: int = 80
    # Face detector confidence floor. Default 0.7 is safe for single viewers;
    # drop to ~0.4 for crowds where many small/partial faces score lower.
    face_confidence_threshold: float = 0.7
    # Tiled/sliced face detection: split the frame into a grid and run the
    # detector on each tile. Big small-face recall win for crowds.
    tiled_detection: bool = False
    tile_grid: int = 2

    def as_cli_args(self) -> list[str]:
        args: list[str] = [
            "--device", self.device,
            "--privacy-mode", self.privacy_mode,
            "--log-level", self.log_level,
            "--min-face-height", str(self.min_face_height_px),
            "--face-confidence", f"{self.face_confidence_threshold:.3f}",
            "--tile-grid", str(self.tile_grid),
        ]
        toggles: list[tuple[str, str]] = [
            ("detect_faces", "detect-faces"),
            ("head_pose", "head-pose"),
            ("track", "track"),
            ("segment_content", "segment-content"),
            ("age_gender", "age-gender"),
            ("gaze", "gaze"),
            ("sink", "sink"),
            ("preview", "preview"),
            ("tiled_detection", "tiled-detection"),
        ]
        for attr, flag in toggles:
            args.append(f"--{flag}" if getattr(self, attr) else f"--no-{flag}")
        if self.video_file:
            args.extend(["--video", self.video_file])
        return args


@dataclass(frozen=True)
class PipelineStatus:
    """Snapshot of what the dashboard knows about the pipeline process."""

    running: bool
    pid: int | None = None
    started_at: float | None = None
    config: PipelineConfig | None = None


def build_command(cfg: PipelineConfig, python: str | None = None) -> list[str]:
    """Return the argv for launching the pipeline via the current interpreter."""
    return [python or sys.executable, "-m", "gaze_analytics", "run", *cfg.as_cli_args()]


def cli_preview(cfg: PipelineConfig) -> str:
    """Human-friendly single-line rendering of the equivalent CLI invocation."""
    return " ".join(build_command(cfg, python="python"))


def _write_state(pid: int, started_at: float, cfg: PipelineConfig, pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(
        json.dumps(
            {"pid": pid, "started_at": started_at, "config": asdict(cfg)},
            indent=2,
        )
    )


def _read_state(pid_path: Path) -> dict | None:
    if not pid_path.exists():
        return None
    try:
        return json.loads(pid_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def status(pid_path: Path = DEFAULT_PID_PATH) -> PipelineStatus:
    """Read the pidfile and check whether that PID is still alive."""
    state = _read_state(pid_path)
    if state is None:
        return PipelineStatus(running=False)
    try:
        pid = int(state["pid"])
    except (KeyError, TypeError, ValueError):
        pid_path.unlink(missing_ok=True)
        return PipelineStatus(running=False)
    if not psutil.pid_exists(pid):
        pid_path.unlink(missing_ok=True)
        return PipelineStatus(running=False)
    try:
        cfg = PipelineConfig(**state["config"])
    except (TypeError, KeyError, ValueError):
        cfg = None
    return PipelineStatus(
        running=True,
        pid=pid,
        started_at=float(state.get("started_at") or 0.0),
        config=cfg,
    )


def start(
    cfg: PipelineConfig,
    *,
    pid_path: Path = DEFAULT_PID_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> PipelineStatus:
    """Spawn the pipeline detached; return the resulting status.

    If a pipeline is already running (according to the pidfile), returns its
    existing status without launching a second one.
    """
    existing = status(pid_path)
    if existing.running:
        return existing

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    cmd = build_command(cfg)

    env = dict(os.environ)
    env.setdefault("NO_COLOR", "1")  # strip rich ANSI codes from the log file

    kwargs: dict = {
        "stdout": log_file,
        "stderr": log_file,
        "stdin": subprocess.DEVNULL,
        "cwd": str(REPO_ROOT),
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        )
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)  # noqa: S603 - argv is fully controlled
    started_at = time.time()
    _write_state(proc.pid, started_at, cfg, pid_path)
    return PipelineStatus(running=True, pid=proc.pid, started_at=started_at, config=cfg)


def stop(*, pid_path: Path = DEFAULT_PID_PATH, timeout: float = 5.0) -> bool:
    """Signal the running pipeline to shut down cleanly. Returns True on hit."""
    cur = status(pid_path)
    if not cur.running or cur.pid is None:
        return False
    try:
        proc = psutil.Process(cur.pid)
    except psutil.NoSuchProcess:
        pid_path.unlink(missing_ok=True)
        return False
    # On Windows the venv python.exe is a shim that spawns a second interpreter,
    # so we must signal / kill the whole process tree, not just the direct child.
    try:
        children = proc.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    procs = [proc, *children]

    # Best-effort graceful shutdown. Any failure here (WinError 87 when the
    # process wasn't launched in its own console group, NoSuchProcess races,
    # etc.) falls through to the force-kill below rather than propagating.
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            for p in procs:
                try:
                    p.send_signal(signal.SIGINT)
                except psutil.NoSuchProcess:
                    pass
        _gone, alive = psutil.wait_procs(procs, timeout=timeout)
    except (OSError, psutil.Error):
        alive = [p for p in procs if p.is_running()]

    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        psutil.wait_procs(alive, timeout=2.0)
    except psutil.Error:
        pass
    pid_path.unlink(missing_ok=True)
    return True


def tail_log(log_path: Path = DEFAULT_LOG_PATH, lines: int = 50) -> str:
    """Return the last ``lines`` lines of the pipeline log, ANSI-stripped."""
    if not log_path.exists():
        return ""
    size = log_path.stat().st_size
    read = min(size, 65536)
    with log_path.open("rb") as f:
        f.seek(size - read)
        chunk = f.read(read).decode("utf-8", errors="replace")
    text = _ANSI_RE.sub("", chunk)
    return "\n".join(text.splitlines()[-lines:])


def uptime_seconds(status_: PipelineStatus, *, now: float | None = None) -> float:
    """Seconds since the pipeline was launched, or 0.0 if not running."""
    if not status_.running or status_.started_at is None:
        return 0.0
    return max(0.0, (now if now is not None else time.time()) - status_.started_at)


def format_uptime(seconds: float) -> str:
    """`123` -> `2m 03s`, `4321` -> `1h 12m 01s`."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
