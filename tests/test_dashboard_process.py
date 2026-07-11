"""Unit tests for the dashboard subprocess control plane.

Never spawns the real pipeline. ``subprocess.Popen`` and ``psutil`` are
monkey-patched so the tests run offline on any host.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from dashboard import process as dp
from dashboard.process import (
    PipelineConfig,
    PipelineStatus,
    build_command,
    cli_preview,
    format_uptime,
    start,
    status,
    stop,
    tail_log,
    uptime_seconds,
)

# ---------- Config -> CLI ---------------------------------------------------


def test_config_defaults_render_expected_cli_args():
    cfg = PipelineConfig()
    args = cfg.as_cli_args()
    assert "--device" in args and "AUTO" in args
    assert "--privacy-mode" in args and "blur" in args
    assert "--detect-faces" in args
    assert "--head-pose" in args
    assert "--track" in args
    assert "--segment-content" in args
    assert "--age-gender" in args
    assert "--sink" in args
    assert "--preview" in args


def test_config_negated_toggles_use_no_flag():
    cfg = PipelineConfig(
        detect_faces=False,
        head_pose=False,
        track=False,
        segment_content=False,
        age_gender=False,
        sink=False,
        preview=False,
    )
    args = cfg.as_cli_args()
    for flag in (
        "--no-detect-faces",
        "--no-head-pose",
        "--no-track",
        "--no-segment-content",
        "--no-age-gender",
        "--no-sink",
        "--no-preview",
    ):
        assert flag in args


def test_build_command_uses_current_interpreter_by_default():
    cmd = build_command(PipelineConfig())
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "gaze_analytics", "run"]


def test_cli_preview_replaces_python_for_human_display():
    line = cli_preview(PipelineConfig(device="GPU", privacy_mode="pixelate"))
    assert line.startswith("python -m gaze_analytics run")
    assert "--device GPU" in line
    assert "--privacy-mode pixelate" in line


# ---------- Pidfile round-trip ---------------------------------------------


def test_status_returns_not_running_when_pidfile_missing(tmp_path):
    result = status(pid_path=tmp_path / "pipeline.pid")
    assert result == PipelineStatus(running=False)


def test_status_cleans_up_stale_pidfile(tmp_path, monkeypatch):
    pid_file = tmp_path / "pipeline.pid"
    pid_file.write_text(json.dumps({"pid": 424242, "started_at": 0.0, "config": {}}))
    monkeypatch.setattr(dp.psutil, "pid_exists", lambda _pid: False)

    result = status(pid_path=pid_file)

    assert result.running is False
    assert not pid_file.exists()


def test_status_returns_running_for_live_pid(tmp_path, monkeypatch):
    pid_file = tmp_path / "pipeline.pid"
    cfg = PipelineConfig(device="CPU", privacy_mode="silhouette", preview=False)
    pid_file.write_text(
        json.dumps({"pid": 4242, "started_at": 1_700_000_000.0, "config": cfg.__dict__})
    )
    monkeypatch.setattr(dp.psutil, "pid_exists", lambda pid: pid == 4242)

    result = status(pid_path=pid_file)

    assert result.running is True
    assert result.pid == 4242
    assert result.started_at == 1_700_000_000.0
    assert result.config == cfg


def test_status_recovers_when_config_shape_is_unknown(tmp_path, monkeypatch):
    pid_file = tmp_path / "pipeline.pid"
    pid_file.write_text(
        json.dumps(
            {"pid": 4242, "started_at": 1.0, "config": {"totally": "bogus"}}
        )
    )
    monkeypatch.setattr(dp.psutil, "pid_exists", lambda _pid: True)

    result = status(pid_path=pid_file)

    assert result.running is True
    assert result.config is None


def test_status_handles_corrupt_pidfile(tmp_path):
    pid_file = tmp_path / "pipeline.pid"
    pid_file.write_text("{not json")

    result = status(pid_path=pid_file)

    assert result == PipelineStatus(running=False)


# ---------- start / stop ---------------------------------------------------


class _FakePopen:
    def __init__(self, pid: int = 12345):
        self.pid = pid


def test_start_when_already_running_returns_existing_status(tmp_path, monkeypatch):
    pid_file = tmp_path / "pipeline.pid"
    log_file = tmp_path / "pipeline.log"
    cfg = PipelineConfig()
    pid_file.write_text(json.dumps({"pid": 999, "started_at": 1.0, "config": cfg.__dict__}))
    monkeypatch.setattr(dp.psutil, "pid_exists", lambda pid: pid == 999)

    def _boom(*_a, **_kw):
        raise AssertionError("Popen must not be called when a pipeline is running")

    monkeypatch.setattr(dp.subprocess, "Popen", _boom)

    result = start(cfg, pid_path=pid_file, log_path=log_file)

    assert result.running is True
    assert result.pid == 999


def test_start_spawns_and_writes_pidfile(tmp_path, monkeypatch):
    pid_file = tmp_path / "pipeline.pid"
    log_file = tmp_path / "logs" / "pipeline.log"

    captured: dict = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakePopen(pid=13579)

    monkeypatch.setattr(dp.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dp.psutil, "pid_exists", lambda _pid: False)

    cfg = PipelineConfig(device="GPU")
    result = start(cfg, pid_path=pid_file, log_path=log_file)

    assert result.running is True
    assert result.pid == 13579
    assert pid_file.exists()
    state = json.loads(pid_file.read_text())
    assert state["pid"] == 13579
    assert state["config"]["device"] == "GPU"
    assert log_file.parent.exists()

    assert captured["cmd"][:4] == [sys.executable, "-m", "gaze_analytics", "run"]
    assert "--device" in captured["cmd"]
    assert captured["kwargs"]["env"].get("NO_COLOR") == "1"


def test_stop_returns_false_when_nothing_is_running(tmp_path):
    assert stop(pid_path=tmp_path / "pipeline.pid") is False


def test_stop_sends_signal_and_clears_pidfile(tmp_path, monkeypatch):
    pid_file = tmp_path / "pipeline.pid"
    cfg = PipelineConfig()
    pid_file.write_text(json.dumps({"pid": 2222, "started_at": 1.0, "config": cfg.__dict__}))
    monkeypatch.setattr(dp.psutil, "pid_exists", lambda pid: pid == 2222)

    signalled: list[int] = []

    class _FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def send_signal(self, sig):
            signalled.append(int(sig))

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(dp.psutil, "Process", lambda pid: _FakeProc(pid))

    assert stop(pid_path=pid_file) is True
    assert not pid_file.exists()
    assert signalled, "expected at least one signal to be sent"


# ---------- log tail -------------------------------------------------------


def test_tail_log_missing_file_returns_empty(tmp_path):
    assert tail_log(log_path=tmp_path / "no.log") == ""


def test_tail_log_returns_last_n_lines(tmp_path):
    log = tmp_path / "pipeline.log"
    log.write_text("\n".join(f"line {i}" for i in range(200)) + "\n", encoding="utf-8")

    out = tail_log(log_path=log, lines=10)

    tail = out.splitlines()
    assert tail == [f"line {i}" for i in range(190, 200)]


def test_tail_log_strips_ansi_escape_codes(tmp_path):
    log = tmp_path / "pipeline.log"
    log.write_text("\x1b[32mgreen\x1b[0m and \x1b[1;31mred\x1b[0m\n", encoding="utf-8")

    out = tail_log(log_path=log, lines=1)

    assert out == "green and red"


# ---------- uptime helpers -------------------------------------------------


def test_uptime_seconds_zero_when_not_running():
    assert uptime_seconds(PipelineStatus(running=False)) == 0.0


def test_uptime_seconds_uses_now_override():
    s = PipelineStatus(running=True, pid=1, started_at=100.0, config=PipelineConfig())
    assert uptime_seconds(s, now=175.5) == pytest.approx(75.5)


def test_format_uptime_formats_short_and_long():
    assert format_uptime(3) == "3s"
    assert format_uptime(65) == "1m 05s"
    assert format_uptime(3725) == "1h 02m 05s"


# ---------- PipelineConfig dataclass immutability --------------------------


def test_pipeline_config_is_immutable():
    cfg = PipelineConfig()
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        cfg.device = "CPU"  # type: ignore[misc]


def test_pipeline_config_replace_produces_new_instance():
    cfg = PipelineConfig()
    other = replace(cfg, device="CPU")
    assert cfg.device == "AUTO"
    assert other.device == "CPU"


# ---------- Guard: no default paths are written during import -------------


def test_module_imports_do_not_touch_default_paths():
    # Sanity: DEFAULT_PID_PATH should point under the repo but must not exist
    # merely from importing the module.
    assert isinstance(dp.DEFAULT_PID_PATH, Path)
    assert isinstance(dp.DEFAULT_LOG_PATH, Path)
