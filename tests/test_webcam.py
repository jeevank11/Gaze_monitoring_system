"""Tests for the webcam capture layer.

We do NOT open the real webcam in CI. These tests validate the module's
contracts using fake ``cv2.VideoCapture`` doubles.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gaze_analytics.capture.webcam import Frame, WebcamOpenError, iter_frames, open_camera
from gaze_analytics.config import Settings


def _fake_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_open_camera_raises_when_device_missing() -> None:
    with patch("gaze_analytics.capture.webcam.cv2.VideoCapture") as vc:
        vc.return_value.isOpened.return_value = False
        with pytest.raises(WebcamOpenError):
            open_camera(Settings(camera_index=42))


def test_open_camera_returns_open_capture() -> None:
    with patch("gaze_analytics.capture.webcam.cv2.VideoCapture") as vc:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.get.return_value = 720
        vc.return_value = instance
        cap = open_camera(Settings(camera_index=0))
        assert cap is instance
        instance.set.assert_called()  # width/height/fps set


def test_iter_frames_yields_frames_and_stops_on_release() -> None:
    """The iterator must yield ``Frame`` objects with increasing indices."""
    with patch("gaze_analytics.capture.webcam.cv2.VideoCapture") as vc:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.get.return_value = 720
        # yield 3 real frames; any further read returns (False, None) which
        # iter_frames treats as a transient grab failure (it retries).
        reads = [
            (True, _fake_frame()),
            (True, _fake_frame()),
            (True, _fake_frame()),
        ]
        call_count = {"n": 0}

        def fake_read() -> tuple[bool, np.ndarray | None]:
            i = call_count["n"]
            call_count["n"] += 1
            if i < len(reads):
                return reads[i]
            return (False, None)

        instance.read.side_effect = fake_read
        vc.return_value = instance

        collected: list[Frame] = []
        # Caller controls termination — matches how main.py's loop works.
        for f in iter_frames(Settings()):
            collected.append(f)
            if len(collected) == 3:
                break

        assert [f.index for f in collected] == [0, 1, 2]
        for f in collected:
            assert isinstance(f, Frame)
            assert f.pixels.dtype == np.uint8
            assert f.pixels.shape == (720, 1280, 3)


def test_frame_is_frozen_dataclass() -> None:
    """A ``Frame`` should be immutable so callers cannot smuggle side data."""
    f = Frame(index=0, timestamp_monotonic=0.0, pixels=_fake_frame())
    with pytest.raises(Exception):  # FrozenInstanceError
        f.index = 1  # type: ignore[misc]
