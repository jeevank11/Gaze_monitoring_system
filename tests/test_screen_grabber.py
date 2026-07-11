"""Tests for ``ScreenGrabber`` — verifies transient BitBlt failures don't crash."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mss.exception import ScreenShotError


@pytest.fixture(autouse=True)
def _fake_mss():
    """Patch ``mss()`` so tests don't touch the real desktop."""
    with patch("gaze_analytics.capture.screen.mss") as mss_ctor:
        instance = MagicMock()
        instance.monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # all monitors
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # primary
        ]
        mss_ctor.return_value = instance
        yield mss_ctor, instance


def _fake_bgra(w: int = 1920, h: int = 1080) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def test_grab_returns_bgr_ndarray(_fake_mss):
    from gaze_analytics.capture.screen import ScreenGrabber

    _, instance = _fake_mss
    instance.grab.return_value = _fake_bgra()

    g = ScreenGrabber(monitor_index=1)
    frame = g.grab()

    assert frame is not None
    assert frame.shape == (1080, 1920, 3)  # BGR
    assert frame.dtype == np.uint8


def test_transient_bitblt_failure_reinits_and_recovers(_fake_mss):
    """First grab fails, second (after reinit) succeeds -> returns valid frame."""
    from gaze_analytics.capture.screen import ScreenGrabber

    mss_ctor, first_instance = _fake_mss
    first_instance.grab.side_effect = ScreenShotError("BitBlt failed")

    # After reinit, mss() returns a new instance whose grab() succeeds.
    second_instance = MagicMock()
    second_instance.monitors = first_instance.monitors
    second_instance.grab.return_value = _fake_bgra()

    g = ScreenGrabber(monitor_index=1)
    # Constructor already consumed one mss() call; next call yields the healthy one.
    mss_ctor.return_value = second_instance

    frame = g.grab()

    assert frame is not None
    assert frame.shape == (1080, 1920, 3)
    assert g._consecutive_failures == 0


def test_repeated_bitblt_failure_returns_none(_fake_mss):
    """If reinit also fails, grab() returns None rather than raising."""
    from gaze_analytics.capture.screen import ScreenGrabber

    mss_ctor, first_instance = _fake_mss
    first_instance.grab.side_effect = ScreenShotError("BitBlt failed")

    second_instance = MagicMock()
    second_instance.monitors = first_instance.monitors
    second_instance.grab.side_effect = ScreenShotError("still failing")
    mss_ctor.return_value = second_instance

    g = ScreenGrabber(monitor_index=1)
    frame = g.grab()

    assert frame is None
    assert g._consecutive_failures == 1


def test_consecutive_failure_counter_resets_on_success(_fake_mss):
    from gaze_analytics.capture.screen import ScreenGrabber

    mss_ctor, first_instance = _fake_mss
    # Fail once, reinit fails once -> None (counter=1).
    first_instance.grab.side_effect = ScreenShotError("boom")
    second_instance = MagicMock()
    second_instance.monitors = first_instance.monitors
    second_instance.grab.side_effect = ScreenShotError("still boom")
    mss_ctor.return_value = second_instance

    g = ScreenGrabber(monitor_index=1)
    assert g.grab() is None
    assert g._consecutive_failures == 1

    # Next call: primary grab succeeds -> counter resets.
    second_instance.grab.side_effect = None
    second_instance.grab.return_value = _fake_bgra()
    g._sct = second_instance  # simulate the reinit that landed us on second_instance

    frame = g.grab()
    assert frame is not None
    assert g._consecutive_failures == 0
