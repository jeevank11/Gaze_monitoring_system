"""Screen capture via ``mss`` — grabs the primary monitor as BGR ndarray.

Like :mod:`gaze_analytics.capture.webcam`, this producer never persists a
frame. Each grab overwrites the previous one.
"""

from __future__ import annotations

import contextlib
import logging

import cv2
import numpy as np
from mss import mss
from mss.exception import ScreenShotError

log = logging.getLogger(__name__)


class ScreenGrabber:
    """Thin wrapper around ``mss`` for BGR full-screen grabs.

    Windows GDI ``BitBlt`` occasionally fails transiently on display
    sleep/wake, DPI change, RDP session switch, or monitor topology
    change. :meth:`grab` catches those, attempts to reinitialize the
    underlying ``mss`` handle once, and returns ``None`` on repeated
    failure so the caller can skip this iteration instead of crashing.
    """

    def __init__(self, monitor_index: int = 1) -> None:
        # ``mss`` uses index 0 for "all monitors", 1..N for individual screens.
        self._monitor_index = monitor_index
        self._sct = mss()
        self._monitor = self._sct.monitors[monitor_index]
        self._consecutive_failures = 0
        log.info(
            "ScreenGrabber ready: monitor=%d resolution=%dx%d",
            monitor_index,
            self._monitor["width"],
            self._monitor["height"],
        )

    def _reinit(self) -> None:
        """Recreate the underlying ``mss`` handle after a GDI failure."""
        with contextlib.suppress(ScreenShotError, OSError, RuntimeError):
            self._sct.close()
        self._sct = mss()
        self._monitor = self._sct.monitors[self._monitor_index]

    def grab(self) -> np.ndarray | None:
        """Return a fresh BGR ndarray of the monitor, or ``None`` on transient failure."""
        try:
            raw = np.asarray(self._sct.grab(self._monitor))
        except ScreenShotError as err:
            log.warning("screen grab failed (%s); reinitializing mss handle", err)
            try:
                self._reinit()
                raw = np.asarray(self._sct.grab(self._monitor))
            except ScreenShotError as err2:
                self._consecutive_failures += 1
                log.warning(
                    "screen grab still failing after reinit (%s); "
                    "skipping this iteration (consecutive_failures=%d)",
                    err2,
                    self._consecutive_failures,
                )
                return None
        self._consecutive_failures = 0
        # mss returns BGRA; strip alpha and keep BGR (what OpenCV expects).
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._sct.close()


def list_monitors() -> list[dict]:
    """Return a list of available monitors as reported by mss.

    The first element (index 0) is the union of all monitors ("virtual screen");
    elements 1..N are individual physical displays. Each dict has integer
    ``left``, ``top``, ``width``, ``height`` keys.
    """
    try:
        with mss() as sct:
            return [dict(m) for m in sct.monitors]
    except (ScreenShotError, OSError, RuntimeError) as err:
        log.warning("failed to enumerate monitors: %s", err)
        return []

