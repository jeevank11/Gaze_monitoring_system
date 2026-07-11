"""Webcam capture loop.

Frames are yielded to the caller and then dropped. This module never writes
a frame to disk, never keeps a history buffer, and never sends a frame off
the process. Any code path that violates this is a bug.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from gaze_analytics.config import Settings, settings

log = logging.getLogger(__name__)


_BACKEND_MAP: dict[str, int] = {
    "AUTO": cv2.CAP_ANY,
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
    "V4L2": cv2.CAP_V4L2,
}


@dataclass(frozen=True)
class Frame:
    """A single captured frame.

    ``pixels`` is a view into a buffer that will be overwritten on the next
    grab. Consumers must finish using it (or copy the tiny region they
    actually need) before the next iteration.
    """

    index: int
    timestamp_monotonic: float
    pixels: np.ndarray  # BGR uint8 (H, W, 3)


class WebcamOpenError(RuntimeError):
    """Raised when the webcam cannot be opened."""


def open_camera(cfg: Settings = settings) -> cv2.VideoCapture:
    """Open the webcam using the configured index and backend.

    Raises ``WebcamOpenError`` if the device cannot be opened.
    """
    backend = _BACKEND_MAP.get(cfg.camera_backend, cv2.CAP_ANY)
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    if not cap.isOpened():
        raise WebcamOpenError(
            f"Could not open camera index {cfg.camera_index} with backend "
            f"{cfg.camera_backend}. Close other apps holding the camera "
            "(Teams / Zoom / OBS) and check Windows camera privacy settings."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(
        "Camera opened: index=%s backend=%s resolution=%dx%d requested_fps=%d",
        cfg.camera_index,
        cfg.camera_backend,
        actual_w,
        actual_h,
        cfg.target_fps,
    )
    return cap


def iter_frames(cfg: Settings = settings) -> Iterator[Frame]:
    """Yield frames from the webcam until stopped.

    The yielded ``Frame.pixels`` buffer is invalid after the next iteration.
    """
    cap = open_camera(cfg)
    index = 0
    try:
        while True:
            ok, pixels = cap.read()
            if not ok or pixels is None:
                log.warning("Frame grab failed at index=%d; retrying", index)
                # brief backoff so we don't spin at 100% CPU on a dead cam
                time.sleep(0.05)
                continue
            yield Frame(
                index=index,
                timestamp_monotonic=time.monotonic(),
                pixels=pixels,
            )
            index += 1
            # explicit del is defensive — makes the "no persistence" intent
            # visible in the code, even though Python would GC it anyway.
            del pixels
    finally:
        cap.release()
        log.info("Camera released after %d frames", index)
