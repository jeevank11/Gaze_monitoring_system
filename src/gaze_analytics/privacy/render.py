"""Privacy-safe rendering for the preview window.

The preview overlay in :mod:`gaze_analytics.main` is a debug aid. For a
public demo (or a screen recording), the raw face pixels should never leave
the machine on video. This module transforms each face region *inside the
frame buffer* into one of four visibly-privacy-safe shapes:

- ``off``         — no masking (default; used by developers only)
- ``blur``        — heavy Gaussian blur of the face region
- ``pixelate``    — downsample-then-upsample mosaic
- ``silhouette``  — solid fill + bbox outline; original pixels are overwritten

Every mode mutates the caller's frame in place. No new arrays are allocated
that could outlive the frame's lifetime, so the "frames are ephemeral"
contract is preserved.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from gaze_analytics.inference import FaceBBox

PrivacyMode = Literal["off", "blur", "pixelate", "silhouette"]

# Solid grey used by silhouette mode — matches the "no pose" overlay colour.
_SILHOUETTE_FILL = (60, 60, 60)
_SILHOUETTE_OUTLINE = (200, 200, 200)


def apply_privacy_mask(
    frame_bgr: np.ndarray,
    faces: list[FaceBBox],
    mode: PrivacyMode,
) -> None:
    """Mutate ``frame_bgr`` in place so face regions are no longer identifiable."""
    if mode == "off" or not faces:
        return
    h, w = frame_bgr.shape[:2]
    for face in faces:
        x1 = max(0, min(w - 1, face.xmin))
        y1 = max(0, min(h - 1, face.ymin))
        x2 = max(x1 + 1, min(w, face.xmax))
        y2 = max(y1 + 1, min(h, face.ymax))
        region = frame_bgr[y1:y2, x1:x2]
        if region.size == 0:
            continue
        if mode == "blur":
            _apply_blur(region)
        elif mode == "pixelate":
            _apply_pixelate(region)
        elif mode == "silhouette":
            _apply_silhouette(frame_bgr, x1, y1, x2, y2)


def _apply_blur(region: np.ndarray) -> None:
    # Kernel scales with the shorter side so small faces still get fully blurred.
    short = min(region.shape[:2])
    k = max(15, (short // 4) | 1)  # odd, at least 15
    blurred = cv2.GaussianBlur(region, (k, k), 0)
    region[:] = blurred


def _apply_pixelate(region: np.ndarray, blocks: int = 8) -> None:
    rh, rw = region.shape[:2]
    bw = max(1, rw // blocks)
    bh = max(1, rh // blocks)
    small = cv2.resize(region, (bw, bh), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    region[:] = mosaic


def _apply_silhouette(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> None:
    # Overwrite the face region with a solid fill, then draw a bbox outline
    # so the viewer still knows a person is there.
    frame_bgr[y1:y2, x1:x2] = _SILHOUETTE_FILL
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _SILHOUETTE_OUTLINE, 2)
