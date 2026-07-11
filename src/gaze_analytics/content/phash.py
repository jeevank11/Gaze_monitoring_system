"""Perceptual hashing for content-change detection.

We use ``ImageHash.phash`` (DCT-based) — a 64-bit fingerprint that is stable
under compression, small resamples, and mild colour shifts. Two hashes are
compared with Hamming distance; values ``<= threshold`` indicate the same
content playing on the screen.

Privacy: the input here is a **screen capture** (what the signage is showing),
never a camera frame or face.
"""

from __future__ import annotations

import cv2
import imagehash
import numpy as np
from PIL import Image


def compute_phash(img_bgr: np.ndarray) -> int:
    """Return the 64-bit perceptual hash of ``img_bgr`` as an ``int``."""
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = img_bgr
    ph = imagehash.phash(Image.fromarray(rgb), hash_size=8)
    return int(str(ph), 16)


def hamming(a: int, b: int) -> int:
    """Bitwise Hamming distance between two 64-bit hashes."""
    return int(bin((a ^ b) & ((1 << 64) - 1)).count("1"))
