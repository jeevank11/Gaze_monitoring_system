"""SORT-style Kalman filter for a single face bounding box.

State layout (7-D):
    x = [u, v, s, r, du, dv, ds]
where
    u,v = bbox center (px)
    s   = bbox area (u * v)
    r   = aspect ratio w/h (assumed constant, no velocity term)

Observed: [u, v, s, r].

We deliberately keep this file free of any face pixels — the tracker only
sees geometry (four ints per detection) and never touches image data.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

from gaze_analytics.inference.face import FaceBBox


def _bbox_to_z(bbox: FaceBBox) -> np.ndarray:
    w = float(bbox.xmax - bbox.xmin)
    h = float(bbox.ymax - bbox.ymin)
    u = bbox.xmin + w / 2.0
    v = bbox.ymin + h / 2.0
    s = w * h
    r = w / h if h > 0 else 1.0
    return np.array([u, v, s, r], dtype=np.float64)


def _z_to_bbox_ints(z: np.ndarray) -> tuple[int, int, int, int]:
    u, v, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    s = max(s, 1.0)
    r = max(r, 1e-3)
    w = np.sqrt(s * r)
    h = s / w
    x1 = round(u - w / 2.0)
    y1 = round(v - h / 2.0)
    x2 = round(u + w / 2.0)
    y2 = round(v + h / 2.0)
    return x1, y1, x2, y2


class KalmanBoxTracker:
    """Constant-velocity Kalman tracker for one face bbox."""

    _next_id: int = 0

    def __init__(self, bbox: FaceBBox) -> None:
        kf = KalmanFilter(dim_x=7, dim_z=4)
        # State transition: velocities are added to (u, v, s).
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        # Measurement: we observe u, v, s, r.
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        # Noise / covariance — the numbers below are the classic SORT settings.
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        z = _bbox_to_z(bbox)
        kf.x[:4] = z.reshape(4, 1)

        self.kf = kf
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.hits: int = 1
        self.misses: int = 0
        self.age: int = 0
        self.score: float = bbox.score

    # -- state helpers --------------------------------------------------------

    def predict(self) -> tuple[int, int, int, int]:
        """Advance the filter one step. Returns the predicted bbox corners."""
        # Guard against area collapsing below zero after a large ds prediction.
        if float(self.kf.x[6, 0]) + float(self.kf.x[2, 0]) <= 0:
            self.kf.x[6, 0] = 0.0
        self.kf.predict()
        self.age += 1
        self.misses += 1
        return _z_to_bbox_ints(self.kf.x[:4].flatten())

    def update(self, bbox: FaceBBox) -> None:
        """Fold a new observation into the filter."""
        self.hits += 1
        self.misses = 0
        self.score = bbox.score
        self.kf.update(_bbox_to_z(bbox))

    def bbox(self) -> tuple[int, int, int, int]:
        """Return the current filtered bbox (no predict step)."""
        return _z_to_bbox_ints(self.kf.x[:4].flatten())

    # -- test hook ------------------------------------------------------------

    @classmethod
    def _reset_ids_for_tests(cls) -> None:
        cls._next_id = 0
