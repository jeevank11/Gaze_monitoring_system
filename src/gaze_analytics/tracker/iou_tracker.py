"""IoU-based multi-face tracker with Kalman motion smoothing.

Contract (privacy):
    * Input: list of :class:`FaceBBox` (integer geometry only).
    * Output: list of :class:`TrackedFace` — integer geometry plus an int track
      ID that lives only for the lifetime of the process.
    * We never see, store, or emit face pixels or embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from gaze_analytics.inference.face import FaceBBox
from gaze_analytics.tracker.kalman_box import KalmanBoxTracker


@dataclass(frozen=True)
class TrackedFace:
    """A single tracked face, uniquely identified by ``track_id`` for this process."""

    track_id: int
    bbox: FaceBBox
    hits: int
    age: int
    first_seen_ts: float
    last_seen_ts: float

    @property
    def dwell_seconds(self) -> float:
        return max(0.0, self.last_seen_ts - self.first_seen_ts)


def _iou(a: tuple[int, int, int, int], b: FaceBBox) -> float:
    """Intersection-over-union of two bboxes."""
    ax1, ay1, ax2, ay2 = a
    ix1 = max(ax1, b.xmin)
    iy1 = max(ay1, b.ymin)
    ix2 = min(ax2, b.xmax)
    iy2 = min(ay2, b.ymax)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    denom = area_a + area_b - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


class IoUTracker:
    """Greedy IoU + Kalman tracker.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU between a predicted track and a detection for them to be
        considered a match.
    max_misses:
        Number of consecutive frames a track may go unobserved before it is
        deleted.
    min_hits:
        Number of matched observations required before a track is emitted.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 15,
        min_hits: int = 3,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._min_hits = min_hits
        self._trackers: list[KalmanBoxTracker] = []
        self._first_seen: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    def update(
        self,
        detections: list[FaceBBox],
        timestamp: float,
    ) -> list[TrackedFace]:
        """Fold ``detections`` into the tracker and return active tracks."""
        # 1. Predict everything forward.
        predicted: list[tuple[int, int, int, int]] = [t.predict() for t in self._trackers]

        # 2. Hungarian association on the IoU matrix.
        matches, unmatched_dets, _unmatched_trks = self._associate(predicted, detections)

        # 3. Update matched trackers.
        for det_idx, trk_idx in matches:
            trk = self._trackers[trk_idx]
            trk.update(detections[det_idx])
            self._last_seen[trk.id] = timestamp

        # 4. New tracks for unmatched detections.
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(detections[det_idx])
            self._trackers.append(trk)
            self._first_seen[trk.id] = timestamp
            self._last_seen[trk.id] = timestamp

        # 5. Cull dead tracks.
        alive: list[KalmanBoxTracker] = []
        for trk in self._trackers:
            if trk.misses <= self._max_misses:
                alive.append(trk)
            else:
                self._first_seen.pop(trk.id, None)
                self._last_seen.pop(trk.id, None)
        self._trackers = alive

        # 6. Emit tracks that have enough observations or are still fresh.
        out: list[TrackedFace] = []
        for trk in self._trackers:
            if trk.hits < self._min_hits and trk.age > self._min_hits:
                continue
            x1, y1, x2, y2 = trk.bbox()
            bbox = FaceBBox(
                xmin=int(max(0, x1)),
                ymin=int(max(0, y1)),
                xmax=int(x2),
                ymax=int(y2),
                score=float(trk.score),
            )
            out.append(
                TrackedFace(
                    track_id=trk.id,
                    bbox=bbox,
                    hits=trk.hits,
                    age=trk.age,
                    first_seen_ts=self._first_seen[trk.id],
                    last_seen_ts=self._last_seen[trk.id],
                )
            )
        return out

    # -- helpers --------------------------------------------------------------

    def _associate(
        self,
        predicted: list[tuple[int, int, int, int]],
        detections: list[FaceBBox],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian assignment between predictions and detections."""
        if not predicted or not detections:
            return [], list(range(len(detections))), list(range(len(predicted)))

        iou_mat = np.zeros((len(detections), len(predicted)), dtype=np.float64)
        for d, det in enumerate(detections):
            for t, pred in enumerate(predicted):
                iou_mat[d, t] = _iou(pred, det)

        # linear_sum_assignment minimises cost → use negative IoU.
        det_idx, trk_idx = linear_sum_assignment(-iou_mat)
        matches: list[tuple[int, int]] = []
        for d, t in zip(det_idx, trk_idx, strict=False):
            if iou_mat[d, t] >= self._iou_threshold:
                matches.append((int(d), int(t)))

        matched_dets = {d for d, _ in matches}
        matched_trks = {t for _, t in matches}
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(predicted)) if t not in matched_trks]
        return matches, unmatched_dets, unmatched_trks
