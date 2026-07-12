"""Content segmenter — auto-detects when the signage screen changes content.

Watches a low-frequency stream of screen captures, computes a perceptual hash
of each, and emits a new :class:`ContentSegment` whenever the hash drifts
farther than ``phash_hamming_threshold`` from the current segment.

The 160x90 thumbnail stored on each segment is a **content** thumbnail
(what the screen was showing) — never a face crop, never a camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gaze_analytics.config import Settings, settings
from gaze_analytics.content.phash import compute_phash, hamming


@dataclass
class ContentSegment:
    """A window of time during which the signage screen showed the same content."""

    segment_id: int
    phash: int
    first_seen_ts: float
    last_seen_ts: float
    thumbnail: np.ndarray | None = field(default=None, repr=False)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_seen_ts - self.first_seen_ts)


class ContentSegmenter:
    """Turns a stream of screen frames into a stream of content segments."""

    def __init__(self, cfg: Settings = settings) -> None:
        self._cfg = cfg
        self._next_id = 0
        self._current: ContentSegment | None = None

    @property
    def current(self) -> ContentSegment | None:
        return self._current

    def update(self, screen_bgr: np.ndarray, timestamp: float) -> ContentSegment:
        """Fold one screen capture into the segmenter and return the active segment."""
        ph = compute_phash(screen_bgr)
        if self._current is None:
            return self._start_segment(ph, screen_bgr, timestamp)

        distance = hamming(ph, self._current.phash)
        if distance > self._cfg.phash_hamming_threshold:
            # pHash has drifted enough to look like new content, but suppress
            # sub-``min_segment_seconds`` flicker (video frames, animations,
            # transient overlays) so downstream aggregates stay meaningful.
            age = timestamp - self._current.first_seen_ts
            if age >= self._cfg.min_segment_seconds:
                return self._start_segment(ph, screen_bgr, timestamp)

        self._current.last_seen_ts = timestamp
        return self._current

    # -- internals ------------------------------------------------------------

    def _start_segment(
        self,
        ph: int,
        screen_bgr: np.ndarray,
        timestamp: float,
    ) -> ContentSegment:
        thumb: np.ndarray | None = None
        if self._cfg.keep_segment_thumbnail:
            tw, th = self._cfg.thumbnail_size
            thumb = cv2.resize(screen_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        seg = ContentSegment(
            segment_id=self._next_id,
            phash=ph,
            first_seen_ts=timestamp,
            last_seen_ts=timestamp,
            thumbnail=thumb,
        )
        self._next_id += 1
        self._current = seg
        return seg
