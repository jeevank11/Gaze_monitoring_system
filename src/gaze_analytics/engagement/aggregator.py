"""Rolling window aggregator — turns per-frame observations into privacy-safe rows.

Contract:

- Input each frame: a list of ``TrackedFace`` currently on screen, plus (optional)
  per-track attention flags and per-track gender labels for **this frame only**.
- Output every ``aggregate_window_seconds``: one :class:`WindowMetrics` row
  containing ONLY counts / max / averages. No track IDs, no bboxes, no
  images. This is what SQLite stores.

Per-track state (last-seen gender, first-seen ts, max dwell in window) lives
in a plain in-memory dict that is **wiped on every flush**. Nothing persists
across process restarts except the aggregate row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from gaze_analytics.config import Settings, settings
from gaze_analytics.tracker import TrackedFace

log = logging.getLogger(__name__)

Gender = Literal["M", "F"]


@dataclass(frozen=True)
class WindowMetrics:
    """One row of aggregate data — safe to store, safe to ship."""

    window_start_ts: float
    window_end_ts: float
    window_seconds: int
    viewers: int
    attending: int
    avg_dwell_ms: int
    male_count: int
    female_count: int
    segment_id: str | None


@dataclass
class _TrackState:
    max_dwell_ms: int = 0
    male_votes: int = 0
    female_votes: int = 0
    was_attending: bool = False

    @property
    def voted_gender(self) -> Gender | None:
        """Return the majority-voted gender, or None if no votes recorded."""
        if self.male_votes == 0 and self.female_votes == 0:
            return None
        return "M" if self.male_votes > self.female_votes else "F"


@dataclass
class _WindowState:
    started_ts: float
    max_viewers: int = 0
    max_attending: int = 0
    tracks: dict[int, _TrackState] = field(default_factory=dict)


class RollingAggregator:
    """Accumulates per-frame observations and emits aggregate rows on window close."""

    def __init__(self, cfg: Settings = settings) -> None:
        self._cfg = cfg
        self._window: _WindowState | None = None

    def record(
        self,
        tracks: list[TrackedFace],
        attending_ids: set[int],
        gender_by_id: dict[int, Gender],
        timestamp: float,
        face_count: int | None = None,
    ) -> None:
        """Fold one frame's observations into the current window."""
        if self._window is None:
            self._window = _WindowState(started_ts=timestamp)

        w = self._window
        # Use actual face detections (not tracks) for viewer count to avoid
        # inflating the number when the tracker briefly creates ghost tracks.
        viewers_this_frame = face_count if face_count is not None else len(tracks)
        w.max_viewers = max(w.max_viewers, viewers_this_frame)
        w.max_attending = max(w.max_attending, len(attending_ids))

        for trk in tracks:
            state = w.tracks.setdefault(trk.track_id, _TrackState())
            dwell_ms = int(trk.dwell_seconds * 1000)
            if dwell_ms > state.max_dwell_ms:
                state.max_dwell_ms = dwell_ms
            g = gender_by_id.get(trk.track_id)
            if g == "M":
                state.male_votes += 1
            elif g == "F":
                state.female_votes += 1
            if trk.track_id in attending_ids:
                state.was_attending = True

    def flush_if_due(
        self,
        timestamp: float,
        segment_id: str | None = None,
    ) -> WindowMetrics | None:
        """If the current window has expired, emit and reset. Returns ``None`` otherwise."""
        if self._window is None:
            return None
        if timestamp - self._window.started_ts < self._cfg.aggregate_window_seconds:
            return None
        return self._emit(timestamp, segment_id)

    def flush(
        self,
        timestamp: float,
        segment_id: str | None = None,
    ) -> WindowMetrics | None:
        """Force-emit the current window (used at shutdown)."""
        if self._window is None:
            return None
        return self._emit(timestamp, segment_id)

    # -- internals ------------------------------------------------------------

    def _emit(self, timestamp: float, segment_id: str | None) -> WindowMetrics:
        assert self._window is not None
        w = self._window
        n_tracks = len(w.tracks)
        avg_dwell_ms = (
            int(sum(s.max_dwell_ms for s in w.tracks.values()) / n_tracks)
            if n_tracks > 0
            else 0
        )
        male = sum(1 for s in w.tracks.values() if s.voted_gender == "M")
        female = sum(1 for s in w.tracks.values() if s.voted_gender == "F")
        metrics = WindowMetrics(
            window_start_ts=w.started_ts,
            window_end_ts=timestamp,
            window_seconds=self._cfg.aggregate_window_seconds,
            viewers=w.max_viewers,
            attending=w.max_attending,
            avg_dwell_ms=avg_dwell_ms,
            male_count=male,
            female_count=female,
            segment_id=segment_id,
        )
        # Wipe per-track state before starting next window.
        self._window = None
        return metrics
