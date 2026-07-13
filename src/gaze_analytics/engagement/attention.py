"""Attention gate — decide whether a viewer is attending the screen.

Two-tier logic:
- **Head-pose only** (far tier or no gaze model): viewer is attending if
  yaw/pitch are within tolerance.
- **Head-pose + gaze** (near tier): head-pose must pass AND gaze angles
  must be within the tighter gaze tolerance. This catches people who face
  the screen but look at their phone.

An optional :class:`AttentionSmoother` applies per-track hysteresis so that
brief head jitter or a quick glance sideways does not flip a viewer to
"away". Away→attending flips instantly; attending→away only after N
consecutive failing frames.
"""

from __future__ import annotations

from gaze_analytics.config import Settings, settings
from gaze_analytics.inference.gaze import GazeVector
from gaze_analytics.inference.head_pose import HeadPose


def is_attending(
    pose: HeadPose,
    cfg: Settings = settings,
    gaze: GazeVector | None = None,
) -> bool:
    """Return True if the viewer is attending the screen.

    If a gaze vector is provided, both head-pose AND gaze must pass.
    Otherwise, only head-pose is checked (far-tier fallback).
    """
    head_ok = abs(pose.yaw) <= cfg.head_yaw_max_deg and abs(pose.pitch) <= cfg.head_pitch_max_deg
    if not head_ok:
        return False
    if gaze is not None:
        return (
            abs(gaze.yaw_deg) <= cfg.gaze_angle_max_deg
            and abs(gaze.pitch_deg) <= cfg.gaze_angle_max_deg
        )
    return True


class AttentionSmoother:
    """Per-track hysteresis over raw attention decisions.

    A track flips from attending → away only after ``away_frames`` consecutive
    frames of raw non-attention. The reverse transition (away → attending) is
    instant. Untracked observations (no ``track_id``) bypass smoothing.
    """

    def __init__(self, away_frames: int = 8) -> None:
        self._away_frames = max(1, int(away_frames))
        self._state: dict[int, bool] = {}
        self._away_streak: dict[int, int] = {}

    def update(self, track_id: int, raw_attending: bool) -> bool:
        """Feed one frame's raw decision for ``track_id``; return smoothed decision."""
        if raw_attending:
            self._away_streak[track_id] = 0
            self._state[track_id] = True
            return True
        # Raw = away. If already away, stay away.
        if not self._state.get(track_id, False):
            self._state[track_id] = False
            return False
        # Was attending — extend the away streak; hold "attending" during grace.
        streak = self._away_streak.get(track_id, 0) + 1
        self._away_streak[track_id] = streak
        if streak >= self._away_frames:
            self._state[track_id] = False
            return False
        return True

    def gc(self, active_ids: set[int]) -> None:
        """Drop state for tracks that no longer exist to avoid unbounded growth."""
        for tid in list(self._state.keys()):
            if tid not in active_ids:
                self._state.pop(tid, None)
                self._away_streak.pop(tid, None)
