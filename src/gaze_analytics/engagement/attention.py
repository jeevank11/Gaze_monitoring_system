"""Attention gate — decide whether a viewer is attending the screen.

Two-tier logic:
- **Head-pose only** (far tier or no gaze model): viewer is attending if
  yaw/pitch are within tolerance.
- **Head-pose + gaze** (near tier): head-pose must pass AND gaze angles
  must be within the tighter gaze tolerance. This catches people who face
  the screen but look at their phone.
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
