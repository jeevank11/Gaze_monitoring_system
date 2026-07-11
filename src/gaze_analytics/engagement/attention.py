"""Attention gate — decide whether a viewer's head-pose counts as attending.

A viewer is "attending" if both yaw and pitch are within the configured
tolerance from center. Roll is ignored — a tilted head still counts as
watching the screen.
"""

from __future__ import annotations

from gaze_analytics.config import Settings, settings
from gaze_analytics.inference.head_pose import HeadPose


def is_attending(pose: HeadPose, cfg: Settings = settings) -> bool:
    """Return True if the head-pose is within the attention cone.

    Thresholds come from :class:`gaze_analytics.config.Settings`.
    """
    return abs(pose.yaw) <= cfg.head_yaw_max_deg and abs(pose.pitch) <= cfg.head_pitch_max_deg
