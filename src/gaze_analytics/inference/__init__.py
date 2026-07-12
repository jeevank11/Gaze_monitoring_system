"""OpenVINO inference wrappers (to be implemented in sprints 2-5)."""

from __future__ import annotations

from gaze_analytics.inference.age_gender import AgeGender, AgeGenderEstimator
from gaze_analytics.inference.face import FaceBBox, FaceDetector, draw_face_bboxes
from gaze_analytics.inference.gaze import GazeEstimator, GazeVector
from gaze_analytics.inference.head_pose import HeadPose, HeadPoseEstimator

__all__ = [
    "AgeGender",
    "AgeGenderEstimator",
    "FaceBBox",
    "FaceDetector",
    "GazeEstimator",
    "GazeVector",
    "HeadPose",
    "HeadPoseEstimator",
    "draw_face_bboxes",
]

