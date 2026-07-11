"""Content segmentation via perceptual hashing — sprint 4."""

from __future__ import annotations

from gaze_analytics.content.phash import compute_phash, hamming
from gaze_analytics.content.segmenter import ContentSegment, ContentSegmenter

__all__ = ["ContentSegment", "ContentSegmenter", "compute_phash", "hamming"]

