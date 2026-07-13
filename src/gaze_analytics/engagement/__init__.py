"""Attention gating, dwell time, aggregation — sprints 3 and 5."""

from __future__ import annotations

from gaze_analytics.engagement.aggregator import RollingAggregator, WindowMetrics
from gaze_analytics.engagement.attention import AttentionSmoother, is_attending

__all__ = ["AttentionSmoother", "RollingAggregator", "WindowMetrics", "is_attending"]
