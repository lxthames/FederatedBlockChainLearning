"""Windowed detector for abrupt and gradual performance degradation."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    reason: str | None
    relative_drop: float
    slope: float


class TrendDetector:
    def __init__(
        self,
        window_size: int = 5,
        relative_drop_threshold: float = 0.05,
        slope_threshold: float = 0.005,
    ):
        if window_size < 3:
            raise ValueError("window_size must be at least 3")
        if relative_drop_threshold <= 0 or slope_threshold <= 0:
            raise ValueError("detector thresholds must be positive")
        self.window_size = window_size
        self.relative_drop_threshold = relative_drop_threshold
        self.slope_threshold = slope_threshold
        self._history: deque[float] = deque(maxlen=window_size)

    def observe(self, score: float) -> DetectionResult:
        self._history.append(float(score))
        if len(self._history) < self.window_size:
            return DetectionResult(False, None, 0.0, 0.0)
        values = np.asarray(self._history, dtype=float)
        reference = float(values[:-1].mean())
        relative_drop = (reference - values[-1]) / max(abs(reference), 1e-12)
        slope = float(np.polyfit(np.arange(self.window_size), values, 1)[0])
        if relative_drop > self.relative_drop_threshold:
            return DetectionResult(True, "abrupt_drop", relative_drop, slope)
        if slope < -self.slope_threshold:
            return DetectionResult(True, "negative_trend", relative_drop, slope)
        return DetectionResult(False, None, relative_drop, slope)
