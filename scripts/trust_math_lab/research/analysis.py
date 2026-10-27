"""Seed-level statistical summaries for preregistered experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float


def learning_curve_auc(values: list[float]) -> float:
    if not values:
        raise ValueError("learning curve cannot be empty")
    if len(values) == 1:
        return float(values[0])
    return float(np.trapezoid(np.asarray(values, dtype=float)))


def bootstrap_mean_ci(
    values: list[float], seed: int, samples: int = 10_000, confidence: float = 0.95
) -> ConfidenceInterval:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("at least two seed-level values are required")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(bootstrap_means, [tail, 1 - tail])
    return ConfidenceInterval(float(array.mean()), float(lower), float(upper))


def holm_adjust(p_values: list[float]) -> list[float]:
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted
