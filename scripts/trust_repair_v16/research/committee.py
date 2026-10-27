"""Committee sizing and quorum invariants."""

import math

from .contracts import CommitteeDecision


class CommitteePolicy:
    def __init__(self, ratio: float = 0.3, minimum_size: int = 3):
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        if minimum_size < 1:
            raise ValueError("minimum_size must be positive")
        self.ratio = ratio
        self.minimum_size = minimum_size

    def for_population(self, eligible_population: int) -> CommitteeDecision:
        if eligible_population < 0:
            raise ValueError("eligible_population cannot be negative")
        size = min(
            eligible_population,
            max(self.minimum_size, math.ceil(self.ratio * eligible_population)),
        )
        quorum = math.ceil(2 * size / 3) if size else 0
        return CommitteeDecision(
            eligible_population=eligible_population,
            size=size,
            quorum=quorum,
            consensus_supported=eligible_population >= self.minimum_size,
        )
