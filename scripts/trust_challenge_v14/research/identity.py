"""External identity-oracle model with explicit enrollment errors."""

import numpy as np

from .contracts import EnrollmentDecision


class IdentityOracle:
    def __init__(self, false_accept_rate: float = 0.0, false_reject_rate: float = 0.0):
        for name, value in (
            ("false_accept_rate", false_accept_rate),
            ("false_reject_rate", false_reject_rate),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.false_accept_rate = false_accept_rate
        self.false_reject_rate = false_reject_rate

    def enroll(
        self, client_id: int, is_duplicate_identity: bool, rng: np.random.Generator
    ) -> EnrollmentDecision:
        if is_duplicate_identity:
            accepted = bool(rng.random() < self.false_accept_rate)
            error_kind = "false_accept" if accepted else None
        else:
            accepted = not bool(rng.random() < self.false_reject_rate)
            error_kind = "false_reject" if not accepted else None
        return EnrollmentDecision(client_id, accepted, is_duplicate_identity, error_kind)
