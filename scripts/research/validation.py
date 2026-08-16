"""Local-baseline validation and quorum aggregation."""

import hashlib
import math
import statistics

from .contracts import ValidationVote, VoteDecision


class LocalValidator:
    def __init__(self, validator_id: int = -1, minimum_relative_delta: float = -0.05):
        self.validator_id = validator_id
        self.minimum_relative_delta = minimum_relative_delta

    def with_id(self, validator_id: int) -> "LocalValidator":
        return LocalValidator(validator_id, self.minimum_relative_delta)

    def vote(
        self,
        candidate_id: int,
        baseline_score: float,
        candidate_score: float,
        signer_is_malicious: bool,
    ) -> ValidationVote:
        denominator = max(abs(baseline_score), 1e-12)
        delta = (candidate_score - baseline_score) / denominator
        approve = delta >= self.minimum_relative_delta or math.isclose(
            delta, self.minimum_relative_delta, rel_tol=1e-12, abs_tol=1e-12
        )
        material = f"{self.validator_id}:{candidate_id}:{delta:.12f}:{int(approve)}"
        signature = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return ValidationVote(
            self.validator_id,
            candidate_id,
            delta,
            approve,
            signer_is_malicious,
            signature,
        )


class VoteAggregator:
    def decide(self, votes: list[ValidationVote], quorum: int) -> VoteDecision:
        if quorum < 1:
            raise ValueError("quorum must be positive")
        if not votes:
            raise ValueError("votes cannot be empty")
        candidate_ids = {vote.candidate_id for vote in votes}
        if len(candidate_ids) != 1:
            raise ValueError("all votes must refer to one candidate")
        affirmative = tuple(sorted(vote.validator_id for vote in votes if vote.approve))
        return VoteDecision(
            accepted=len(affirmative) >= quorum,
            quorum=quorum,
            affirmative_signers=affirmative,
            median_normalized_delta=float(
                statistics.median(vote.normalized_delta for vote in votes)
            ),
        )
