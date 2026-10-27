"""Local-baseline validation and quorum aggregation."""

import hashlib
import math
import statistics

from .contracts import ValidationVote, VoteDecision


class LocalValidator:
    def __init__(self, validator_id: int = -1, minimum_relative_delta: float = -0.05, absolute_tolerance: float = 0.02):
        self.validator_id = validator_id
        self.minimum_relative_delta = minimum_relative_delta
        self.absolute_tolerance = absolute_tolerance

    def with_id(self, validator_id: int) -> "LocalValidator":
        return LocalValidator(validator_id, self.minimum_relative_delta, self.absolute_tolerance)

    def vote(
        self,
        candidate_id: int,
        baseline_score: float,
        candidate_score: float,
        signer_is_malicious: bool,
    ) -> ValidationVote:
        denominator = max(abs(baseline_score), 1e-12)
        delta = (candidate_score - baseline_score) / denominator
        # Compare the SAME fully applied candidate on the SAME private examples.
        # Relative changes near zero utility exaggerated small absolute changes.
        change = candidate_score - baseline_score
        approve = change >= -self.absolute_tolerance or math.isclose(
            change, -self.absolute_tolerance, rel_tol=1e-12, abs_tol=1e-12
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
    def decide(self, votes: list[ValidationVote], quorum: int, expected_validators=None) -> VoteDecision:
        if quorum < 1:
            raise ValueError("quorum must be positive")
        if not votes and expected_validators is None:
            raise ValueError("votes cannot be empty")
        expected = tuple(expected_validators) if expected_validators is not None else tuple(v.validator_id for v in votes)
        if len(set(expected)) != len(expected) or quorum > len(expected):
            raise ValueError('invalid committee or quorum')
        if any(v.validator_id not in expected for v in votes):
            raise ValueError('vote from outside committee')
        candidate_ids = {vote.candidate_id for vote in votes}
        if len({vote.validator_id for vote in votes}) != len(votes):
            raise ValueError("duplicate validator vote")
        if votes and len(candidate_ids) != 1:
            raise ValueError("all votes must refer to one candidate")
        affirmative = tuple(sorted(vote.validator_id for vote in votes if vote.approve))
        missing = len(expected) - len(votes)
        # Reject on quality only when the received negative votes already block
        # approval, even if every missing member were to approve.
        outcome = ('accepted' if len(affirmative) >= quorum else
                   'quality_rejected' if len(affirmative) + missing < quorum else
                   'insufficient_response')
        return VoteDecision(
            accepted=len(affirmative) >= quorum,
            quorum=quorum,
            affirmative_signers=affirmative,
            median_normalized_delta=float(
                statistics.median(vote.normalized_delta for vote in votes)
            ) if votes else 0.0,
            outcome=outcome, response_count=len(votes), expected_count=len(expected),
            negative_count=len(votes)-len(affirmative),
        )
