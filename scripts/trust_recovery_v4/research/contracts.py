"""Stable data contracts shared by research components."""

from dataclasses import dataclass, field
from typing import Any, Mapping


StateDict = Mapping[str, Any]


@dataclass(frozen=True)
class EnrollmentDecision:
    client_id: int
    accepted: bool
    is_duplicate_identity: bool
    error_kind: str | None = None


@dataclass(frozen=True)
class CommitteeDecision:
    eligible_population: int
    size: int
    quorum: int
    consensus_supported: bool


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    state: StateDict
    sample_count: int
    is_malicious: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregationResult:
    state: StateDict
    accepted_client_ids: tuple[int, ...]
    rejected_client_ids: tuple[int, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationVote:
    validator_id: int
    candidate_id: int
    normalized_delta: float
    approve: bool
    signer_is_malicious: bool
    signature: str


@dataclass(frozen=True)
class VoteDecision:
    accepted: bool
    quorum: int
    affirmative_signers: tuple[int, ...]
    median_normalized_delta: float


@dataclass(frozen=True)
class RecoveryRecord:
    rejected_version: int
    restored_version: int
    trigger: str
    affirmative_signers: tuple[int, ...]


@dataclass(frozen=True)
class SlashingResult:
    penalized_client_ids: tuple[int, ...]
    wrongful_slashing_count: int
