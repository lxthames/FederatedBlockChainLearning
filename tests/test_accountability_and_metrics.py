import json

import pytest
import torch

from scripts.research.accountability import AccountabilityManager
from scripts.research.events import EventLogger
from scripts.research.ledger import AppendOnlyRecoveryLedger
from scripts.research.metrics import RoundSecurityMetrics, SecurityMetricsTracker
from scripts.research.validation import LocalValidator, VoteAggregator


def test_local_validation_normalizes_against_each_validators_baseline():
    vote = LocalValidator(validator_id=4, minimum_relative_delta=-0.05).vote(
        candidate_id=9,
        baseline_score=0.8,
        candidate_score=0.76,
        signer_is_malicious=False,
    )

    assert vote.normalized_delta == pytest.approx(-0.05)
    assert vote.approve is True


def test_vote_aggregator_requires_exact_quorum():
    validator = LocalValidator(minimum_relative_delta=-0.05)
    votes = [
        validator.with_id(0).vote(9, 0.8, 0.8, False),
        validator.with_id(1).vote(9, 0.8, 0.8, False),
        validator.with_id(2).vote(9, 0.8, 0.6, False),
    ]

    accepted = VoteAggregator().decide(votes, quorum=2)
    rejected = VoteAggregator().decide(votes, quorum=3)

    assert accepted.accepted is True
    assert accepted.affirmative_signers == (0, 1)
    assert rejected.accepted is False


def test_recovery_restores_checkpoint_without_removing_ledger_entries():
    ledger = AppendOnlyRecoveryLedger()
    ledger.checkpoint(1, {"weight": torch.tensor([1.0])})
    ledger.append_model_commit(version=1, metadata={"round": 1})
    ledger.append_model_commit(version=2, metadata={"round": 2})
    length_before = len(ledger.entries)

    restored, record = ledger.recover(
        rejected_version=2,
        restored_version=1,
        trigger="trend_detector",
        affirmative_signers=(2, 3),
    )

    assert torch.equal(restored["weight"], torch.tensor([1.0]))
    assert len(ledger.entries) == length_before + 1
    assert ledger.entries[-1].kind == "recovery"
    assert record.rejected_version == 2
    assert record.restored_version == 1
    assert ledger.serialized_size_bytes() > 0


def test_slashing_targets_only_affirmative_signers_and_counts_wrongful_penalties():
    manager = AccountabilityManager(penalty=5.0)
    manager.initialize({0: 20.0, 1: 20.0, 2: 20.0})

    result = manager.slash_affirmative_signers(
        affirmative_signers=(0, 2), malicious_client_ids={2}
    )

    assert manager.balance(0) == 15.0
    assert manager.balance(1) == 20.0
    assert manager.balance(2) == 15.0
    assert result.penalized_client_ids == (0, 2)
    assert result.wrongful_slashing_count == 1


def test_jsonl_event_logger_preserves_required_reproducibility_fields(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, experiment_id="exp-1", seed=47)

    logger.log("round_completed", round_index=3, payload={"participants": [1, 4]})

    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["experiment_id"] == "exp-1"
    assert event["seed"] == 47
    assert event["round_index"] == 3
    assert event["event_type"] == "round_completed"
    assert event["payload"]["participants"] == [1, 4]


def test_security_metrics_do_not_conflate_update_acceptance_and_attack_success():
    tracker = SecurityMetricsTracker()
    tracker.add_round(
        RoundSecurityMetrics(
            round_index=1,
            malicious_submitted=4,
            malicious_accepted=2,
            targeted_attack_success=0.8,
            true_positives=2,
            false_positives=1,
            false_negatives=2,
            true_negatives=5,
        )
    )

    summary = tracker.summary()

    assert summary["malicious_update_acceptance_rate"] == 0.5
    assert summary["mean_targeted_attack_success"] == 0.8
    assert summary["detection_true_positive_rate"] == 0.5
    assert summary["detection_false_positive_rate"] == 1 / 6
