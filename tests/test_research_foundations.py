import json

import numpy as np
import pytest

from scripts.research.committee import CommitteePolicy
from scripts.research.config import ExperimentConfig
from scripts.research.identity import IdentityOracle


@pytest.mark.parametrize(
    ("eligible", "expected_size", "expected_quorum", "supported"),
    [
        (2, 2, 2, False),
        (4, 3, 2, True),
        (8, 3, 2, True),
        (20, 6, 4, True),
    ],
)
def test_committee_policy_uses_ceilings_and_reports_small_population(
    eligible, expected_size, expected_quorum, supported
):
    decision = CommitteePolicy().for_population(eligible)

    assert decision.size == expected_size
    assert decision.quorum == expected_quorum
    assert decision.consensus_supported is supported


def test_identity_oracle_models_false_accepts_and_false_rejects():
    oracle = IdentityOracle(false_accept_rate=1.0, false_reject_rate=1.0)
    rng = np.random.default_rng(17)

    duplicate = oracle.enroll(client_id=7, is_duplicate_identity=True, rng=rng)
    unique = oracle.enroll(client_id=8, is_duplicate_identity=False, rng=rng)

    assert duplicate.accepted is True
    assert duplicate.error_kind == "false_accept"
    assert unique.accepted is False
    assert unique.error_kind == "false_reject"


def test_identity_oracle_does_not_receive_or_infer_malicious_intent():
    oracle = IdentityOracle(false_accept_rate=0.0, false_reject_rate=0.0)

    decision = oracle.enroll(
        client_id=11,
        is_duplicate_identity=False,
        rng=np.random.default_rng(4),
    )

    assert decision.accepted is True
    assert not hasattr(decision, "is_malicious")


def test_experiment_config_round_trips_and_rejects_invalid_fractions(tmp_path):
    config = ExperimentConfig(
        experiment_id="smoke-fmnist",
        dataset="fashion_mnist",
        num_clients=20,
        participation_rate=0.2,
        malicious_fraction=0.3,
        dirichlet_alpha=0.5,
        rounds=3,
        local_epochs=1,
        seed=29,
        attack="sign_flip",
        defense="fedavg",
    )
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    loaded = ExperimentConfig.from_json(path)

    assert loaded == config
    assert loaded.participants_per_round == 4

    with pytest.raises(ValueError, match="participation_rate"):
        ExperimentConfig(
            experiment_id="bad",
            dataset="fashion_mnist",
            num_clients=20,
            participation_rate=0.0,
        )

    with pytest.raises(ValueError, match="experiment_id"):
        ExperimentConfig(
            experiment_id="../escape",
            dataset="synthetic",
            num_clients=4,
            participation_rate=1.0,
        )
