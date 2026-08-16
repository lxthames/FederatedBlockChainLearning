import json

import pytest

from scripts.research.results import aggregate_summaries


def write_summary(path, seed, utility):
    path.write_text(
        json.dumps(
            {
                "experiment_id": f"exp-{seed}",
                "dataset": "fashion_mnist",
                "attack": "sign_flip",
                "defense": "two_layer",
                "seed": seed,
                "final_utility": utility,
                "accuracy_history": [0.0, utility],
                "malicious_update_acceptance_rate": 0.25,
                "transmitted_bytes": 100,
                "ledger_bytes": 20,
                "runtime_seconds_total": 2.0,
                "worst_client_utility": utility - 0.1,
            }
        ),
        encoding="utf-8",
    )


def test_result_aggregation_uses_seed_level_values_and_learning_curve_auc(tmp_path):
    paths = []
    for seed, utility in zip((13, 29, 47, 71, 101), (0.5, 0.6, 0.7, 0.8, 0.9)):
        path = tmp_path / f"{seed}.json"
        write_summary(path, seed, utility)
        paths.append(path)

    rows = aggregate_summaries(paths, require_confirmatory_seeds=True)

    assert len(rows) == 1
    assert rows[0]["seed_count"] == 5
    assert rows[0]["final_utility_mean"] == pytest.approx(0.7)
    assert rows[0]["learning_curve_auc_mean"] == pytest.approx(0.35)
    assert rows[0]["final_utility_ci_lower"] <= 0.7
    assert rows[0]["final_utility_ci_upper"] >= 0.7


def test_result_aggregation_refuses_incomplete_confirmatory_seed_set(tmp_path):
    path = tmp_path / "13.json"
    write_summary(path, 13, 0.5)

    with pytest.raises(ValueError, match="confirmatory seeds"):
        aggregate_summaries([path], require_confirmatory_seeds=True)
