import json

import numpy as np
import torch
import torch.nn as nn
import pytest

from scripts.research.config import ExperimentConfig
from scripts.research.runner import BenchmarkRunner


def model_factory():
    return nn.Sequential(nn.Linear(2, 2))


def test_runner_is_deterministic_and_does_not_exclude_malicious_clients(tmp_path):
    features = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]] * 10,
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1] * 10, dtype=np.int64)
    config = ExperimentConfig(
        experiment_id="synthetic-smoke",
        dataset="synthetic",
        num_clients=4,
        participation_rate=1.0,
        malicious_fraction=0.25,
        rounds=2,
        local_epochs=1,
        seed=13,
        attack="sign_flip",
        defense="coordinate_median",
        batch_size=8,
        learning_rate=0.05,
    )

    first = BenchmarkRunner(config, tmp_path / "first").run_arrays(
        features, labels, model_factory
    )
    second = BenchmarkRunner(config, tmp_path / "second").run_arrays(
        features, labels, model_factory
    )

    assert {
        key: value
        for key, value in first.items()
        if key not in {"round_runtime_seconds", "runtime_seconds_total"}
    } == {
        key: value
        for key, value in second.items()
        if key not in {"round_runtime_seconds", "runtime_seconds_total"}
    }
    assert first["dataset"] == "synthetic"
    assert first["attack"] == "sign_flip"
    assert first["defense"] == "coordinate_median"
    assert first["seed"] == 13
    assert first["rounds_completed"] == 2
    assert first["malicious_updates_submitted"] == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "first" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    round_events = [event for event in events if event["event_type"] == "round_completed"]
    malicious_id = first["malicious_client_ids"][0]
    assert all(malicious_id in event["payload"]["participants"] for event in round_events)


def test_runner_records_environment_and_transmitted_bytes(tmp_path):
    features = np.array([[-1.0, 0.0], [1.0, 0.0]] * 8, dtype=np.float32)
    labels = np.array([0, 1] * 8, dtype=np.int64)
    config = ExperimentConfig(
        experiment_id="environment-smoke",
        dataset="synthetic",
        num_clients=2,
        participation_rate=1.0,
        rounds=1,
        seed=7,
        batch_size=4,
    )

    summary = BenchmarkRunner(config, tmp_path).run_arrays(features, labels, model_factory)

    first_event = json.loads(
        (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_event["event_type"] == "environment"
    assert "torch_version" in first_event["payload"]
    assert summary["transmitted_bytes"] > 0


def test_runner_executes_two_layer_committee_without_oracle_exclusion(tmp_path):
    features = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]] * 12,
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1] * 12, dtype=np.int64)
    config = ExperimentConfig(
        experiment_id="two-layer-smoke",
        dataset="synthetic",
        num_clients=4,
        participation_rate=1.0,
        malicious_fraction=0.25,
        rounds=1,
        seed=29,
        attack="sign_flip",
        defense="two_layer",
        batch_size=8,
        learning_rate=0.02,
    )

    summary = BenchmarkRunner(config, tmp_path).run_arrays(features, labels, model_factory)

    event = next(
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "round_completed"
    )
    malicious_id = summary["malicious_client_ids"][0]
    assert malicious_id in event["payload"]["participants"]
    assert event["payload"]["committee_decisions"]
    assert all(item["quorum"] == 2 for item in event["payload"]["committee_decisions"].values())
    assert summary["ledger_entries"] >= 2
    assert summary["ledger_bytes"] > 0
    assert len(summary["round_runtime_seconds"]) == 1
    assert summary["round_runtime_seconds"][0] > 0
    assert 0.0 <= summary["worst_client_utility"] <= 1.0
    assert summary["detection_true_positive_rate"] is not None
    assert summary["detection_false_positive_rate"] is not None


def test_validator_corruption_is_independent_from_poisoning_clients(tmp_path):
    features = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]] * 10,
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1] * 10, dtype=np.int64)
    config = ExperimentConfig(
        experiment_id="validator-capture",
        dataset="synthetic",
        num_clients=4,
        participation_rate=1.0,
        malicious_fraction=0.0,
        malicious_validator_fraction=0.5,
        rounds=1,
        seed=47,
        defense="two_layer",
        batch_size=8,
    )

    summary = BenchmarkRunner(config, tmp_path).run_arrays(features, labels, model_factory)

    assert summary["malicious_client_ids"] == []
    assert len(summary["malicious_validator_ids"]) == 2


def test_runner_supports_sift_style_routing_targets(tmp_path):
    features = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]] * 6,
        dtype=np.float32,
    )
    targets = np.array(
        [[3.0, 0.0], [2.0, 1.0], [0.0, 3.0], [1.0, 2.0]] * 6,
        dtype=np.float32,
    )
    config = ExperimentConfig(
        experiment_id="routing-smoke",
        dataset="sift_routing",
        num_clients=2,
        participation_rate=1.0,
        rounds=1,
        seed=71,
        defense="coordinate_median",
        batch_size=4,
    )

    summary = BenchmarkRunner(config, tmp_path).run_arrays(
        features, targets, model_factory
    )

    assert summary["task"] == "routing"
    assert 0.0 <= summary["final_utility"] <= 1.0


@pytest.mark.parametrize("defense", ["fltrust", "boba"])
def test_runner_supplies_trusted_reference_updates_to_contextual_baselines(
    tmp_path, defense
):
    features = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]] * 12,
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1] * 12, dtype=np.int64)
    config = ExperimentConfig(
        experiment_id=f"{defense}-smoke",
        dataset="synthetic",
        num_clients=4,
        participation_rate=1.0,
        malicious_fraction=0.25,
        rounds=1,
        seed=101,
        defense=defense,
        batch_size=8,
        server_samples_per_class=2,
    )

    summary = BenchmarkRunner(config, tmp_path / defense).run_arrays(
        features, labels, model_factory
    )

    assert summary["rounds_completed"] == 1
    assert summary["trusted_server_samples"] == 4


def test_model_replacement_records_targeted_backdoor_success_separately(tmp_path):
    features = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]] * 12,
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1] * 12, dtype=np.int64)
    config = ExperimentConfig(
        experiment_id="backdoor-smoke",
        dataset="synthetic",
        num_clients=4,
        participation_rate=1.0,
        malicious_fraction=0.25,
        rounds=1,
        seed=13,
        attack="model_replacement",
        defense="coordinate_median",
        batch_size=8,
        backdoor_target_label=0,
        backdoor_fraction=0.5,
    )

    summary = BenchmarkRunner(config, tmp_path).run_arrays(
        features, labels, model_factory
    )

    assert len(summary["targeted_attack_success_history"]) == 1
    assert 0.0 <= summary["mean_targeted_attack_success"] <= 1.0
    assert summary["malicious_update_acceptance_rate"] == 1.0
