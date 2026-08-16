import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

from scripts.research.config import ExperimentConfig
from scripts.research.dataio import load_sift_arrays
from scripts.research.models import create_model


def test_config_loads_from_yaml(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "yaml-smoke",
                "dataset": "synthetic",
                "num_clients": 4,
                "participation_rate": 1.0,
                "rounds": 1,
            }
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_file(path)

    assert config.experiment_id == "yaml-smoke"
    assert config.rounds == 1


def test_sift_loader_normalizes_features_and_preserves_count_targets(tmp_path):
    feature_path = tmp_path / "features.npy"
    target_path = tmp_path / "targets.npy"
    np.save(feature_path, np.array([[0, 255], [128, 64]], dtype=np.uint8))
    np.save(target_path, np.array([[2, 0], [1, 1]], dtype=np.int32))

    features, targets = load_sift_arrays(feature_path, target_path, limit=1)

    assert features.dtype == np.float32
    assert np.array_equal(features, np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(targets, np.array([[2.0, 0.0]], dtype=np.float32))


def test_model_factory_matches_dataset_shapes():
    fashion = create_model("fashion_mnist", input_shape=(1, 28, 28), outputs=10)
    sift = create_model("sift_routing", input_shape=(128,), outputs=20)

    assert fashion(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert sift(torch.zeros(2, 128)).shape == (2, 20)


def test_cli_runs_synthetic_smoke_experiment(tmp_path):
    config_path = tmp_path / "smoke.json"
    output_path = tmp_path / "output"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "cli-smoke",
                "dataset": "synthetic",
                "num_clients": 4,
                "participation_rate": 1.0,
                "malicious_fraction": 0.25,
                "rounds": 1,
                "seed": 13,
                "attack": "sign_flip",
                "defense": "coordinate_median",
                "batch_size": 8,
                "synthetic_samples": 40,
                "input_dim": 4,
                "num_classes": 2,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.cli",
            "run",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["rounds_completed"] == 1
    assert (output_path / "events.jsonl").exists()


def test_cli_runs_and_resumes_manifest_without_overwriting_completed_cells(tmp_path):
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "batch"
    base = {
        "dataset": "synthetic",
        "num_clients": 4,
        "participation_rate": 1.0,
        "rounds": 1,
        "batch_size": 8,
        "synthetic_samples": 40,
        "input_dim": 4,
        "num_classes": 2,
    }
    manifest.write_text(
        json.dumps(
            [
                {**base, "experiment_id": "batch-a", "seed": 13},
                {**base, "experiment_id": "batch-b", "seed": 29},
            ]
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "scripts.research.cli",
        "run-manifest",
        "--manifest",
        str(manifest),
        "--output-root",
        str(output),
    ]
    first = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True)
    second = subprocess.run(
        command + ["--resume"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    batch = json.loads((output / "batch-summary.json").read_text(encoding="utf-8"))
    assert batch == {"completed": 0, "failed": 0, "skipped": 2, "total": 2}
    assert (output / "batch-a" / "summary.json").exists()
    assert (output / "batch-b" / "summary.json").exists()
