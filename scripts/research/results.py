"""Aggregate seed-level run summaries without treating rounds as replicates."""

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .analysis import bootstrap_mean_ci, learning_curve_auc
from .matrix import SEEDS


GROUP_FIELDS = (
    "dataset",
    "attack",
    "defense",
    "num_clients",
    "participation_rate",
    "malicious_fraction",
    "dirichlet_alpha",
    "identity_false_accept_rate",
    "identity_false_reject_rate",
    "malicious_validator_fraction",
)


def aggregate_summaries(
    paths: Iterable[str | Path], require_confirmatory_seeds: bool = False
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        key = tuple(summary.get(field) for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(summary)
    rows = []
    for key, summaries in sorted(groups.items(), key=lambda item: str(item[0])):
        observed_seeds = {int(summary["seed"]) for summary in summaries}
        if require_confirmatory_seeds and observed_seeds != set(SEEDS):
            raise ValueError(
                f"confirmatory seeds incomplete for {key}: {sorted(observed_seeds)}"
            )
        utilities = [float(summary["final_utility"]) for summary in summaries]
        aucs = [
            learning_curve_auc([float(value) for value in summary["accuracy_history"]])
            for summary in summaries
        ]
        row = dict(zip(GROUP_FIELDS, key))
        row["seed_count"] = len(summaries)
        row["seeds"] = sorted(observed_seeds)
        row["final_utility_mean"] = float(np.mean(utilities))
        row["learning_curve_auc_mean"] = float(np.mean(aucs))
        row["malicious_update_acceptance_rate_mean"] = float(
            np.mean(
                [
                    float(summary["malicious_update_acceptance_rate"])
                    for summary in summaries
                ]
            )
        )
        row["transmitted_bytes_mean"] = float(
            np.mean([float(summary["transmitted_bytes"]) for summary in summaries])
        )
        row["ledger_bytes_mean"] = float(
            np.mean([float(summary["ledger_bytes"]) for summary in summaries])
        )
        row["runtime_seconds_total_mean"] = float(
            np.mean(
                [float(summary["runtime_seconds_total"]) for summary in summaries]
            )
        )
        row["worst_client_utility_mean"] = float(
            np.mean([float(summary["worst_client_utility"]) for summary in summaries])
        )
        if len(utilities) >= 2:
            utility_ci = bootstrap_mean_ci(utilities, seed=20260816)
            auc_ci = bootstrap_mean_ci(aucs, seed=20260816)
            row["final_utility_ci_lower"] = utility_ci.lower
            row["final_utility_ci_upper"] = utility_ci.upper
            row["learning_curve_auc_ci_lower"] = auc_ci.lower
            row["learning_curve_auc_ci_upper"] = auc_ci.upper
        rows.append(row)
    return rows
