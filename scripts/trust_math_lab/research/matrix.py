"""Preregistered experiment matrices for the journal study."""

from itertools import product

from .config import ExperimentConfig


SEEDS = (13, 29, 47, 71, 101)
DEFENSES = (
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
    "multi_krum",
    "fltrust",
    "boba",
    "two_layer",
)
CORE_ATTACKS = ("sign_flip", "alie", "model_replacement", "low_and_slow")


class ExperimentMatrix:
    def _config(self, prefix: str, **changes) -> ExperimentConfig:
        defaults = {
            "dataset": "fashion_mnist",
            "num_clients": 50,
            "participation_rate": 0.2,
            "malicious_fraction": 0.3,
            "dirichlet_alpha": 0.5,
            "rounds": 100,
            "local_epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.01,
            "seed": 13,
            "attack": "sign_flip",
            "defense": "fedavg",
        }
        defaults.update(changes)
        alpha = defaults.get("dirichlet_alpha")
        identity = "-".join(
            [
                prefix,
                str(defaults["dataset"]),
                str(defaults["defense"]),
                str(defaults["attack"]),
                f"n{defaults['num_clients']}",
                f"a{'iid' if alpha is None else alpha}",
                f"m{defaults['malicious_fraction']}",
                f"s{defaults['seed']}",
            ]
        )
        return ExperimentConfig(experiment_id=identity, **defaults)

    def screening(self) -> list[ExperimentConfig]:
        attacks = ("none", "sign_flip", "gaussian", "alie", "model_replacement", "low_and_slow")
        return [
            self._config("screen", defense=defense, attack=attack, seed=seed)
            for defense, attack, seed in product(DEFENSES, attacks, SEEDS[:3])
        ]

    def core_confirmation(self) -> list[ExperimentConfig]:
        return [
            self._config(
                "core", dataset=dataset, defense=defense, attack=attack, seed=seed
            )
            for dataset, defense, attack, seed in product(
                ("fashion_mnist", "cifar10"), DEFENSES, CORE_ATTACKS, SEEDS
            )
        ]

    def scale_sweep(self) -> list[ExperimentConfig]:
        attacks = ("model_replacement", "low_and_slow")
        return [
            self._config(
                "scale",
                num_clients=num_clients,
                defense=defense,
                attack=attack,
                seed=seed,
            )
            for num_clients, defense, attack, seed in product(
                (20, 50, 80), DEFENSES, attacks, SEEDS
            )
        ]

    def heterogeneity_sweep(self) -> list[ExperimentConfig]:
        attacks = ("model_replacement", "low_and_slow")
        return [
            self._config(
                "heterogeneity",
                dirichlet_alpha=alpha,
                defense=defense,
                attack=attack,
                seed=seed,
            )
            for alpha, defense, attack, seed in product(
                (None, 1.0, 0.5, 0.1), DEFENSES, attacks, SEEDS
            )
        ]

    def attacker_fraction_sweep(self) -> list[ExperimentConfig]:
        attacks = ("model_replacement", "low_and_slow")
        return [
            self._config(
                "attackers",
                malicious_fraction=fraction,
                defense=defense,
                attack=attack,
                seed=seed,
            )
            for fraction, defense, attack, seed in product(
                (0.1, 0.3, 0.5), DEFENSES, attacks, SEEDS
            )
        ]

    def identity_committee_sweep(self) -> list[ExperimentConfig]:
        return [
            self._config(
                "identity",
                defense="two_layer",
                attack=attack,
                identity_false_accept_rate=false_accept,
                identity_false_reject_rate=false_reject,
                malicious_validator_fraction=validator_fraction,
                seed=seed,
            )
            for false_accept, false_reject, validator_fraction, attack, seed in product(
                (0.0, 0.05, 0.1, 0.2),
                (0.0, 0.05, 0.1),
                (0.0, 0.33, 0.5),
                ("sign_flip", "low_and_slow"),
                SEEDS,
            )
        ]

    def sift_case_study(self) -> list[ExperimentConfig]:
        return [
            self._config(
                "sift",
                dataset="sift_routing",
                num_clients=num_clients,
                rounds=50,
                defense=defense,
                attack=attack,
                seed=seed,
            )
            for num_clients, defense, attack, seed in product(
                (20, 50, 80),
                ("fedavg", "multi_krum", "fltrust", "two_layer"),
                ("sign_flip", "alie", "low_and_slow"),
                SEEDS,
            )
        ]
