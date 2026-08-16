"""Validated, serializable experiment configuration."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    dataset: str
    num_clients: int
    participation_rate: float
    malicious_fraction: float = 0.0
    dirichlet_alpha: float | None = None
    rounds: int = 100
    local_epochs: int = 1
    seed: int = 42
    attack: str = "none"
    defense: str = "fedavg"
    batch_size: int = 64
    learning_rate: float = 0.01
    identity_false_accept_rate: float = 0.0
    identity_false_reject_rate: float = 0.0
    duplicate_identity_fraction: float = 0.0
    malicious_validator_fraction: float = 0.0
    synthetic_samples: int = 200
    input_dim: int = 8
    num_classes: int = 2
    server_samples_per_class: int = 10
    backdoor_target_label: int = 0
    backdoor_fraction: float = 0.3

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id cannot be empty")
        if re.fullmatch(r"[A-Za-z0-9._-]+", self.experiment_id) is None:
            raise ValueError(
                "experiment_id may contain only letters, digits, dot, underscore, and hyphen"
            )
        if self.num_clients < 1:
            raise ValueError("num_clients must be positive")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        if not 0 <= self.malicious_fraction <= 1:
            raise ValueError("malicious_fraction must be in [0, 1]")
        if self.dirichlet_alpha is not None and self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds and local_epochs must be positive")
        if self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive")
        if self.synthetic_samples < 1 or self.input_dim < 1 or self.num_classes < 2:
            raise ValueError("synthetic dataset dimensions are invalid")
        if self.server_samples_per_class < 1:
            raise ValueError("server_samples_per_class must be positive")
        if self.backdoor_target_label < 0:
            raise ValueError("backdoor_target_label cannot be negative")
        if not 0 < self.backdoor_fraction <= 1:
            raise ValueError("backdoor_fraction must be in (0, 1]")
        for name, value in (
            ("identity_false_accept_rate", self.identity_false_accept_rate),
            ("identity_false_reject_rate", self.identity_false_reject_rate),
            ("duplicate_identity_fraction", self.duplicate_identity_fraction),
            ("malicious_validator_fraction", self.malicious_validator_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def participants_per_round(self) -> int:
        return max(1, math.ceil(self.num_clients * self.participation_rate))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        if source.suffix.lower() == ".json":
            return cls.from_json(source)
        if source.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            with source.open("r", encoding="utf-8") as handle:
                return cls(**yaml.safe_load(handle))
        raise ValueError("configuration must be JSON or YAML")
