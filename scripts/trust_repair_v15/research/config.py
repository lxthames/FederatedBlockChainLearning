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
    protocol_version: str = "trust-repair-v15"
    device: str = "auto"
    model: str = "default"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    evaluation_batch_size: int = 512
    detector: str = "trend"
    detector_window: int = 5
    relative_drop_threshold: float = 0.05
    slope_threshold: float = 0.005
    minimum_validation_delta: float = -0.05
    initial_trust: float = 0.5
    minimum_trust: float = 0.45
    trust_floor: float = 0.15
    trust_reward: float = 0.02
    trust_penalty: float = 0.15
    initial_stake: float = 20.0
    minimum_stake: float = 10.0
    slashing_penalty: float = 5.0
    ablation: str = "full"
    ledger_backend: str = "memory"
    rpc_url: str = "http://127.0.0.1:8545"
    deployment_path: str = "blockchain/deployment.json"
    routing_budget: int = 1
    gradual_ramp_rounds: int = 10
    gradual_max_norm_ratio: float = 1.5
    attack_start_round: int = 1
    benign_drift_round: int = 0
    benign_drift_magnitude: float = 0.0
    max_train_samples: int | None = None
    evidence_tier: str = "development"
    sampling_policy: str = "uniform"

    natural_splits_path: str | None = None
    recovery_policy: str = "protocol"
    intervention_round: int = 20
    intervention_checkpoint: int = 15
    validator_behavior: str = "approve_bad_reject_good"
    adaptive_clean_tolerance: float = 0.02
    adaptive_norm_ratio: float = 1.5
    validation_absolute_tolerance: float = 0.02
    rejection_trust_penalty: float = 0.02
    probation_weight_floor: float = 0.1
    optimizer_momentum: float = 0.0
    vote_rule: str = 'accuracy'
    loss_tolerance: float = 0.01
    loss_standard_errors: float = 1.64
    ranking_keep_fraction: float = 0.8
    ranking_norm_penalty: float = 0.001
    candidate_clip_ratio: float = 0.0
    recovery_benefit_guard: bool = True
    recovery_loss_margin: float = 0.02
    recovery_standard_errors: float = 2.0
    recovery_penalties_enabled: bool = False
    recovery_signal: str = 'accuracy_then_loss'
    flame_noise_multiplier: float = .001
    shield_min_class_samples: int = 10
    shield_max_class_samples: int = 30
    shield_contamination: float = .1
    archive_updates: bool = False
    validation_variant: str = 'original'
    validation_distribution: str = 'private'
    trust_weighting: str = 'dynamic'
    trust_eligibility: str = 'dynamic'
    trust_updates: str = 'dynamic'
    rejection_trust_policy: str = 'none'
    legacy_inconclusive_penalty: bool = False
    local_training_budget: str = 'epoch'
    local_steps: int = 4
    aggregation_mass_policy: str = 'renormalized'

    def __post_init__(self) -> None:
        if self.local_training_budget!='epoch' or self.local_epochs!=1 or self.aggregation_mass_policy!='renormalized':
            raise ValueError('V13 requires one epoch and renormalized aggregation')
        if self.local_training_budget not in {'epoch','fixed_steps'}:
            raise ValueError('unknown local training budget')
        if isinstance(self.local_steps,bool) or not isinstance(self.local_steps,int) or self.local_steps < 1:
            raise ValueError('local_steps must be a positive integer')
        if self.local_training_budget=='fixed_steps' and self.local_epochs!=1:
            raise ValueError('fixed steps requires local_epochs=1')
        if self.aggregation_mass_policy not in {'nominal','renormalized','replay'}:
            raise ValueError('unknown aggregation mass policy')
        if self.aggregation_mass_policy=='nominal' and self.vote_rule!='scaled_loss':
            raise ValueError('nominal mass requires scaled voting')
        if self.aggregation_mass_policy=='replay' and (self.vote_rule!='rank_loss' or self.ranking_keep_fraction!=1. or self.defense!='two_layer'):
            raise ValueError('replay damping requires clipping-only acceptance')
        if self.validation_distribution not in {'private', 'redistributed'}:
            raise ValueError('unknown validation distribution')
        if any(v not in {'dynamic', 'fixed'} for v in
               (self.trust_weighting, self.trust_eligibility, self.trust_updates)):
            raise ValueError('unknown trust channel mode')
        if self.rejection_trust_policy not in {'none', 'rank_rejection'}:
            raise ValueError('unknown rejection trust policy')
        if self.legacy_inconclusive_penalty and self.rejection_trust_policy != 'rank_rejection':
            raise ValueError('legacy penalty requires explicit rank-rejection diagnostic')
        if self.ablation != 'validation_only' or self.recovery_penalties_enabled or self.ledger_backend != 'memory':
            raise ValueError('V13 requires memory-backed validation-only execution')
        if self.validation_variant not in {'original','probe','macro_probe','margin_probe'}:
            raise ValueError('unknown V9 validation variant')
        if self.validation_variant!='original' and (self.vote_rule!='rank_loss' or self.ablation!='validation_only' or self.recovery_penalties_enabled):
            raise ValueError('V9 probes require rank voting, validation only and disabled monetary penalties')
        if self.flame_noise_multiplier<0 or not 0<self.shield_contamination<.5:
            raise ValueError('invalid literature defense parameters')
        if not 0<=self.shield_min_class_samples<self.shield_max_class_samples:
            raise ValueError('invalid class sample thresholds')
        if self.defense=='flshield_cluster' and self.malicious_validator_fraction:
            raise ValueError('this FLShield pilot does not implement malicious report injection')
        if self.recovery_signal not in {'accuracy_then_loss','paired_loss_quorum'}:
            raise ValueError('invalid recovery signal')
        if self.recovery_signal == 'paired_loss_quorum' and (self.detector != 'trend' or not self.recovery_benefit_guard):
            raise ValueError('paired loss quorum requires fixed trend windows and benefit checks')
        if min(self.recovery_loss_margin,self.recovery_standard_errors)<0:
            raise ValueError('invalid recovery confirmation parameters')
        if self.vote_rule not in {'accuracy','paired_loss','scaled_loss','rank_loss'}:
            raise ValueError('unknown experimental vote rule')
        if min(self.loss_tolerance,self.loss_standard_errors,self.ranking_norm_penalty,self.candidate_clip_ratio)<0:
            raise ValueError('invalid loss rule parameters')
        if not 0 < self.ranking_keep_fraction <= 1:
            raise ValueError('invalid rank fraction')
        if self.vote_rule=='scaled_loss' and (self.trust_updates != 'fixed' or self.trust_weighting != 'fixed' or self.trust_eligibility != 'fixed'):
            raise ValueError('scaled loss requires fixed update, weight and eligibility channels')
        if self.vote_rule=='scaled_loss' and (self.trust_reward!=0 or self.rejection_trust_penalty!=0 or self.ablation!='validation_only'):
            raise ValueError('scaled-loss screening requires fixed trust and no recovery')
        if not 0 <= self.validation_absolute_tolerance < 1:
            raise ValueError('invalid absolute validation tolerance')
        if not 0 <= self.rejection_trust_penalty <= 1 or not 0 < self.probation_weight_floor <= 1:
            raise ValueError('invalid probation parameters')
        if not 0 <= self.optimizer_momentum < 1:
            raise ValueError('invalid optimizer momentum')
        if self.recovery_policy not in {'protocol','continue','scheduled_plain','scheduled_accountable'}:
            raise ValueError('invalid recovery policy')
        if self.validator_behavior not in {'approve_bad_reject_good','false_recovery','withhold','honest'}:
            raise ValueError('invalid validator behavior')
        if not 0 <= self.intervention_checkpoint < self.intervention_round:
            raise ValueError('invalid recovery intervention interval')
        if self.adaptive_clean_tolerance < 0 or self.adaptive_norm_ratio <= 0:
            raise ValueError('invalid adaptive attack limits')
        if self.protocol_version != "trust-repair-v15":
            raise ValueError("unsupported protocol_version")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if self.model not in {"default", "wide_cnn", "emh_dnn", "mlp"}:
            raise ValueError("unsupported model")
        if self.detector not in {"trend", "adjacent"}:
            raise ValueError("unsupported detector")
        if self.ablation not in {"full", "validation_only", "no_penalties"}:
            raise ValueError("unsupported ablation")
        if self.ledger_backend not in {"memory", "evm"}:
            raise ValueError("unsupported ledger_backend")
        if not 0 < self.validation_fraction < 0.5 or not 0 < self.test_fraction < 0.5:
            raise ValueError("split fractions must be in (0, 0.5)")
        if not 0 <= self.trust_floor <= self.minimum_trust <= 1 or not 0 < self.initial_trust <= 1:
            raise ValueError("invalid trust parameters")
        if min(self.initial_stake, self.minimum_stake, self.slashing_penalty) <= 0:
            raise ValueError("stake parameters must be positive")
        if self.detector_window < 3 or min(self.relative_drop_threshold, self.slope_threshold) <= 0:
            raise ValueError("invalid detector parameters")
        if min(self.evaluation_batch_size, self.routing_budget, self.gradual_ramp_rounds, self.attack_start_round) < 1:
            raise ValueError("batch, budget, ramp and attack start must be positive")
        if self.gradual_max_norm_ratio <= 0 or min(self.trust_reward, self.trust_penalty, self.benign_drift_magnitude) < 0:
            raise ValueError("invalid attack or trust parameters")
        if self.max_train_samples is not None and self.max_train_samples < 1:
            raise ValueError("max_train_samples must be positive")
        if self.evidence_tier not in {"development", "confirmatory", "case_study"}:
            raise ValueError("invalid evidence tier")
        if self.sampling_policy not in {'uniform','bounded'}:
            raise ValueError('invalid sampling policy')
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
