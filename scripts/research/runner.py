"""Deterministic array-based federated benchmark runner."""

import copy
from dataclasses import asdict
import math
import platform
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .attacks import (
    ALIEAttack,
    AttackContext,
    GaussianAttack,
    LowAndSlowAttack,
    ModelReplacementAttack,
    SignFlipAttack,
)
from .accountability import AccountabilityManager
from .config import ExperimentConfig
from .committee import CommitteePolicy
from .contracts import ClientUpdate, VoteDecision
from .datasets import partition_indices
from .defenses import (
    CoordinateMedianDefense,
    AggregationContext,
    BOBADefense,
    FedAvgDefense,
    FLTrustDefense,
    MultiKrumDefense,
    TrimmedMeanDefense,
    TwoLayerDefense,
    flatten_update,
)
from .events import EventLogger
from .identity import IdentityOracle
from .ledger import AppendOnlyRecoveryLedger
from .detector import TrendDetector
from .validation import LocalValidator, VoteAggregator


ModelFactory = Callable[[], nn.Module]


class _NoAttack:
    def apply(self, update: ClientUpdate, context: AttackContext) -> ClientUpdate:
        return update


def _attack(name: str):
    attacks = {
        "none": _NoAttack,
        "sign_flip": SignFlipAttack,
        "gaussian": GaussianAttack,
        "model_replacement": ModelReplacementAttack,
        "alie": ALIEAttack,
        "low_and_slow": LowAndSlowAttack,
    }
    try:
        return attacks[name]()
    except KeyError as error:
        raise ValueError(f"unknown attack: {name}") from error


def _defense(
    name: str,
    malicious_count: int,
    trusted_update: dict[str, torch.Tensor] | None = None,
    server_vectors: torch.Tensor | None = None,
):
    if name == "fedavg":
        return FedAvgDefense()
    if name == "coordinate_median":
        return CoordinateMedianDefense()
    if name == "trimmed_mean":
        return TrimmedMeanDefense()
    if name == "multi_krum":
        return MultiKrumDefense(byzantine_count=malicious_count)
    if name == "two_layer":
        return TwoLayerDefense()
    if name == "fltrust":
        if trusted_update is None:
            raise ValueError("FLTrust requires a trusted root update")
        return FLTrustDefense(trusted_update)
    if name == "boba":
        if server_vectors is None:
            raise ValueError("BOBA requires class-wise trusted server vectors")
        return BOBADefense(malicious_count, server_vectors)
    raise ValueError(f"unknown or context-dependent defense: {name}")


def _state_delta(local: nn.Module, global_model: nn.Module) -> dict[str, torch.Tensor]:
    global_state = global_model.state_dict()
    return {
        name: value.detach().cpu() - global_state[name].detach().cpu()
        if value.is_floating_point()
        else value.detach().cpu().clone()
        for name, value in local.state_dict().items()
    }


def _apply_delta(model: nn.Module, delta: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    updated = {}
    for name, value in state.items():
        candidate = delta.get(name)
        updated[name] = value + candidate.to(value.device) if value.is_floating_point() else value
    model.load_state_dict(updated)


def _update_bytes(update: ClientUpdate) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in update.state.values()
        if isinstance(value, torch.Tensor)
    )


class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = EventLogger(
            self.output_dir / "events.jsonl", config.experiment_id, config.seed
        )

    def run_arrays(
        self, features: np.ndarray, labels: np.ndarray, model_factory: ModelFactory
    ) -> dict[str, object]:
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)
        if labels.ndim == 1:
            labels = labels.astype(np.int64)
            partition_labels = labels
            task = "classification"
        elif labels.ndim == 2:
            labels = labels.astype(np.float32)
            partition_labels = labels.argmax(axis=1)
            task = "routing"
        else:
            raise ValueError("labels must be class IDs or routing-target rows")
        if features.ndim < 2 or len(features) != len(labels):
            raise ValueError("features and labels have incompatible shapes")

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        model = model_factory()
        ledger = AppendOnlyRecoveryLedger()
        ledger.checkpoint(0, model.state_dict())
        ledger.append_model_commit(0, {"round": 0, "genesis": True})
        detector = TrendDetector()
        accountability = AccountabilityManager(penalty=5.0)
        accountability.initialize(
            {client_id: 20.0 for client_id in range(self.config.num_clients)}
        )
        partitions = partition_indices(
            partition_labels,
            self.config.num_clients,
            self.config.dirichlet_alpha,
            self.config.seed,
        )

        malicious_count = math.floor(
            self.config.num_clients * self.config.malicious_fraction
        )
        malicious_ids = set(
            rng.choice(self.config.num_clients, size=malicious_count, replace=False).tolist()
        )
        malicious_validator_count = math.floor(
            self.config.num_clients * self.config.malicious_validator_fraction
        )
        malicious_validator_ids = set(
            rng.choice(
                self.config.num_clients,
                size=malicious_validator_count,
                replace=False,
            ).tolist()
        )
        duplicate_count = math.floor(
            self.config.num_clients * self.config.duplicate_identity_fraction
        )
        duplicate_ids = set(
            rng.choice(self.config.num_clients, size=duplicate_count, replace=False).tolist()
        )
        oracle = IdentityOracle(
            self.config.identity_false_accept_rate,
            self.config.identity_false_reject_rate,
        )
        eligible = [
            client_id
            for client_id in range(self.config.num_clients)
            if oracle.enroll(client_id, client_id in duplicate_ids, rng).accepted
            and len(partitions[client_id]) > 0
        ]
        if not eligible:
            raise ValueError("identity enrollment left no eligible clients")

        self.logger.log(
            "environment",
            round_index=0,
            payload={
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "device": "cpu",
            },
        )
        attack = _attack(self.config.attack)
        expected_malicious = math.floor(
            self.config.participants_per_round * self.config.malicious_fraction
        )
        contextual_baseline = self.config.defense in {"fltrust", "boba"}
        defense = (
            None
            if contextual_baseline
            else _defense(self.config.defense, expected_malicious)
        )
        root_indices = self._trusted_root_indices(labels, rng)
        transmitted_bytes = 0
        submitted = 0
        accepted_malicious = 0
        accuracy_history: list[float] = []
        targeted_asr_history: list[float] = []
        recovery_count = 0
        wrongful_slashing_count = 0
        round_runtime_seconds: list[float] = []
        detection_tp = detection_fp = detection_fn = detection_tn = 0
        first_attack_round: int | None = None
        detection_delay: int | None = None

        for round_index in range(1, self.config.rounds + 1):
            round_started = time.perf_counter()
            participant_count = min(self.config.participants_per_round, len(eligible))
            participants = sorted(
                rng.choice(eligible, size=participant_count, replace=False).tolist()
            )
            updates: list[ClientUpdate] = []
            for client_id in participants:
                local_model = copy.deepcopy(model)
                local_features = features[partitions[client_id]]
                local_labels = labels[partitions[client_id]]
                if (
                    client_id in malicious_ids
                    and self.config.attack == "model_replacement"
                    and labels.ndim == 1
                ):
                    local_features, local_labels = self._poison_backdoor(
                        local_features,
                        local_labels,
                        round_index,
                        client_id,
                    )
                self._train_client(
                    local_model,
                    local_features,
                    local_labels,
                    round_index,
                    client_id,
                )
                update = ClientUpdate(
                    client_id,
                    _state_delta(local_model, model),
                    len(partitions[client_id]),
                    client_id in malicious_ids,
                )
                updates.append(update)

            honest_references = tuple(update for update in updates if not update.is_malicious)
            attacked_updates = []
            round_malicious = sum(update.is_malicious for update in updates)
            attack_active = round_malicious > 0 and self.config.attack != "none"
            if attack_active and first_attack_round is None:
                first_attack_round = round_index
            for update in updates:
                if update.is_malicious:
                    update = attack.apply(
                        update,
                        AttackContext(
                            round_index=round_index,
                            seed=self.config.seed,
                            participant_count=len(updates),
                            malicious_count=max(1, round_malicious),
                            peer_updates=honest_references,
                        ),
                    )
                attacked_updates.append(update)
                transmitted_bytes += _update_bytes(update)

            if contextual_baseline:
                trusted_update, server_vectors = self._trusted_reference_updates(
                    model,
                    features,
                    labels,
                    root_indices,
                    round_index,
                )
                defense = _defense(
                    self.config.defense,
                    expected_malicious,
                    trusted_update,
                    server_vectors,
                )

            aggregation_context = None
            committee_payload: dict[str, object] = {}
            if isinstance(defense, TwoLayerDefense):
                decisions = self._committee_decisions(
                    model,
                    attacked_updates,
                    features,
                    labels,
                    partitions,
                    eligible,
                    malicious_ids,
                    malicious_validator_ids,
                    rng,
                )
                aggregation_context = AggregationContext(round_index, decisions)
                committee_payload = {
                    str(client_id): asdict(decision)
                    for client_id, decision in decisions.items()
                }
            result = defense.aggregate(
                model.state_dict(), attacked_updates, aggregation_context
            )
            _apply_delta(model, dict(result.state))
            ledger.checkpoint(round_index, model.state_dict())
            affirmative_signers = tuple(
                sorted(
                    {
                        signer
                        for client_id in result.accepted_client_ids
                        for signer in (
                            aggregation_context.validation_decisions[
                                client_id
                            ].affirmative_signers
                            if aggregation_context is not None
                            else ()
                        )
                    }
                )
            )
            ledger.append_model_commit(
                round_index,
                {
                    "round": round_index,
                    "accepted_client_ids": list(result.accepted_client_ids),
                    "affirmative_signers": list(affirmative_signers),
                },
            )
            detection = None
            if isinstance(defense, TwoLayerDefense):
                monitor_score = self._decentralized_monitor_score(
                    model,
                    features,
                    labels,
                    partitions,
                    eligible,
                    malicious_validator_ids,
                    rng,
                )
                detection = detector.observe(monitor_score)
                if detection.detected:
                    restored_version = max(0, round_index - 1)
                    restored_state, _ = ledger.recover(
                        rejected_version=round_index,
                        restored_version=restored_version,
                        trigger=detection.reason or "trend_detector",
                        affirmative_signers=affirmative_signers,
                    )
                    model.load_state_dict(restored_state)
                    slashing = accountability.slash_affirmative_signers(
                        affirmative_signers, malicious_validator_ids
                    )
                    recovery_count += 1
                    wrongful_slashing_count += slashing.wrongful_slashing_count
                if detection.detected and attack_active:
                    detection_tp += 1
                    if detection_delay is None and first_attack_round is not None:
                        detection_delay = round_index - first_attack_round
                elif detection.detected:
                    detection_fp += 1
                elif attack_active:
                    detection_fn += 1
                else:
                    detection_tn += 1
            accepted = set(result.accepted_client_ids)
            submitted += round_malicious
            accepted_malicious += len(accepted & malicious_ids)
            accuracy = self._accuracy(model, features, labels)
            accuracy_history.append(accuracy)
            targeted_asr = None
            if self.config.attack == "model_replacement" and labels.ndim == 1:
                targeted_asr = self._targeted_attack_success(model, features, labels)
                targeted_asr_history.append(targeted_asr)
            round_runtime = time.perf_counter() - round_started
            round_runtime_seconds.append(round_runtime)
            self.logger.log(
                "round_completed",
                round_index,
                {
                    "participants": participants,
                    "accepted_client_ids": list(result.accepted_client_ids),
                    "malicious_submitted": round_malicious,
                    "malicious_accepted": len(accepted & malicious_ids),
                    "accuracy": accuracy,
                    "targeted_attack_success": targeted_asr,
                    "transmitted_bytes": sum(_update_bytes(item) for item in attacked_updates),
                    "committee_decisions": committee_payload,
                    "detection": asdict(detection) if detection is not None else None,
                    "runtime_seconds": round_runtime,
                },
            )

        client_utilities = [
            self._accuracy(model, features[indices], labels[indices])
            for indices in partitions
            if len(indices) > 0
        ]
        detector_enabled = isinstance(defense, TwoLayerDefense)

        summary: dict[str, object] = {
            "experiment_id": self.config.experiment_id,
            "dataset": self.config.dataset,
            "attack": self.config.attack,
            "defense": self.config.defense,
            "seed": self.config.seed,
            "num_clients": self.config.num_clients,
            "participation_rate": self.config.participation_rate,
            "malicious_fraction": self.config.malicious_fraction,
            "dirichlet_alpha": self.config.dirichlet_alpha,
            "identity_false_accept_rate": self.config.identity_false_accept_rate,
            "identity_false_reject_rate": self.config.identity_false_reject_rate,
            "malicious_validator_fraction": self.config.malicious_validator_fraction,
            "task": task,
            "rounds_completed": self.config.rounds,
            "malicious_client_ids": sorted(malicious_ids),
            "malicious_validator_ids": sorted(malicious_validator_ids),
            "eligible_client_ids": eligible,
            "malicious_updates_submitted": submitted,
            "malicious_updates_accepted": accepted_malicious,
            "malicious_update_acceptance_rate": accepted_malicious / submitted if submitted else 0.0,
            "final_accuracy": accuracy_history[-1],
            "final_utility": accuracy_history[-1],
            "accuracy_history": accuracy_history,
            "targeted_attack_success_history": targeted_asr_history,
            "mean_targeted_attack_success": (
                sum(targeted_asr_history) / len(targeted_asr_history)
                if targeted_asr_history
                else None
            ),
            "transmitted_bytes": transmitted_bytes,
            "ledger_entries": len(ledger.entries),
            "ledger_bytes": ledger.serialized_size_bytes(),
            "recovery_count": recovery_count,
            "wrongful_slashing_count": wrongful_slashing_count,
            "trusted_server_samples": len(root_indices),
            "round_runtime_seconds": round_runtime_seconds,
            "runtime_seconds_total": sum(round_runtime_seconds),
            "worst_client_utility": min(client_utilities),
            "mean_client_utility": sum(client_utilities) / len(client_utilities),
            "detection_true_positive_rate": (
                detection_tp / (detection_tp + detection_fn)
                if detector_enabled and detection_tp + detection_fn
                else (0.0 if detector_enabled else None)
            ),
            "detection_false_positive_rate": (
                detection_fp / (detection_fp + detection_tn)
                if detector_enabled and detection_fp + detection_tn
                else (0.0 if detector_enabled else None)
            ),
            "detection_delay_rounds": detection_delay,
        }
        self.logger.log("experiment_completed", self.config.rounds, summary)
        return summary

    def _poison_backdoor(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        round_index: int,
        client_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        poisoned_features = features.copy()
        poisoned_labels = labels.copy()
        count = max(1, math.ceil(len(labels) * self.config.backdoor_fraction))
        rng = np.random.default_rng(
            self.config.seed * 1_000_003 + round_index * 1_009 + client_id
        )
        selected = rng.choice(len(labels), size=min(count, len(labels)), replace=False)
        poisoned_features[selected] = self._apply_trigger(poisoned_features[selected])
        poisoned_labels[selected] = self.config.backdoor_target_label
        return poisoned_features, poisoned_labels

    @staticmethod
    def _apply_trigger(features: np.ndarray) -> np.ndarray:
        triggered = features.copy()
        if triggered.ndim == 4:
            triggered[:, :, -2:, -2:] = 1.0
        else:
            triggered[..., -1] = triggered[..., -1] + 5.0
        return triggered

    def _targeted_attack_success(
        self, model: nn.Module, features: np.ndarray, labels: np.ndarray
    ) -> float:
        eligible = labels != self.config.backdoor_target_label
        if not np.any(eligible):
            return 0.0
        triggered = self._apply_trigger(features[eligible])
        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(triggered)).argmax(dim=1).numpy()
        return float((predictions == self.config.backdoor_target_label).mean())

    def _trusted_root_indices(
        self, labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        classes = labels if labels.ndim == 1 else labels.argmax(axis=1)
        selected = []
        for class_id in np.unique(classes):
            candidates = np.flatnonzero(classes == class_id).copy()
            rng.shuffle(candidates)
            selected.extend(candidates[: self.config.server_samples_per_class].tolist())
        return np.asarray(sorted(selected), dtype=np.int64)

    def _trusted_reference_updates(
        self,
        model: nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        root_indices: np.ndarray,
        round_index: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        root_model = copy.deepcopy(model)
        self._train_client(
            root_model,
            features[root_indices],
            labels[root_indices],
            round_index,
            -1,
        )
        trusted_state = _state_delta(root_model, model)
        classes = labels if labels.ndim == 1 else labels.argmax(axis=1)
        server_vectors = []
        for class_id in np.unique(classes[root_indices]):
            class_indices = root_indices[classes[root_indices] == class_id]
            class_model = copy.deepcopy(model)
            self._train_client(
                class_model,
                features[class_indices],
                labels[class_indices],
                round_index,
                -2 - int(class_id),
            )
            class_update = ClientUpdate(
                -2 - int(class_id),
                _state_delta(class_model, model),
                len(class_indices),
            )
            server_vectors.append(flatten_update(class_update))
        return trusted_state, torch.stack(server_vectors)

    def _decentralized_monitor_score(
        self,
        model: nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        partitions: list[np.ndarray],
        eligible: list[int],
        malicious_validator_ids: set[int],
        rng: np.random.Generator,
    ) -> float:
        committee = CommitteePolicy().for_population(len(eligible))
        validators = rng.choice(
            eligible, size=committee.size, replace=False
        ).tolist()
        reports = []
        for validator_id in validators:
            if validator_id in malicious_validator_ids:
                reports.append(1.0)
            else:
                indices = partitions[validator_id]
                reports.append(
                    self._accuracy(model, features[indices], labels[indices])
                )
        return float(np.median(reports))

    def _committee_decisions(
        self,
        model: nn.Module,
        updates: list[ClientUpdate],
        features: np.ndarray,
        labels: np.ndarray,
        partitions: list[np.ndarray],
        eligible: list[int],
        malicious_ids: set[int],
        malicious_validator_ids: set[int],
        rng: np.random.Generator,
    ) -> dict[int, VoteDecision]:
        decisions: dict[int, VoteDecision] = {}
        policy = CommitteePolicy()
        for update in updates:
            pool = [client_id for client_id in eligible if client_id != update.client_id]
            committee = policy.for_population(len(pool))
            if not committee.consensus_supported:
                decisions[update.client_id] = VoteDecision(False, committee.quorum, (), 0.0)
                continue
            validators = sorted(
                rng.choice(pool, size=committee.size, replace=False).tolist()
            )
            votes = []
            for validator_id in validators:
                indices = partitions[validator_id]
                baseline = self._accuracy(model, features[indices], labels[indices])
                if validator_id in malicious_validator_ids:
                    candidate_score = (
                        baseline
                        if update.client_id in malicious_ids or not malicious_ids
                        else baseline - max(abs(baseline), 1.0)
                    )
                else:
                    candidate_model = copy.deepcopy(model)
                    _apply_delta(candidate_model, dict(update.state))
                    candidate_score = self._accuracy(
                        candidate_model, features[indices], labels[indices]
                    )
                votes.append(
                    LocalValidator(validator_id).vote(
                        update.client_id,
                        baseline,
                        candidate_score,
                        validator_id in malicious_validator_ids,
                    )
                )
            decisions[update.client_id] = VoteAggregator().decide(
                votes, committee.quorum
            )
        return decisions

    def _train_client(
        self,
        model: nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        round_index: int,
        client_id: int,
    ) -> None:
        dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
        generator = torch.Generator().manual_seed(
            self.config.seed * 1_000_003 + round_index * 1_009 + client_id
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
        model.train()
        for _ in range(self.config.local_epochs):
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                logits = model(batch_features)
                if batch_labels.ndim == 1:
                    loss = nn.functional.cross_entropy(logits, batch_labels)
                else:
                    target_mass = batch_labels.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    target_distribution = batch_labels / target_mass
                    loss = nn.functional.mse_loss(
                        torch.softmax(logits, dim=1), target_distribution
                    )
                loss.backward()
                optimizer.step()

    @staticmethod
    def _accuracy(model: nn.Module, features: np.ndarray, labels: np.ndarray) -> float:
        model.eval()
        with torch.no_grad():
            outputs = model(torch.from_numpy(features))
            if labels.ndim == 1:
                predictions = outputs.argmax(dim=1).numpy()
                return float((predictions == labels).mean())
            target = torch.from_numpy(labels)
            budget = max(1, math.ceil(outputs.shape[1] * 0.3))
            selected = outputs.topk(budget, dim=1).indices
            captured = target.gather(1, selected).sum(dim=1)
            total = target.sum(dim=1).clamp_min(1e-12)
            return float((captured / total).mean().item())
