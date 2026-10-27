"""Aggregation baselines used by the stress-test harness."""

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as functional

from .contracts import AggregationResult, ClientUpdate, StateDict, VoteDecision


@dataclass(frozen=True)
class AggregationContext:
    round_index: int
    validation_decisions: dict[int, VoteDecision]


def _floating_keys(updates: list[ClientUpdate]) -> list[str]:
    if not updates:
        raise ValueError("at least one update is required")
    return [
        name
        for name, value in updates[0].state.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]


def _mean_selected(updates: list[ClientUpdate], selected: Iterable[int]) -> dict[str, object]:
    chosen = [updates[index] for index in selected]
    keys = _floating_keys(chosen)
    result: dict[str, object] = dict(chosen[0].state)
    for key in keys:
        result[key] = torch.stack([item.state[key] for item in chosen]).mean(dim=0)
    return result


def _flatten(update: ClientUpdate) -> torch.Tensor:
    parts = [update.state[key].reshape(-1) for key in _floating_keys([update])]
    if not parts:
        raise ValueError("update contains no floating-point tensors")
    return torch.cat(parts)


def flatten_update(update: ClientUpdate) -> torch.Tensor:
    """Flatten floating-point update tensors in state-dictionary order."""
    return _flatten(update)


def _state_from_vector(vector: torch.Tensor, template: ClientUpdate) -> dict[str, object]:
    state: dict[str, object] = dict(template.state)
    offset = 0
    for key in _floating_keys([template]):
        value = template.state[key]
        length = value.numel()
        state[key] = vector[offset : offset + length].reshape(value.shape).to(value.dtype)
        offset += length
    if offset != vector.numel():
        raise ValueError("vector length does not match update state")
    return state


class FedAvgDefense:
    def aggregate(self, global_state: StateDict, updates: list[ClientUpdate], context) -> AggregationResult:
        keys = _floating_keys(updates)
        total = sum(update.sample_count for update in updates)
        if total <= 0:
            raise ValueError("total sample count must be positive")
        state: dict[str, object] = dict(updates[0].state)
        for key in keys:
            state[key] = sum(
                update.state[key] * (update.sample_count / total) for update in updates
            )
        return AggregationResult(state, tuple(update.client_id for update in updates))


class CoordinateMedianDefense:
    def aggregate(self, global_state: StateDict, updates: list[ClientUpdate], context) -> AggregationResult:
        keys = _floating_keys(updates)
        state: dict[str, object] = dict(updates[0].state)
        for key in keys:
            state[key] = torch.stack([update.state[key] for update in updates]).median(dim=0).values
        return AggregationResult(state, tuple(update.client_id for update in updates))


class TrimmedMeanDefense:
    def __init__(self, trim_ratio: float = 0.2):
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio

    def aggregate(self, global_state: StateDict, updates: list[ClientUpdate], context) -> AggregationResult:
        keys = _floating_keys(updates)
        trim = math.floor(len(updates) * self.trim_ratio)
        if len(updates) - 2 * trim < 1:
            raise ValueError("trim ratio removes every update")
        state: dict[str, object] = dict(updates[0].state)
        for key in keys:
            values = torch.stack([update.state[key] for update in updates]).sort(dim=0).values
            kept = values[trim : len(updates) - trim] if trim else values
            state[key] = kept.mean(dim=0)
        return AggregationResult(state, tuple(update.client_id for update in updates))


class MultiKrumDefense:
    def __init__(self, byzantine_count: int, select_count: int = 1):
        if byzantine_count < 0 or select_count < 1:
            raise ValueError("invalid Multi-Krum parameters")
        self.byzantine_count = byzantine_count
        self.select_count = select_count

    def aggregate(self, global_state: StateDict, updates: list[ClientUpdate], context) -> AggregationResult:
        count = len(updates)
        neighbor_count = count - self.byzantine_count - 2
        if count < 2 * self.byzantine_count + 3 or neighbor_count < 1:
            raise ValueError("Multi-Krum requires n >= 2f + 3")
        vectors = torch.stack([_flatten(update) for update in updates])
        distances = torch.cdist(vectors, vectors, p=2).pow(2)
        scores = []
        for index in range(count):
            peers = torch.cat((distances[index, :index], distances[index, index + 1 :]))
            scores.append(peers.topk(neighbor_count, largest=False).values.sum())
        selected_count = min(self.select_count, neighbor_count)
        selected = torch.stack(scores).topk(selected_count, largest=False).indices.tolist()
        accepted = tuple(updates[index].client_id for index in selected)
        rejected = tuple(update.client_id for index, update in enumerate(updates) if index not in selected)
        return AggregationResult(_mean_selected(updates, selected), accepted, rejected)


class FLTrustDefense:
    def __init__(self, trusted_update: StateDict):
        self.trusted_update = trusted_update

    def aggregate(self, global_state: StateDict, updates: list[ClientUpdate], context) -> AggregationResult:
        trusted = ClientUpdate(-1, self.trusted_update, 1)
        trusted_vector = _flatten(trusted)
        trusted_norm = trusted_vector.norm().clamp_min(1e-12)
        vectors = [_flatten(update) for update in updates]
        weights = [functional.cosine_similarity(vector, trusted_vector, dim=0).clamp_min(0) for vector in vectors]
        weight_sum = torch.stack(weights).sum()
        if weight_sum <= 0:
            raise ValueError("no update has positive trust alignment")
        state: dict[str, object] = dict(updates[0].state)
        keys = _floating_keys(updates)
        for key in keys:
            normalized = []
            for update, vector in zip(updates, vectors):
                scale = trusted_norm / vector.norm().clamp_min(1e-12)
                normalized.append(update.state[key] * scale)
            state[key] = sum(value * weight for value, weight in zip(normalized, weights)) / weight_sum
        accepted = tuple(update.client_id for update, weight in zip(updates, weights) if weight > 0)
        rejected = tuple(update.client_id for update, weight in zip(updates, weights) if weight <= 0)
        return AggregationResult(state, accepted, rejected, {"trust_weights": tuple(float(w) for w in weights)})


class TwoLayerDefense:
    def __init__(
        self,
        initial_trust: float = 0.5,
        trust_reward: float = 0.05,
        trust_penalty: float = 0.1,
    ):
        if not 0 <= initial_trust <= 1:
            raise ValueError("initial_trust must be in [0, 1]")
        if trust_reward < 0 or trust_penalty < 0:
            raise ValueError("trust updates cannot be negative")
        self.initial_trust = initial_trust
        self.trust_reward = trust_reward
        self.trust_penalty = trust_penalty
        self._trust: dict[int, float] = {}

    def trust_score(self, client_id: int) -> float:
        return self._trust.get(client_id, self.initial_trust)

    def set_trust_score(self, client_id: int, score: float) -> None:
        if not 0 <= score <= 1:
            raise ValueError("trust score must be in [0, 1]")
        self._trust[client_id] = score

    def aggregate(
        self,
        global_state: StateDict,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> AggregationResult:
        if context is None:
            raise ValueError("two-layer defense requires validation decisions")
        missing = [
            update.client_id
            for update in updates
            if update.client_id not in context.validation_decisions
        ]
        if missing:
            raise ValueError(f"missing validation decisions for clients {missing}")
        approved = [
            update
            for update in updates
            if context.validation_decisions[update.client_id].accepted
        ]
        if not approved:
            raise ValueError("committee rejected every update")
        keys = _floating_keys(approved)
        weights = [
            self.trust_score(update.client_id) * update.sample_count for update in approved
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("approved updates have zero aggregate trust weight")
        state: dict[str, object] = dict(approved[0].state)
        for key in keys:
            state[key] = sum(
                update.state[key] * (weight / total_weight)
                for update, weight in zip(approved, weights)
            )
        approved_ids = tuple(update.client_id for update in approved)
        rejected_ids = tuple(
            update.client_id for update in updates if update.client_id not in approved_ids
        )
        for update in updates:
            current = self.trust_score(update.client_id)
            if update.client_id in approved_ids:
                self._trust[update.client_id] = min(1.0, current + self.trust_reward)
            else:
                self._trust[update.client_id] = max(0.0, current - self.trust_penalty)
        return AggregationResult(
            state,
            approved_ids,
            rejected_ids,
            {"trust_weights": tuple(weights)},
        )


class BOBADefense:
    """Clean-room adapter of BOBA's published PCA and simplex filtering stages.

    The baseline requires one trusted reference update per class. This dependency
    is intentionally explicit so comparisons do not hide BOBA's server-data
    assumption.
    """

    source_url = "https://proceedings.mlr.press/v238/bao24a.html"

    def __init__(
        self,
        byzantine_count: int,
        server_vectors: torch.Tensor,
        p_min: float = 0.0,
        max_iterations: int = 20,
    ):
        if byzantine_count < 0:
            raise ValueError("byzantine_count cannot be negative")
        if server_vectors.ndim != 2 or server_vectors.shape[0] < 2:
            raise ValueError("server_vectors must contain at least two class vectors")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.byzantine_count = byzantine_count
        self.server_vectors = server_vectors.float()
        self.p_min = p_min
        self.max_iterations = max_iterations

    @staticmethod
    def _fit_subspace(matrix: torch.Tensor, components: int):
        mean = matrix.mean(dim=0)
        centered = matrix - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:components]
        return mean, basis

    @staticmethod
    def _project(matrix: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor):
        latent = (matrix - mean) @ basis.T
        reconstructed = latent @ basis + mean
        return latent, reconstructed

    def aggregate(self, global_state: StateDict, updates: list[ClientUpdate], context) -> AggregationResult:
        if not updates:
            raise ValueError("at least one update is required")
        matrix = torch.stack([_flatten(update).float() for update in updates])
        if self.server_vectors.shape[1] != matrix.shape[1]:
            raise ValueError("server vectors and client updates have different dimensions")
        keep_count = len(updates) - self.byzantine_count
        components = min(self.server_vectors.shape[0] - 1, matrix.shape[1])
        if keep_count < components + 1:
            raise ValueError("insufficient clients for BOBA subspace estimation")

        mean, basis = self._fit_subspace(self.server_vectors, components)
        previous: tuple[int, ...] | None = None
        selected: tuple[int, ...] = ()
        for _ in range(self.max_iterations):
            _, reconstructed = self._project(matrix, mean, basis)
            distances = (matrix - reconstructed).pow(2).sum(dim=1)
            selected = tuple(
                sorted(distances.topk(keep_count, largest=False).indices.tolist())
            )
            if selected == previous:
                break
            previous = selected
            mean, basis = self._fit_subspace(matrix[list(selected)], components)

        latent, _ = self._project(matrix, mean, basis)
        server_latent, _ = self._project(self.server_vectors, mean, basis)
        latent_one = torch.cat([latent, torch.ones(len(latent), 1)], dim=1)
        server_one = torch.cat(
            [server_latent, torch.ones(len(server_latent), 1)], dim=1
        )
        coefficients = latent_one @ torch.linalg.pinv(server_one)
        minimum_coefficients = coefficients.min(dim=1).values
        accepted_indices = torch.where(minimum_coefficients >= self.p_min)[0].tolist()
        if len(accepted_indices) < keep_count:
            accepted_indices = minimum_coefficients.topk(keep_count).indices.tolist()
        accepted_indices = sorted(accepted_indices)
        # BOBA reconstructs the accepted latent mean on the fitted affine
        # subspace; averaging original vectors would undo stage-one filtering.
        aggregate_vector = latent[accepted_indices].mean(dim=0) @ basis + mean
        accepted = tuple(updates[index].client_id for index in accepted_indices)
        rejected = tuple(
            update.client_id
            for index, update in enumerate(updates)
            if index not in accepted_indices
        )
        return AggregationResult(
            _state_from_vector(aggregate_vector, updates[0]),
            accepted,
            rejected,
            {
                "source": self.source_url,
                "minimum_coefficients": tuple(
                    float(value) for value in minimum_coefficients
                ),
            },
        )
