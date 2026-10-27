"""Deterministic data partitioning and SIFT routing-target utilities."""

import numpy as np


def partition_indices(
    labels: np.ndarray,
    num_clients: int,
    dirichlet_alpha: float | None,
    seed: int,
) -> list[np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if dirichlet_alpha is not None and dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    rng = np.random.default_rng(seed)
    if dirichlet_alpha is None:
        shuffled = rng.permutation(len(labels))
        return [part.astype(np.int64) for part in np.array_split(shuffled, num_clients)]

    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        counts = rng.multinomial(len(class_indices), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            partitions[client_id].extend(class_indices[start : start + count].tolist())
            start += count
    result = []
    for values in partitions:
        array = np.asarray(values, dtype=np.int64)
        rng.shuffle(array)
        result.append(array)
    return result


def routing_targets_from_neighbors(
    neighbor_indices: np.ndarray,
    shard_assignments: np.ndarray,
    num_clients: int,
) -> np.ndarray:
    neighbors = np.asarray(neighbor_indices)
    shards = np.asarray(shard_assignments)
    if neighbors.ndim != 2:
        raise ValueError("neighbor_indices must have shape [queries, top_k]")
    if shards.ndim != 1:
        raise ValueError("shard_assignments must be one-dimensional")
    if neighbors.size and (neighbors.min() < 0 or neighbors.max() >= len(shards)):
        raise ValueError("neighbor index is outside shard assignments")
    if shards.size and (shards.min() < 0 or shards.max() >= num_clients):
        raise ValueError("shard assignment is outside client range")
    targets = np.zeros((neighbors.shape[0], num_clients), dtype=np.int32)
    for query_index, row in enumerate(neighbors):
        targets[query_index] = np.bincount(shards[row], minlength=num_clients)
    return targets
