import numpy as np
import pytest

from scripts.research.analysis import bootstrap_mean_ci, holm_adjust, learning_curve_auc
from scripts.research.datasets import partition_indices, routing_targets_from_neighbors


def test_iid_partition_is_deterministic_and_assigns_each_sample_once():
    labels = np.array([0, 1] * 10)

    first = partition_indices(labels, num_clients=4, dirichlet_alpha=None, seed=13)
    second = partition_indices(labels, num_clients=4, dirichlet_alpha=None, seed=13)

    assert [part.tolist() for part in first] == [part.tolist() for part in second]
    assigned = np.concatenate(first)
    assert sorted(assigned.tolist()) == list(range(20))
    assert [len(part) for part in first] == [5, 5, 5, 5]


def test_dirichlet_partition_assigns_every_sample_exactly_once():
    labels = np.repeat(np.arange(4), 25)

    partitions = partition_indices(labels, num_clients=5, dirichlet_alpha=0.1, seed=29)

    assigned = np.concatenate(partitions)
    assert len(assigned) == 100
    assert len(np.unique(assigned)) == 100


def test_sift_routing_targets_count_neighbors_per_client():
    neighbor_indices = np.array([[0, 1, 3], [2, 4, 5]])
    shard_assignments = np.array([0, 0, 1, 2, 2, 2])

    targets = routing_targets_from_neighbors(
        neighbor_indices, shard_assignments, num_clients=3
    )

    assert np.array_equal(targets, np.array([[2, 0, 1], [0, 1, 2]], dtype=np.int32))


def test_learning_curve_auc_uses_trapezoidal_area():
    assert learning_curve_auc([0.0, 1.0, 1.0]) == pytest.approx(1.5)


def test_bootstrap_ci_is_reproducible_and_contains_sample_mean():
    first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], seed=47, samples=1000)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], seed=47, samples=1000)

    assert first == second
    assert first.lower <= 2.5 <= first.upper


def test_holm_adjustment_preserves_order_and_caps_at_one():
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
