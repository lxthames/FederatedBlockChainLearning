import torch

from scripts.research.attacks import (
    ALIEAttack,
    AttackContext,
    GaussianAttack,
    LowAndSlowAttack,
    ModelReplacementAttack,
    SignFlipAttack,
)
from scripts.research.contracts import ClientUpdate
from scripts.research.defenses import (
    BOBADefense,
    CoordinateMedianDefense,
    FedAvgDefense,
    FLTrustDefense,
    MultiKrumDefense,
    TrimmedMeanDefense,
)


def update(client_id, values, *, samples=1, malicious=False):
    return ClientUpdate(
        client_id=client_id,
        state={"weight": torch.tensor(values, dtype=torch.float32)},
        sample_count=samples,
        is_malicious=malicious,
    )


def test_sign_flip_negates_floating_point_update():
    attacked = SignFlipAttack().apply(
        update(3, [1.0, -2.0], malicious=True), AttackContext(round_index=1)
    )

    assert torch.equal(attacked.state["weight"], torch.tensor([-1.0, 2.0]))


def test_model_replacement_uses_participant_to_attacker_ratio():
    attacked = ModelReplacementAttack().apply(
        update(3, [1.0, 2.0], malicious=True),
        AttackContext(round_index=1, participant_count=10, malicious_count=2),
    )

    assert torch.equal(attacked.state["weight"], torch.tensor([5.0, 10.0]))


def test_alie_uses_peer_coordinate_mean_and_standard_deviation():
    peers = (update(0, [1.0, 3.0]), update(1, [3.0, 7.0]))
    attacked = ALIEAttack(z_max=0.5).apply(
        update(3, [0.0, 0.0], malicious=True),
        AttackContext(round_index=1, peer_updates=peers),
    )

    assert torch.allclose(attacked.state["weight"], torch.tensor([2.5, 6.0]))


def test_low_and_slow_ramps_without_exceeding_maximum_scale():
    attack = LowAndSlowAttack(max_scale=1.5, ramp_rounds=10)

    first = attack.apply(update(3, [2.0]), AttackContext(round_index=1))
    last = attack.apply(update(3, [2.0]), AttackContext(round_index=20))

    assert torch.allclose(first.state["weight"], torch.tensor([1.5]))
    assert torch.allclose(last.state["weight"], torch.tensor([-3.0]))
    assert last.state['weight'].norm() <= 1.5 * torch.tensor([2.0]).norm()


def test_gaussian_attack_is_reproducible_from_experiment_context():
    attack = GaussianAttack(stddev=0.1)
    context = AttackContext(round_index=2, seed=47)

    first = attack.apply(update(3, [1.0, 2.0]), context)
    second = attack.apply(update(3, [1.0, 2.0]), context)

    assert torch.equal(first.state["weight"], second.state["weight"])
    assert not torch.equal(first.state["weight"], torch.tensor([1.0, 2.0]))


def test_fedavg_weights_updates_by_sample_count():
    result = FedAvgDefense().aggregate(
        {"weight": torch.tensor([0.0])},
        [update(0, [1.0], samples=1), update(1, [3.0], samples=3)],
        None,
    )

    assert torch.allclose(result.state["weight"], torch.tensor([2.5]))
    assert result.accepted_client_ids == (0, 1)


def test_coordinate_median_rejects_single_extreme_update():
    result = CoordinateMedianDefense().aggregate(
        {}, [update(0, [1.0]), update(1, [2.0]), update(2, [100.0])], None
    )

    assert torch.equal(result.state["weight"], torch.tensor([2.0]))


def test_trimmed_mean_removes_one_value_from_each_tail():
    result = TrimmedMeanDefense(trim_ratio=0.2).aggregate(
        {},
        [
            update(0, [-100.0]),
            update(1, [1.0]),
            update(2, [2.0]),
            update(3, [3.0]),
            update(4, [100.0]),
        ],
        None,
    )

    assert torch.equal(result.state["weight"], torch.tensor([2.0]))


def test_multi_krum_selects_close_updates_and_rejects_outlier():
    result = MultiKrumDefense(byzantine_count=1, select_count=2).aggregate(
        {},
        [
            update(0, [0.0, 0.0]),
            update(1, [0.1, 0.0]),
            update(2, [0.0, 0.1]),
            update(3, [0.1, 0.1]),
            update(4, [8.0, 8.0]),
        ],
        None,
    )

    assert 4 not in result.accepted_client_ids
    assert 4 in result.rejected_client_ids


def test_fltrust_assigns_zero_weight_to_opposing_update():
    result = FLTrustDefense(trusted_update={"weight": torch.tensor([1.0, 0.0])}).aggregate(
        {}, [update(0, [1.0, 0.0]), update(1, [-2.0, 0.0])], None
    )

    assert torch.allclose(result.state["weight"], torch.tensor([1.0, 0.0]))
    assert result.accepted_client_ids == (0,)
    assert result.rejected_client_ids == (1,)


def test_boba_filters_update_outside_server_label_subspace():
    defense = BOBADefense(
        byzantine_count=1,
        server_vectors=torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
        p_min=0.0,
        max_iterations=10,
    )
    updates = [
        update(0, [1.0, 0.0]),
        update(1, [2.0, 0.0]),
        update(2, [3.0, 0.0]),
        update(3, [0.0, 100.0]),
    ]

    result = defense.aggregate({}, updates, None)

    assert 3 in result.rejected_client_ids
    assert torch.allclose(result.state["weight"], torch.tensor([2.0, 0.0]))
