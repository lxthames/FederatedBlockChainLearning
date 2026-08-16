from scripts.research.matrix import ExperimentMatrix


def test_core_matrix_is_frozen_to_two_datasets_seven_defenses_four_attacks_five_seeds():
    configs = ExperimentMatrix().core_confirmation()

    assert len(configs) == 2 * 7 * 4 * 5
    assert {config.dataset for config in configs} == {"fashion_mnist", "cifar10"}
    assert {config.num_clients for config in configs} == {50}
    assert {config.seed for config in configs} == {13, 29, 47, 71, 101}


def test_sensitivity_matrices_change_one_factor_at_a_time():
    matrix = ExperimentMatrix()

    scale = matrix.scale_sweep()
    heterogeneity = matrix.heterogeneity_sweep()
    attackers = matrix.attacker_fraction_sweep()

    assert {config.num_clients for config in scale} == {20, 50, 80}
    assert {config.dirichlet_alpha for config in scale} == {0.5}
    assert {config.dirichlet_alpha for config in heterogeneity} == {
        None,
        1.0,
        0.5,
        0.1,
    }
    assert {config.num_clients for config in heterogeneity} == {50}
    assert {config.malicious_fraction for config in attackers} == {0.1, 0.3, 0.5}


def test_identity_committee_matrix_covers_requested_error_and_capture_rates():
    configs = ExperimentMatrix().identity_committee_sweep()

    assert {config.identity_false_accept_rate for config in configs} == {
        0.0,
        0.05,
        0.1,
        0.2,
    }
    assert {config.identity_false_reject_rate for config in configs} == {
        0.0,
        0.05,
        0.1,
    }
    assert {config.malicious_validator_fraction for config in configs} == {
        0.0,
        0.33,
        0.5,
    }
    assert all(config.defense == "two_layer" for config in configs)


def test_sift_matrix_uses_requested_scales_attacks_and_rounds():
    configs = ExperimentMatrix().sift_case_study()

    assert {config.num_clients for config in configs} == {20, 50, 80}
    assert {config.attack for config in configs} == {
        "sign_flip",
        "alie",
        "low_and_slow",
    }
    assert {config.rounds for config in configs} == {50}
    assert {config.dataset for config in configs} == {"sift_routing"}
