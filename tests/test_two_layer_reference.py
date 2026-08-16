import numpy as np
import torch

from scripts.research.contracts import ClientUpdate, VoteDecision
from scripts.research.defenses import AggregationContext, TwoLayerDefense
from scripts.research.detector import TrendDetector


def update(client_id, value, samples=1):
    return ClientUpdate(
        client_id=client_id,
        state={"weight": torch.tensor([value], dtype=torch.float32)},
        sample_count=samples,
    )


def decision(accepted, signers=()):
    return VoteDecision(
        accepted=accepted,
        quorum=2,
        affirmative_signers=tuple(signers),
        median_normalized_delta=0.0,
    )


def test_two_layer_aggregates_only_quorum_approved_updates():
    defense = TwoLayerDefense(initial_trust=0.5, trust_reward=0.1, trust_penalty=0.2)
    context = AggregationContext(
        round_index=1,
        validation_decisions={0: decision(False), 1: decision(True, (2, 3))},
    )

    result = defense.aggregate({}, [update(0, 100.0), update(1, 2.0)], context)

    assert torch.equal(result.state["weight"], torch.tensor([2.0]))
    assert result.accepted_client_ids == (1,)
    assert result.rejected_client_ids == (0,)
    assert defense.trust_score(0) == 0.3
    assert defense.trust_score(1) == 0.6


def test_two_layer_weights_approved_updates_by_trust_and_sample_count():
    defense = TwoLayerDefense(initial_trust=0.5, trust_reward=0.0)
    defense.set_trust_score(0, 0.25)
    defense.set_trust_score(1, 0.75)
    context = AggregationContext(
        round_index=1,
        validation_decisions={0: decision(True), 1: decision(True)},
    )

    result = defense.aggregate({}, [update(0, 0.0), update(1, 4.0)], context)

    assert torch.allclose(result.state["weight"], torch.tensor([3.0]))


def test_trend_detector_ignores_benign_oscillation_but_detects_gradual_decline():
    benign = TrendDetector(window_size=5, relative_drop_threshold=0.1, slope_threshold=0.005)
    gradual = TrendDetector(window_size=5, relative_drop_threshold=0.1, slope_threshold=0.005)

    benign_results = [benign.observe(value) for value in [1.0, 0.99, 1.01, 1.0, 0.995]]
    gradual_results = [gradual.observe(value) for value in [1.0, 0.99, 0.98, 0.97, 0.96]]

    assert not any(result.detected for result in benign_results)
    assert gradual_results[-1].detected is True
    assert gradual_results[-1].reason == "negative_trend"
