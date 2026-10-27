"""Constrain-and-scale search using attacker-owned training examples only."""
import copy
import numpy as np
import torch
from .research.runner_helpers import _apply_delta


def constrained_replacement(runner, model, honest_delta, poisoned_delta, features, labels):
    c = runner.config
    mask = labels != c.backdoor_target_label
    if not np.any(mask):
        raise ValueError('attacker proxy has no non-target examples')
    # Fixed proxy cap; these are attacker-owned training examples, never committee/test data.
    x, y = features[:256], labels[:256]
    mask = y != c.backdoor_target_label
    if not np.any(mask):
        raise ValueError('capped attacker proxy has no non-target examples')
    baseline = runner._score(model, x, y)
    honest_norm = torch.cat([v.flatten() for v in honest_delta.values() if v.is_floating_point()]).norm()
    max_norm = c.adaptive_norm_ratio*float(honest_norm)
    options = []
    for scale in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        state = {k: v*scale if v.is_floating_point() else v for k, v in poisoned_delta.items()}
        norm = torch.cat([v.flatten() for v in state.values() if v.is_floating_point()]).norm()
        if norm > max_norm and norm > 0:
            state = {k: v*(max_norm/float(norm)) if v.is_floating_point() else v for k, v in state.items()}
        candidate = copy.deepcopy(model)
        _apply_delta(candidate, state)
        clean = runner._score(candidate, x, y)
        success = runner._score(candidate, runner._apply_trigger(x[mask]),
                                np.full(int(mask.sum()), c.backdoor_target_label, dtype=np.int64))
        if clean >= baseline-c.adaptive_clean_tolerance:
            options.append((success, -scale, state, clean))
    if not options:
        raise ValueError('adaptive search has no feasible candidate, including zero update')
    success, _, state, clean = max(options, key=lambda value: value[:2])
    return state, {'proxy_clean_accuracy': clean, 'proxy_baseline': baseline,
                   'proxy_targeted_asr': success, 'norm_bound': max_norm,
                   'knowledge': 'own training examples and public global model; no committee validation or held-out test'}
