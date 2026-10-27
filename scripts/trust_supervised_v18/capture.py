"""Passive checkpoint capture around the unchanged, frozen current-quorum runner."""
from pathlib import Path

import torch

from scripts.trust_alignment_v17.runner import BenchmarkRunner
from scripts.trust_repair_v16_fixed.integrity import snapshot, state_hash, assert_hash


class CaptureLogger:
    def __init__(self, logger, live_model, folder, checkpoints):
        self.logger, self.live_model = logger, live_model
        self.folder, self.checkpoints = Path(folder), tuple(checkpoints)
        self.captured = {}

    def log(self, event_type, round_index, payload):
        self.logger.log(event_type, round_index, payload)
        if event_type != 'model_integrity' or round_index not in self.checkpoints:
            return
        state = snapshot(self.live_model[0])
        observed = state_hash(state)
        if not payload['passed'] or observed != payload['committed_hash']:
            raise ValueError('capture differs from integrity-checked committed model')
        if round_index in self.captured:
            raise ValueError('duplicate checkpoint')
        directory = self.folder/'checkpoints'
        directory.mkdir(exist_ok=True)
        torch.save(dict(round=round_index, model_hash=observed, state=state), directory/f'round-{round_index:03}.pt')
        assert_hash(self.live_model[0], observed, 'passive checkpoint capture')
        self.captured[round_index] = observed


def capture(config, folder, features, labels, factory, test_arrays, checkpoints):
    if not checkpoints or not set(checkpoints) <= set(range(1, config.rounds+1)):
        raise ValueError('checkpoint outside replay')
    runner = BenchmarkRunner(config, folder)
    runner.challenge_arm = 'current'
    live_model = []

    def wrapped_factory():
        model = factory()
        live_model.append(model)
        if len(live_model) != 1:
            raise ValueError('unexpected extra model factory invocation')
        return model

    observer = CaptureLogger(runner.logger, live_model, folder, checkpoints)
    runner.logger = observer
    runner.run_arrays(features, labels, wrapped_factory, test_arrays)
    if set(observer.captured) != set(checkpoints):
        raise ValueError('incomplete checkpoint capture')
    return observer.captured
