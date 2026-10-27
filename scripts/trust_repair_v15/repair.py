"""Aggregate behavioral validation and bounded single-update omission repair.

Decisions use private validation only. Poisoning labels and test ASR are never inputs.
Calibration is empirical, and does not supply a false-positive/security guarantee.
"""
import copy
import math
from pathlib import Path
import numpy as np
import torch
from .loss_voting import paired_decision
from .research.contracts import AggregationResult, ClientUpdate
from .research.runner_helpers import _apply_delta, _update_bytes
from .research.splits import stream

CAP = 8
PANEL = 5
GRID = 7
QUANTILE = .995
EPS = 1e-6
ARMS = ('fedavg', 'clip_only', 'fedval', 'current', 'aggregate', 'repair')


def transformations(height, width):
    """Fixed spatial grid, including every image boundary; not exhaustive coverage."""
    result = []
    for size, value in ((2, 0.), (2, 1.), (4, .5)):
        if min(height, width) < size:
            raise ValueError('challenge images must be at least 4x4')
        rows = np.unique(np.linspace(0, height-size, GRID, dtype=int))
        cols = np.unique(np.linspace(0, width-size, GRID, dtype=int))
        result.extend((size, value, int(r), int(c)) for r in rows for c in cols)
    return result


def challenge_bank(images, seed, rnd, validator):
    if images.ndim != 4 or images.shape[1] != 1 or not len(images):
        raise ValueError('expected nonempty grayscale NCHW images')
    blocks = [images.copy()]
    for size, value, top, left in transformations(*images.shape[-2:]):
        block = images.copy()
        block[:, :, top:top+size, left:left+size] = value
        blocks.append(block)
    rng = stream(seed, 'repair-noise-v15', rnd, validator)
    blocks.append(np.clip(images+rng.normal(0., .08, images.shape).astype('float32'), 0., 1.))
    return np.concatenate(blocks).astype('float32')


def behavior(parent, candidate):
    if parent.shape != candidate.shape or parent.ndim != 3 or parent.shape[0] < 2:
        raise ValueError('expected paired transformation/example/class probability blocks')
    if not np.isfinite(parent).all() or not np.isfinite(candidate).all():
        raise ValueError('nonfinite model probabilities')
    before = (parent[1:]-parent[0]).mean(axis=1)
    after = (candidate[1:]-candidate[0]).mean(axis=1)
    incremental = after-before
    return dict(absolute=max(0., float(after.max())),
                incremental=max(0., float(incremental.max()))), before, after


def weighted_result(updates, omitted=None):
    kept = [u for u in updates if u.client_id != omitted]
    if not kept or any(u.sample_count <= 0 for u in kept):
        raise ValueError('repair requires positive sample weights and a nonempty aggregate')
    mass = sum(u.sample_count for u in kept)
    state = {k: sum(u.state[k]*(u.sample_count/mass) for u in kept)
             if v.is_floating_point() else torch.zeros_like(v)
             for k, v in kept[0].state.items()}
    return AggregationResult(state, tuple(u.client_id for u in kept),
                             () if omitted is None else (omitted,))


def choose_repair(original, candidates):
    """Screening only: require behavioral improvement, not merely a lower rank."""
    admissible = [(client, verdict) for client, verdict in candidates
                  if verdict['passed'] and verdict['risk'] < original['risk']-EPS]
    return min(admissible, key=lambda item: (item[1]['risk'], item[0])) if admissible else None


class RepairRound:
    def __init__(self, runner, parent, x, y, split, rnd, updates, eligible):
        self.runner, self.parent, self.x, self.y = runner, parent, x, y
        self.split, self.rnd, self.updates = split, rnd, updates
        self.arm = runner.challenge_arm
        if self.arm not in ARMS+('calibrate',):
            raise ValueError('unknown V15 arm')
        self.thresholds = runner.challenge_thresholds
        if self.arm in ('aggregate', 'repair'):
            if not self.thresholds:
                raise ValueError('frozen clean calibration required')
            for stage in ('screen', 'verify'):
                for metric in ('absolute', 'incremental', 'clean_excess'):
                    value = self.thresholds[stage][metric]
                    if not math.isfinite(value) or value < 0:
                        raise ValueError('invalid calibration threshold')
        self.extra_transmitted_bytes = 0
        self.cache = {}
        contributors = {u.client_id for u in updates}
        pool = sorted(set(eligible)-contributors)
        self.panels = None
        if self.arm in ('calibrate', 'aggregate', 'repair'):
            if len(pool) < 2*PANEL:
                raise ValueError('two disjoint noncontributor validator panels required')
            sampled = stream(runner.config.seed, 'repair-panels-v15', rnd).choice(pool, 2*PANEL, replace=False).tolist()
            self.panels = dict(screen=sorted(sampled[:PANEL]), verify=sorted(sampled[PANEL:]))

    def candidate(self, validator, client):
        return True

    def probabilities(self, model, bank, examples):
        work = self.runner.validation_work
        work['repair_example_evaluations'] = work.get('repair_example_evaluations', 0)+len(bank)
        batch = self.runner.config.evaluation_batch_size
        work['repair_forward_passes'] = work.get('repair_forward_passes', 0)+math.ceil(len(bank)/batch)
        model.eval()
        with torch.no_grad():
            values = [model(torch.as_tensor(bank[start:start+batch], device=self.runner.device)).softmax(1).cpu().numpy()
                      for start in range(0, len(bank), batch)]
        result = np.concatenate(values).reshape(-1, examples, self.runner.config.num_classes)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite challenge probabilities')
        return result

    def baseline(self, validator):
        if validator not in self.cache:
            ids = np.asarray(self.split.validation[validator])
            rng = stream(self.runner.config.seed, 'repair-examples-v15', self.rnd, validator)
            ids = ids[rng.permutation(len(ids))[:CAP]]
            if len(ids) < 2:
                raise ValueError('at least two private examples required')
            bank = challenge_bank(self.x[ids], self.runner.config.seed, self.rnd, validator)
            self.cache[validator] = (ids, bank, self.probabilities(self.parent, bank, len(ids)))
        return self.cache[validator]

    def evaluate(self, result, stage, omitted=None):
        model = copy.deepcopy(self.parent)
        _apply_delta(model, dict(result.state))
        members = self.panels[stage]
        self.extra_transmitted_bytes += (_update_bytes(ClientUpdate(-1, result.state, 0))+96)*len(members)
        votes, risks = [], []
        for validator in members:
            ids, bank, before = self.baseline(validator)
            after = self.probabilities(model, bank, len(ids))
            metrics, parent_contrast, contrast = behavior(before, after)
            labels = self.y[ids]
            ix = np.arange(len(ids))
            tiny = np.finfo('float32').tiny
            loss_before = -np.log(np.maximum(before[0, ix, labels], tiny))
            loss_after = -np.log(np.maximum(after[0, ix, labels], tiny))
            _, clean = paired_decision(loss_before, loss_after, 0., 2.)
            metrics['clean_excess'] = max(0., clean['mean_loss_change']-2*clean['standard_error'])
            thresholds = None if self.arm == 'calibrate' else self.thresholds[stage]
            ok = True if thresholds is None else all(metrics[k] <= thresholds[k] for k in metrics)
            risk = max(metrics[k]/max(thresholds[k], EPS) for k in ('absolute', 'incremental')) if thresholds else 0.
            votes.append(ok)
            risks.append(risk)
            self.runner.logger.log('repair_vote', self.rnd, dict(stage=stage, validator=validator,
                omitted=omitted, metrics=metrics, thresholds=thresholds, approve=ok,
                examples=len(ids), clean=clean, establishes_misconduct=False))
            # Full transformation/class contrasts support later diagnosis without private images.
            directory = self.runner.output_dir/'behavior'
            directory.mkdir(exist_ok=True)
            token = 'full' if omitted is None else f'omit-{omitted}'
            np.savez_compressed(directory/f'r{self.rnd:03}-{stage}-v{validator}-{token}.npz',
                parent_contrast=parent_contrast, candidate_contrast=contrast,
                clean_parent_mean=before[0].mean(axis=0), clean_candidate_mean=after[0].mean(axis=0))
        quorum = math.ceil(2*len(members)/3)
        verdict = dict(passed=sum(votes) >= quorum, affirmative=sum(votes), quorum=quorum,
                       risk=float(np.median(risks)), members=members)
        self.runner.logger.log('repair_panel', self.rnd, dict(stage=stage, omitted=omitted, **verdict))
        return verdict

    def archive(self, result, failed=False):
        attempted = getattr(self.runner, 'repair_archive_count', 0)
        if self.rnd not in (1, 11, 51, self.runner.config.rounds) and not (failed and attempted < 3):
            return
        folder = self.runner.output_dir/'checkpoints'
        folder.mkdir(exist_ok=True)
        payload = dict(round=self.rnd, parent={k:v.detach().cpu() for k,v in self.parent.state_dict().items()},
                       proposed_delta={k:v.detach().cpu() for k,v in result.state.items()},
                       accepted=list(result.accepted_client_ids))
        if failed and attempted < 3:
            payload['clipped_updates'] = [dict(client_id=u.client_id, sample_count=u.sample_count,
                state={k:v.detach().cpu() for k,v in u.state.items()}) for u in self.updates]
            self.runner.repair_archive_count = attempted+1
        torch.save(payload, folder/f'round-{self.rnd:03}.pt')

    def block(self, result, reason):
        self.runner.logger.log('repair_decision', self.rnd, dict(action='hold', reason=reason,
            included=[], establishes_misconduct=False))
        return AggregationResult({k:torch.zeros_like(v) for k,v in result.state.items()}, (),
            tuple(sorted({u.client_id for u in self.updates})), dict(result.diagnostics, aggregate_held=True)), True

    def aggregate(self, result):
        if self.arm not in ('calibrate', 'aggregate', 'repair'):
            self.archive(result)
            return result, False
        screen = self.evaluate(result, 'screen')
        # Verify the full aggregate before any repair search. Verification feedback
        # never selects an omission, and at most one chosen omission is verified.
        verification = self.evaluate(result, 'verify')
        failed = not (screen['passed'] and verification['passed'])
        self.archive(result, failed=failed and self.arm == 'repair')
        if self.arm == 'calibrate':
            return result, False
        if not failed:
            self.runner.logger.log('repair_decision', self.rnd, dict(action='accept', included=list(result.accepted_client_ids)))
            return result, False
        if self.arm != 'repair' or screen['passed']:
            return self.block(result, 'aggregate_failed' if self.arm != 'repair' else 'verification_failed')
        candidates = []
        results = {}
        for u in sorted(self.updates, key=lambda value: value.client_id):
            candidate = weighted_result(self.updates, u.client_id)
            results[u.client_id] = candidate
            candidates.append((u.client_id, self.evaluate(candidate, 'screen', u.client_id)))
        selected = choose_repair(screen, candidates)
        if selected is None:
            return self.block(result, 'no_improving_single_omission')
        client, _ = selected
        repaired = results[client]
        verified = self.evaluate(repaired, 'verify', client)
        if not verified['passed']:
            return self.block(result, 'selected_omission_failed_verification')
        self.runner.logger.log('repair_decision', self.rnd, dict(action='repair', omitted=client,
            included=list(repaired.accepted_client_ids), establishes_misconduct=False))
        return repaired, False
