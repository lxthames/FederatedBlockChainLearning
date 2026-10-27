"""Attack-agnostic measurements on the exact aggregate admitted by clean quorum.

Neither scoring nor decisions access attacker roles, target labels or test arrays.
Scalar thresholds are empirical development calibration, not security guarantees.
"""
import copy
import math
from pathlib import Path

import numpy as np
import torch

from scripts.trust_repair_v16_fixed.integrity import assert_hash, snapshot, state_hash
from scripts.trust_repair_v16_fixed.loss_voting import paired_decision
from scripts.trust_repair_v16_fixed.repair import challenge_bank, transformations, CAP, PANEL
from scripts.trust_repair_v16_fixed.research.contracts import AggregationResult, ClientUpdate
from scripts.trust_repair_v16_fixed.research.runner_helpers import _apply_delta, _update_bytes
from scripts.trust_repair_v16_fixed.research.splits import stream

VARIANTS = ('probability_independent', 'probability_aligned', 'label_independent', 'label_aligned')
METRICS = ('absolute', 'incremental')
STAGES = ('screen', 'verify')
QUANTILE = .995


def contrast(probabilities, labels=False):
    values = np.asarray(probabilities)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError('expected transformation/example/class probabilities with two examples')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite probabilities')
    if labels:
        values = np.eye(values.shape[-1], dtype=np.float64)[values.argmax(axis=-1)]
    return (values[1:] - values[0]).mean(axis=1)


def panel_score(responses, aligned):
    """Two negative responses out of five, preserving transformation/class identity."""
    values = np.asarray(responses, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != PANEL or not np.isfinite(values).all():
        raise ValueError('five finite validator response matrices required')
    # Second-largest corresponds to failure of a four-of-five approval quorum.
    if aligned:
        result = np.partition(values, PANEL-2, axis=0)[PANEL-2].max()
    else:
        result = np.partition(values.max(axis=(1, 2)), PANEL-2)[PANEL-2]
    return max(0., float(result))


def scores(before, after):
    """Inputs map representation to paired [validator,transform,class] matrices."""
    result = {}
    for representation in ('probability', 'label'):
        for aligned in (False, True):
            name = representation + ('_aligned' if aligned else '_independent')
            result[name] = dict(
                absolute=panel_score(after[representation], aligned),
                incremental=panel_score(after[representation]-before[representation], aligned))
    return result


def decide(panel, thresholds, variant):
    """Missing responses hold a round, and never establish proposer misconduct."""
    if panel['responses'] != PANEL or not panel['complete']:
        return False
    limits = thresholds[variant]
    for key in METRICS + ('clean_excess',):
        if not math.isfinite(limits[key]) or limits[key] < 0:
            raise ValueError('invalid calibration threshold')
    return (all(panel['scores'][variant][key] <= limits[key] for key in METRICS)
            and sum(value <= limits['clean_excess'] for value in panel['clean_excess']) >= 4)


def original_decide(panel, thresholds):
    if panel['responses'] != PANEL or not panel['complete']:
        return False
    return sum(all(metrics[key] <= thresholds[key]
                   for key in METRICS + ('clean_excess',))
               for metrics in panel['original_metrics']) >= 4


class AlignmentRound:
    def __init__(self, runner, parent, x, y, split, rnd, updates, eligible):
        self.runner, self.parent, self.x, self.y = runner, parent, x, y
        self.split, self.rnd, self.updates = split, rnd, updates
        self.arm = runner.challenge_arm
        if self.arm not in ('passive', 'current', 'clip_only', 'original', 'candidate'):
            raise ValueError('unknown alignment arm')
        self.expected_committed_hash = None
        self.extra_transmitted_bytes = 0
        self.response_bytes = 0
        self.thresholds = runner.challenge_thresholds
        if self.arm == 'candidate' and self.thresholds is None:
            raise ValueError('frozen candidate calibration required')
        if self.arm == 'original' and runner.original_thresholds is None:
            raise ValueError('frozen original calibration required')
        contributors = {u.client_id for u in updates}
        pool = sorted(set(eligible)-contributors)
        self.panels = None
        if self.arm in ('passive', 'original', 'candidate') and len(pool) >= 2*PANEL:
            members = stream(runner.config.seed, 'repair-panels-v15', rnd).choice(pool, 2*PANEL, replace=False).tolist()
            self.panels = dict(screen=sorted(members[:PANEL]), verify=sorted(members[PANEL:]))

    def candidate(self, validator, client):
        # Existing per-update clean-loss quorum always retains control here.
        return True

    def probabilities(self, model, bank, examples):
        expected = state_hash(snapshot(model))
        work = self.runner.validation_work
        work['alignment_example_evaluations'] = work.get('alignment_example_evaluations', 0) + len(bank)
        batch = self.runner.config.evaluation_batch_size
        work['alignment_forward_passes'] = work.get('alignment_forward_passes', 0) + math.ceil(len(bank)/batch)
        model.eval()
        with torch.no_grad():
            pieces = [model(torch.as_tensor(bank[i:i+batch], device=self.runner.device)).softmax(1).cpu().numpy()
                      for i in range(0, len(bank), batch)]
        values = np.concatenate(pieces).reshape(-1, examples, self.runner.config.num_classes)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite behavioral inference')
        assert_hash(model, expected, 'alignment inference')
        return values

    def evaluate(self, model, stage):
        members = self.panels[stage] if self.panels else []
        before = {key: [] for key in ('probability', 'label')}
        after = {key: [] for key in before}
        losses, original, files = [], [], []
        for validator in members:
            ids = np.asarray(self.split.validation[validator])
            rng = stream(self.runner.config.seed, 'repair-examples-v15', self.rnd, validator)
            ids = ids[rng.permutation(len(ids))[:CAP]]
            if len(ids) < 2:
                self.runner.logger.log('alignment_abstention', self.rnd, dict(stage=stage, validator=validator, examples=len(ids)))
                continue
            bank = challenge_bank(self.x[ids], self.runner.config.seed, self.rnd, validator)
            parent = self.probabilities(self.parent, bank, len(ids))
            candidate = self.probabilities(model, bank, len(ids))
            transforms = transformations(*self.x.shape[-2:])
            spatial_count = len(transforms)
            bp, ap = contrast(parent), contrast(candidate)
            bl, al = contrast(parent, True), contrast(candidate, True)
            before['probability'].append(bp[:spatial_count])
            after['probability'].append(ap[:spatial_count])
            before['label'].append(bl[:spatial_count])
            after['label'].append(al[:spatial_count])
            ix = np.arange(len(ids)); labels = self.y[ids]; tiny = np.finfo('float32').tiny
            _, clean = paired_decision(-np.log(np.maximum(parent[0, ix, labels], tiny)),
                                      -np.log(np.maximum(candidate[0, ix, labels], tiny)), 0., 2.)
            excess = max(0., clean['mean_loss_change']-2*clean['standard_error'])
            losses.append(excess)
            original.append(dict(absolute=max(0., float(ap.max())),
                                 incremental=max(0., float((ap-bp).max())), clean_excess=excess))
            directory = Path(self.runner.output_dir)/'behavior'
            directory.mkdir(exist_ok=True)
            filename = f'r{self.rnd:03}-{stage}-v{validator}.npz'
            counts = lambda p: np.stack([np.bincount(row, minlength=p.shape[-1])
                                         for row in p.argmax(-1)]).astype('int32')
            np.savez_compressed(directory/filename, transformation_ids=np.arange(spatial_count, dtype='int32'),
                transformation_specs=np.asarray(transforms), sample_count=np.array(len(ids)),
                parent_class_counts=counts(parent), candidate_class_counts=counts(candidate),
                probability_parent_contrast=bp, probability_candidate_contrast=ap,
                label_parent_contrast=bl, label_candidate_contrast=al)
            # The aligned gate requires coordinate reports, not merely scalar votes.
            # Passive telemetry is charged separately from the unchanged quorum.
            if self.arm == 'candidate':
                self.response_bytes += bl[:spatial_count].nbytes + al[:spatial_count].nbytes + 16
            elif self.arm == 'passive':
                self.response_bytes += bp.nbytes + ap.nbytes + bl.nbytes + al.nbytes + counts(parent).nbytes + counts(candidate).nbytes + 16
            files.append(filename)
            self.runner.logger.log('alignment_measurement', self.rnd, dict(stage=stage, validator=validator,
                examples=len(ids), clean=clean, original_metrics=original[-1], artifact=filename,
                model_hash=state_hash(snapshot(model)), establishes_misconduct=False))
        complete = len(losses) == PANEL
        values = scores({k: np.stack(v) for k, v in before.items()},
                        {k: np.stack(v) for k, v in after.items()}) if complete else None
        result = dict(stage=stage, members=members, responses=len(losses), complete=complete,
            scores=values, clean_excess=losses, original_metrics=original, artifacts=files,
            model_hash=state_hash(snapshot(model)), establishes_misconduct=False)
        self.runner.logger.log('alignment_panel', self.rnd, result)
        return result

    def aggregate(self, result):
        if self.arm in ('current', 'clip_only'):
            return result, False
        candidate = copy.deepcopy(self.parent)
        _apply_delta(candidate, dict(result.state))
        evaluated_hash = state_hash(snapshot(candidate))
        panels = {stage: self.evaluate(candidate, stage) for stage in STAGES}
        if self.arm == 'passive':
            passed = True
        elif self.arm == 'candidate':
            passed = all(decide(panels[stage], self.thresholds[stage], 'label_aligned') for stage in STAGES)
        else:
            passed = all(original_decide(panels[stage], self.runner.original_thresholds[stage]) for stage in STAGES)
        assert_hash(candidate, evaluated_hash, 'alignment after panels')
        model_bytes = _update_bytes(ClientUpdate(-1, result.state, 0))
        self.extra_transmitted_bytes += sum((model_bytes+96)*len(p['members']) for p in panels.values()) + self.response_bytes
        self.runner.logger.log('alignment_decision', self.rnd, dict(arm=self.arm, passed=passed,
            passive=self.arm == 'passive', model_hash=evaluated_hash,
            included=list(result.accepted_client_ids) if passed else [], establishes_misconduct=False))
        if passed:
            self.expected_committed_hash = evaluated_hash
            return result, False
        # A pre-existing empty aggregate is already counted as unsupported.
        blocked = bool(result.accepted_client_ids)
        self.expected_committed_hash = state_hash(snapshot(self.parent))
        return AggregationResult({k: torch.zeros_like(v) for k, v in result.state.items()}, (),
            tuple(sorted(set(result.accepted_client_ids)|set(result.rejected_client_ids))),
            dict(result.diagnostics, alignment_held=True)), blocked
