"""Correct-label repair. This module has no attack configuration or test inputs.

Private clients execute sequential model fine-tuning in this simulator. Raw
examples stay in their local callback; model/gradient privacy is not guaranteed.
"""
import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from scripts.trust_repair_v16_fixed.integrity import snapshot, state_hash, assert_hash, assert_state
from scripts.trust_repair_v16_fixed.loss_voting import paired_decision
from scripts.trust_repair_v16_fixed.repair import transformations
from scripts.trust_repair_v16_fixed.research.splits import stream, fingerprint

ARMS = ('unchanged', 'clean_tune', 'random_patch', 'hard_patch')
STAGES = ('screen', 'verify')


@dataclass(frozen=True)
class RepairPolicy:
    repair_clients: int = 5
    panel_size: int = 5
    examples_per_client: int = 32
    passes: int = 2
    steps_per_client: int = 20
    batch_size: int = 16
    proposals: int = 4
    learning_rate: float = .01
    accuracy_tolerance: float = .02
    loss_tolerance: float = .02
    standard_errors: float = 2.


def allocate(validation, contributors, seed, checkpoint, policy):
    """Select disjoint clients without consulting roles, labels or attack status."""
    pool = sorted(set(range(len(validation)))-set(contributors))
    count = policy.repair_clients + 2*policy.panel_size
    if len(pool) < count:
        raise ValueError('insufficient noncontributor repair/verification clients')
    chosen = stream(seed, 'supervised-v18-clients', checkpoint).choice(pool, count, replace=False).tolist()
    clients = dict(repair=chosen[:policy.repair_clients],
                   screen=chosen[policy.repair_clients:policy.repair_clients+policy.panel_size],
                   verify=chosen[policy.repair_clients+policy.panel_size:])
    groups = {}
    for stage, members in clients.items():
        groups[stage] = {}
        for client in members:
            ids = np.asarray(validation[client], dtype=np.int64)
            ids = ids[stream(seed, 'supervised-v18-examples', checkpoint, client).permutation(len(ids))]
            ids = ids[:policy.examples_per_client]
            if len(ids) < 2:
                raise ValueError('at least two examples required per selected client')
            groups[stage][client] = ids
    assert_disjoint(groups)
    return groups


def assert_disjoint(groups):
    if set(groups) != {'repair', *STAGES}:
        raise ValueError('repair and two verification panels required')
    seen_clients, seen_examples = set(), set()
    for members in groups.values():
        for client, ids in members.items():
            if client in seen_clients or seen_examples.intersection(map(int, ids)) or len(set(map(int, ids))) != len(ids):
                raise ValueError('repair/verification clients or examples overlap')
            seen_clients.add(client)
            seen_examples.update(map(int, ids))


def partition_binding(groups):
    assert_disjoint(groups)
    return {stage: {str(c): dict(count=len(ids), index_sha256=fingerprint(ids))
                    for c, ids in members.items()} for stage, members in groups.items()}


def patch_batch(images, specs, choices):
    values = images.clone()
    if len(values) != len(choices):
        raise ValueError('one transformation per example required')
    for i, choice in enumerate(choices):
        size, value, row, col = specs[int(choice)]
        values[i, :, row:row+size, col:col+size] = value
    return values


def checked_load(model, state, context):
    model.load_state_dict({key:value.to(next(model.parameters()).device) for key, value in state.items()})
    assert_state(model, state, context)


def cpu_sgd_step(model, learning_rate):
    """Apply the plain-SGD update with CPU arithmetic, then verify GPU loading."""
    expected = snapshot(model)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().cpu().clone()
            if not torch.isfinite(gradient).all():
                raise ValueError('nonfinite repair gradient')
            expected[name] = expected[name] - learning_rate*gradient
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in expected.values()):
        raise ValueError('nonfinite repaired model')
    checked_load(model, expected, 'CPU repair optimizer commit')
    return expected


def repair_model(parent, local_data, arm, seed, checkpoint, policy=RepairPolicy()):
    """local_data maps repair clients to (features, correct labels), never verifier/test data."""
    if arm not in ARMS:
        raise ValueError('unknown repair arm')
    if not local_data or any(len(x) < 2 or len(x) != len(y) for x, y in local_data.values()):
        raise ValueError('insufficient local repair data')
    if not (policy.passes > 0 and policy.steps_per_client > 0 and policy.proposals > 0 and policy.learning_rate > 0):
        raise ValueError('invalid repair budget')
    original = state_hash(snapshot(parent))
    model = copy.deepcopy(parent)
    device = next(model.parameters()).device
    work = dict(optimizer_steps=0, training_examples=0, proposal_examples=0,
                model_transmissions=0, transmitted_bytes=0)
    model_bytes = sum(v.numel()*v.element_size() for v in model.state_dict().values())
    telemetry = []
    if arm == 'unchanged':
        assert_hash(parent, original, 'unchanged repair parent')
        return model, dict(work=work, steps=telemetry, model_hash=original, policy=asdict(policy))
    specs = transformations(*next(iter(local_data.values()))[0].shape[-2:])
    # Domain streams are identical across repair arms, and restored after repair.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for epoch in range(policy.passes):
            for client in sorted(local_data):
                x, y = local_data[client]
                x = np.asarray(x, dtype=np.float32)
                y = np.asarray(y, dtype=np.int64)
                rng = stream(seed, 'supervised-v18-repair', checkpoint, epoch, client)
                torch.manual_seed(int(rng.integers(2**31)))
                work['model_transmissions'] += 2
                work['transmitted_bytes'] += 2*model_bytes
                for step in range(policy.steps_per_client):
                    ids = rng.choice(len(y), policy.batch_size, replace=len(y) < policy.batch_size)
                    choices = rng.integers(len(specs), size=(policy.proposals, len(ids)))
                    bx = torch.as_tensor(x[ids], device=device)
                    by = torch.as_tensor(y[ids], device=device)
                    model.eval()
                    # All tuned arms pay the same proposal inference budget.
                    # clean_tune evaluates duplicate clean views; random_patch
                    # selects proposal zero; hard_patch selects per-example CE.
                    proposals = [bx.clone() if arm == 'clean_tune' else patch_batch(bx, specs, row) for row in choices]
                    before = state_hash(snapshot(model))
                    with torch.no_grad():
                        losses = torch.stack([F.cross_entropy(model(p), by, reduction='none') for p in proposals])
                        selected = losses.argmax(0) if arm == 'hard_patch' else torch.zeros(len(by), dtype=torch.long, device=device)
                        views = torch.stack(proposals)[selected, torch.arange(len(by), device=device)]
                    assert_hash(model, before, 'repair proposal inference')
                    model.train()
                    model.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(torch.cat((bx, views))), torch.cat((by, by)))
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite supervised repair loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('nonfinite supervised repair gradient')
                    state = cpu_sgd_step(model, policy.learning_rate)
                    work['optimizer_steps'] += 1
                    work['training_examples'] += 2*len(ids)
                    work['proposal_examples'] += policy.proposals*len(ids)
                    telemetry.append(dict(epoch=epoch, client=client, step=step, loss=float(loss.detach().cpu()),
                        selected_transformation_ids=[int(choices[int(k), i]) for i,k in enumerate(selected.cpu().tolist())]
                        if arm != 'clean_tune' else [], model_hash=state_hash(state)))
    model.eval()
    assert_hash(parent, original, 'repair must preserve parent')
    return model, dict(work=work, steps=telemetry, model_hash=state_hash(snapshot(model)), policy=asdict(policy))


def measure(model, x, y, batch=128):
    """Supervised clean/patch measurements; correct labels, no attacker target."""
    if len(y) < 2 or len(x) != len(y):
        raise ValueError('insufficient verification examples')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite verification examples')
    device = next(model.parameters()).device
    expected = state_hash(snapshot(model))
    model.eval()
    specs = transformations(*x.shape[-2:])
    clean_losses, predictions, patch_losses, patch_correct = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(y), batch):
            bx = torch.as_tensor(x[start:start+batch], dtype=torch.float32, device=device)
            by = torch.as_tensor(y[start:start+batch], dtype=torch.long, device=device)
            logits = model(bx)
            clean_losses.extend(F.cross_entropy(logits, by, reduction='none').cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            losses, correct = [], []
            for index in range(len(specs)):
                logits = model(patch_batch(bx, specs, [index]*len(bx)))
                losses.append(F.cross_entropy(logits, by, reduction='none'))
                correct.append(logits.argmax(1) == by)
            patch_losses.extend(torch.stack(losses).max(0).values.cpu().tolist())
            patch_correct.extend(torch.stack(correct).all(0).cpu().tolist())
    assert_hash(model, expected, 'supervised verification inference')
    if not np.isfinite(clean_losses).all() or not np.isfinite(patch_losses).all():
        raise ValueError('nonfinite verification losses')
    predictions = np.asarray(predictions)
    counts = np.bincount(predictions, minlength=int(max(max(y), max(predictions)))+1)
    return dict(model_hash=expected, example_binding=fingerprint(x,y), n=len(y), clean_losses=clean_losses,
                worst_patch_losses=patch_losses, correct_count=int((predictions == y).sum()),
                worst_patch_correct_count=int(sum(patch_correct)), prediction_counts=counts.tolist(),
                label_class_count=len(np.unique(y)), label_counts=np.bincount(y).tolist(),
                constant_prediction=bool(np.count_nonzero(counts) == 1 and len(np.unique(y)) > 1),
                example_evaluations=len(y)*(1+len(specs)))


def verify(parent_measurements, candidate_measurements, policy=RepairPolicy()):
    """Two panels must pass. Constant predictions cannot pass on diverse labels."""
    if set(parent_measurements) != set(STAGES) or set(candidate_measurements) != set(STAGES):
        raise ValueError('both disjoint verification panels required')
    panels = {}
    hashes = set()
    for stage in STAGES:
        before, after = parent_measurements[stage], candidate_measurements[stage]
        if len(before) != policy.panel_size or set(before) != set(after):
            raise ValueError('incomplete verification panel')
        clean_before, clean_after, patch_before, patch_after = [], [], [], []
        correct_before = correct_after = 0
        prediction_counts = np.zeros(0, dtype=np.int64)
        label_counts = np.zeros(0, dtype=np.int64)
        evaluations = 0
        for client in before:
            b, a = before[client], after[client]
            if b['n'] != a['n'] or b['n'] < 2 or b['example_binding'] != a['example_binding']:
                raise ValueError('unpaired verification examples')
            hashes.add(a['model_hash'])
            clean_before.extend(b['clean_losses']); clean_after.extend(a['clean_losses'])
            patch_before.extend(b['worst_patch_losses']); patch_after.extend(a['worst_patch_losses'])
            correct_before += b['correct_count']; correct_after += a['correct_count']
            size = max(len(prediction_counts), len(a['prediction_counts']))
            prediction_counts = np.pad(prediction_counts, (0,size-len(prediction_counts)))
            prediction_counts[:len(a['prediction_counts'])] += a['prediction_counts']
            size = max(len(label_counts), len(a['label_counts']))
            label_counts = np.pad(label_counts, (0,size-len(label_counts)))
            label_counts[:len(a['label_counts'])] += a['label_counts']
            evaluations += a['example_evaluations']
        clean_passed, paired = paired_decision(clean_before, clean_after, policy.loss_tolerance, policy.standard_errors)
        n = len(clean_after)
        accuracy_delta = (correct_after-correct_before)/n
        patch_delta = float(np.mean(patch_after)-np.mean(patch_before))
        # Reject constant prediction even if clean and patched outputs agree.
        collapsed = bool(np.count_nonzero(prediction_counts) == 1 and np.count_nonzero(label_counts) > 1)
        passed = bool(clean_passed and accuracy_delta >= -policy.accuracy_tolerance-1e-12
                      and patch_delta <= 1e-12 and not collapsed)
        panels[stage] = dict(passed=passed, clean_accuracy_delta=accuracy_delta,
                             clean_loss_check=paired, worst_patch_loss_delta=patch_delta,
                             constant_prediction=collapsed, example_evaluations=evaluations)
    if len(hashes) != 1:
        raise ValueError('verification panels scored different models')
    return dict(passed=all(p['passed'] for p in panels.values()), panels=panels,
                scored_model_hash=hashes.pop(), empirical_rule=True, formal_security_guarantee=False)
