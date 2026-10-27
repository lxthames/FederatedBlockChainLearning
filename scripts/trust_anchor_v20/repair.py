"""Repair uses only original repair examples and a frozen parent teacher.

No test metrics, attack identities, target labels, or verifier data enter here.
"""
import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from scripts.trust_interleave_v19 import repair as base
from scripts.trust_supervised_v18.repair import RepairPolicy, patch_batch, cpu_sgd_step
from scripts.trust_repair_v16_fixed.integrity import snapshot, state_hash, assert_hash

ARMS = ('hard_patch', 'anchored_clean', 'anchored_patch', 'anchored_suppress')


@dataclass(frozen=True)
class AnchorPolicy:
    temperature: float = 2.0
    alpha: float = .5
    clean_weight: float = 2.0
    invariance_weight: float = 2.0
    # Proposal did not specify gamma: freeze one value before observing outcomes.
    suppression_weight: float = .1


def loss_terms(clean, patched, teacher, labels, arm, policy=AnchorPolicy()):
    if arm not in ARMS[1:]:
        raise ValueError('anchored objective required')
    teacher = teacher.detach()
    t = policy.temperature
    if t <= 0 or min(policy.alpha, policy.clean_weight, policy.invariance_weight,
                     policy.suppression_weight) < 0:
        raise ValueError('invalid anchoring weights')
    target = F.softmax(teacher / t, dim=1)
    kl_clean = t*t*F.kl_div(F.log_softmax(clean/t, dim=1), target, reduction='batchmean')
    kl_patch = t*t*F.kl_div(F.log_softmax(patched/t, dim=1), target, reduction='batchmean')
    ce = F.cross_entropy(clean if arm == 'anchored_clean' else patched, labels)
    # Literal proposed raw-logit penalty: sum over non-label classes, mean over
    # examples. Not gauge invariant, not guaranteed safe on clean models.
    mask = ~F.one_hot(labels, num_classes=patched.shape[1]).bool()
    suppression = (F.relu(patched-teacher).square()*mask).sum(1).mean()
    loss = policy.alpha*ce + policy.clean_weight*kl_clean
    if arm != 'anchored_clean':
        loss = loss + policy.invariance_weight*kl_patch
    if arm == 'anchored_suppress':
        loss = loss + policy.suppression_weight*suppression
    return loss, dict(ce=ce, clean_kl=kl_clean, patch_kl=kl_patch, suppression=suppression)


def freeze_convolutions(model):
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for name, value in module.named_parameters(recurse=False):
                value.requires_grad_(False)
                names.append(f'{module_name}.{name}' if module_name else name)
    if not names or not any(p.requires_grad for p in model.parameters()):
        raise ValueError('requires convolutional backbone and trainable head')
    return names


def repair_model(parent, local_data, arm, seed, checkpoint, policy=RepairPolicy(),
                 plan=None, anchor=AnchorPolicy()):
    if arm not in ARMS:
        raise ValueError('unknown V20 arm')
    expected = base.make_plan(local_data, seed, checkpoint, policy)
    if plan is None:
        plan = expected
    if plan != expected:
        raise ValueError('plan differs from frozen V19 streams')
    if arm == 'hard_patch':
        return base.repair_model(parent, local_data, arm, 'interleaved', seed, checkpoint, policy, plan)
    if any(isinstance(m, (torch.nn.modules.dropout._DropoutNd,
                         torch.nn.modules.batchnorm._BatchNorm)) for m in parent.modules()):
        raise ValueError('requires deterministic stateless architecture')
    original = state_hash(snapshot(parent))
    teacher = copy.deepcopy(parent).eval()
    teacher.requires_grad_(False)
    model = copy.deepcopy(parent)
    frozen_names = freeze_convolutions(model)
    frozen = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if n in frozen_names}
    device = next(model.parameters()).device
    specs = base.transformations(*next(iter(local_data.values()))[0].shape[-2:])
    model_bytes = sum(v.numel()*v.element_size() for v in model.state_dict().values())
    # Teacher sent once to each repair client and cached; full student exchanged
    # on each visit. These are modeled transfers, not measured networking.
    work = dict(optimizer_steps=0, training_examples=0, proposal_examples=0,
                diagnostic_examples=0, model_transmissions=0, transmitted_bytes=0,
                teacher_examples=0, teacher_transmissions=len(local_data),
                teacher_transmitted_bytes=len(local_data)*model_bytes)
    steps=[]; observations=[]; previous_owner=None
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        values,n=base.observe(model,local_data)
        observations.append(dict(after_update=0,clients=values));work['diagnostic_examples']+=n
        for update,spec in enumerate(base.scheduled(plan,'interleaved'),1):
            if spec['step']==0:
                torch.manual_seed(spec['rng_seed'])
            owner=(spec['epoch'],spec['client'])
            if owner != previous_owner:
                work['model_transmissions']+=2;work['transmitted_bytes']+=2*model_bytes
                previous_owner=owner
            x,y=local_data[spec['client']];ids=np.asarray(spec['ids'],dtype=np.int64)
            bx=torch.as_tensor(np.asarray(x,dtype=np.float32)[ids],device=device)
            by=torch.as_tensor(np.asarray(y,dtype=np.int64)[ids],device=device)
            choices=np.asarray(spec['choices'],dtype=np.int64)
            proposals=[patch_batch(bx,specs,row) for row in choices]
            model.eval();before=state_hash(snapshot(model))
            with torch.no_grad():
                losses=torch.stack([F.cross_entropy(model(p),by,reduction='none') for p in proposals])
                selected=losses.argmax(0)
                views=torch.stack(proposals)[selected,torch.arange(len(by),device=device)]
                teacher_logits=teacher(bx)
            assert_hash(model,before,'proposal inference')
            model.train();model.zero_grad(set_to_none=True)
            clean_logits,patch_logits=model(torch.cat((bx,views))).chunk(2)
            loss,components=loss_terms(clean_logits,patch_logits,teacher_logits,by,arm,anchor)
            if not torch.isfinite(loss) or any(not torch.isfinite(v) for v in components.values()):
                raise ValueError('nonfinite anchored objective')
            loss.backward();state=cpu_sgd_step(model,policy.learning_rate)
            if any(not torch.equal(state[n],v) for n,v in frozen.items()):
                raise ValueError('frozen convolution changed')
            work['optimizer_steps']+=1;work['training_examples']+=2*len(ids)
            work['proposal_examples']+=policy.proposals*len(ids);work['teacher_examples']+=len(ids)
            steps.append(dict(epoch=spec['epoch'],client=spec['client'],step=spec['step'],
                loss=float(loss.detach().cpu()),components={k:float(v.detach().cpu()) for k,v in components.items()},
                selected_transformation_ids=[int(choices[int(k),i]) for i,k in enumerate(selected.cpu().tolist())],
                model_hash=state_hash(state)))
            if update % policy.steps_per_client==0:
                values,n=base.observe(model,local_data)
                observations.append(dict(after_update=update,clients=values));work['diagnostic_examples']+=n
    model.eval();assert_hash(parent,original,'parent preservation');assert_hash(teacher,original,'teacher preservation')
    return model,dict(work=work,steps=steps,observations=observations,model_hash=state_hash(snapshot(model)),
        teacher_hash=original,frozen_parameter_names=frozen_names,policy=asdict(policy),anchor_policy=asdict(anchor),
        plan_sha256=base.plan_hash(plan),schedule='interleaved',objective=arm)
