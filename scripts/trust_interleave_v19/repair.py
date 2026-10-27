"""Reorder the same V18 minibatch/proposal plan; no evaluation or attack inputs."""
import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F

from scripts.trust_supervised_v18.repair import RepairPolicy, patch_batch, cpu_sgd_step
from scripts.trust_repair_v16_fixed.integrity import snapshot, state_hash, assert_hash
from scripts.trust_repair_v16_fixed.repair import transformations
from scripts.trust_repair_v16_fixed.research.splits import stream

OBJECTIVES = ('clean_tune', 'hard_patch')
SCHEDULES = ('sequential', 'interleaved')


def make_plan(local_data, seed, checkpoint, policy=RepairPolicy()):
    """Generate V18's exact per-client stream before either scheduling treatment."""
    if not local_data or any(len(x) < 2 or len(x) != len(y) for x,y in local_data.values()):
        raise ValueError('insufficient local repair data')
    if min(policy.passes,policy.steps_per_client,policy.batch_size,policy.proposals) < 1 or policy.learning_rate <= 0:
        raise ValueError('invalid repair budget')
    specs=transformations(*next(iter(local_data.values()))[0].shape[-2:])
    result=[]
    for epoch in range(policy.passes):
        for client in sorted(local_data):
            rng=stream(seed,'supervised-v18-repair',checkpoint,epoch,client)
            rng_seed=int(rng.integers(2**31))
            for step in range(policy.steps_per_client):
                ids=rng.choice(len(local_data[client][1]),policy.batch_size,
                               replace=len(local_data[client][1]) < policy.batch_size)
                choices=rng.integers(len(specs),size=(policy.proposals,len(ids)))
                result.append(dict(epoch=epoch,client=client,step=step,rng_seed=rng_seed,
                                   ids=ids.tolist(),choices=choices.tolist()))
    return result


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def scheduled(plan,schedule):
    if schedule not in SCHEDULES:
        raise ValueError('unknown repair schedule')
    keys=[(s['epoch'],s['client'],s['step']) for s in plan]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate planned update')
    order=(lambda s:(s['epoch'],s['client'],s['step'])) if schedule=='sequential' else (
        lambda s:(s['epoch'],s['step'],s['client']))
    return sorted(plan,key=order)


def observe(model,local_data):
    """Passive measurements on repair-side examples only; never select a model."""
    expected=state_hash(snapshot(model)); device=next(model.parameters()).device
    model.eval(); values={}; count=0
    with torch.no_grad():
        for client,(x,y) in sorted(local_data.items()):
            bx=torch.as_tensor(x,dtype=torch.float32,device=device)
            by=torch.as_tensor(y,dtype=torch.long,device=device)
            logits=model(bx)
            loss=F.cross_entropy(logits,by)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite diagnostic clean loss')
            values[str(client)]=dict(n=len(y),accuracy=float((logits.argmax(1)==by).float().mean().cpu()),
                                     loss=float(loss.cpu()))
            count+=len(y)
    assert_hash(model,expected,'passive repair diagnostics')
    return values,count


def repair_model(parent,local_data,objective,schedule,seed,checkpoint,policy=RepairPolicy(),plan=None):
    if objective not in OBJECTIVES or schedule not in SCHEDULES:
        raise ValueError('unknown factorial arm')
    # Current V18 CNN has neither dropout nor running-state layers. Reject
    # architectures for which reorderings would introduce additional RNG/state treatments.
    stochastic=(torch.nn.modules.dropout._DropoutNd,torch.nn.modules.batchnorm._BatchNorm)
    if any(isinstance(module,stochastic) for module in parent.modules()):
        raise ValueError('schedule diagnostic requires stateless deterministic layers')
    expected_plan=make_plan(local_data,seed,checkpoint,policy)
    if plan is None:
        plan=expected_plan
    elif plan != expected_plan:
        raise ValueError('plan differs from frozen V18 minibatch/proposal streams')
    original=state_hash(snapshot(parent)); model=copy.deepcopy(parent)
    device=next(model.parameters()).device
    specs=transformations(*next(iter(local_data.values()))[0].shape[-2:])
    model_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values())
    work=dict(optimizer_steps=0,training_examples=0,proposal_examples=0,
              model_transmissions=0,transmitted_bytes=0,diagnostic_examples=0)
    telemetry=[]; observations=[]; previous_owner=None
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        values,n=observe(model,local_data)
        observations.append(dict(after_update=0,clients=values));work['diagnostic_examples']+=n
        for update,spec in enumerate(scheduled(plan,schedule),1):
            epoch,client,step=spec['epoch'],spec['client'],spec['step']
            if step==0:
                torch.manual_seed(spec['rng_seed'])
            owner=(epoch,client)
            if owner != previous_owner:
                work['model_transmissions']+=2;work['transmitted_bytes']+=2*model_bytes
                previous_owner=owner
            x,y=local_data[client]; ids=np.asarray(spec['ids'],dtype=np.int64)
            choices=np.asarray(spec['choices'],dtype=np.int64)
            bx=torch.as_tensor(np.asarray(x,dtype=np.float32)[ids],device=device)
            by=torch.as_tensor(np.asarray(y,dtype=np.int64)[ids],device=device)
            model.eval()
            proposals=[bx.clone() if objective=='clean_tune' else patch_batch(bx,specs,row) for row in choices]
            before=state_hash(snapshot(model))
            with torch.no_grad():
                losses=torch.stack([F.cross_entropy(model(p),by,reduction='none') for p in proposals])
                selected=losses.argmax(0) if objective=='hard_patch' else torch.zeros(len(by),dtype=torch.long,device=device)
                views=torch.stack(proposals)[selected,torch.arange(len(by),device=device)]
            assert_hash(model,before,'proposal inference')
            model.train();model.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(torch.cat((bx,views))),torch.cat((by,by)))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite repair loss')
            loss.backward()
            state=cpu_sgd_step(model,policy.learning_rate)
            work['optimizer_steps']+=1;work['training_examples']+=2*len(ids)
            work['proposal_examples']+=policy.proposals*len(ids)
            telemetry.append(dict(epoch=epoch,client=client,step=step,loss=float(loss.detach().cpu()),
                selected_transformation_ids=[int(choices[int(k),i]) for i,k in enumerate(selected.cpu().tolist())]
                    if objective!='clean_tune' else [],model_hash=state_hash(state)))
            # Identical observation times/work in both schedules. Every V18
            # sequential client block ends at one of these update counts.
            if update % policy.steps_per_client==0:
                values,n=observe(model,local_data)
                observations.append(dict(after_update=update,clients=values));work['diagnostic_examples']+=n
    model.eval();assert_hash(parent,original,'repair parent preservation')
    return model,dict(work=work,steps=telemetry,observations=observations,
        model_hash=state_hash(snapshot(model)),policy=asdict(policy),plan_sha256=plan_hash(plan),
        schedule=schedule,objective=objective)
