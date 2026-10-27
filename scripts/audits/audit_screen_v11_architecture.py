"""CPU-only architecture/optimizer diagnostics; frozen V11 is never modified.

Uses seed-11 training data only. Runs one local epoch and a paired four-step
prefix for all 50 FEMNIST clients, then a fixed 200-step training memorization
check on one enrolled training example per observed class. This is a numerical
sanity check, not a replacement architecture benchmark or a generalization test.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.trust_screen_v11 import study
from scripts.trust_screen_v11.research.runner_v2 import BenchmarkRunner
from scripts.trust_screen_v11.research.splits import stream


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/screen-v11-architecture-audit'


def read(path):
    return json.loads(path.read_text())


def delta_metrics(model,initial):
    groups={}
    for name,p in model.named_parameters():
        group='conv' if name.split('.')[0] in ('0','3') else ('dense_hidden' if name.startswith('7.') else 'classifier')
        value=float((p.detach()-initial[name]).square().sum())
        groups[group]=groups.get(group,0.)+value
    return dict(norm=math.sqrt(sum(groups.values())),layer_squared_norm=groups,
                classifier_bias_norm=float((model[9].bias.detach()-initial['9.bias']).norm()))


def capped_train(model,x,y,c,client,steps):
    # Same initialization, shuffle and optimizer as BenchmarkRunner._train;
    # only the maximum step count changes. Prefix fidelity is checked below.
    seed=int(stream(c.seed,'training',1,client+100).integers(2**31))
    torch.manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(x.copy()),torch.from_numpy(y.copy())),
                      batch_size=c.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.SGD(model.parameters(),lr=c.learning_rate,momentum=c.optimizer_momentum)
    model.train()
    for step,(bx,by) in enumerate(loader):
        if step==steps:break
        optimizer.zero_grad()
        loss=nn.functional.cross_entropy(model(bx),by.long())
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1])


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert study.inputs()==read(study.JOB/'frozen-inputs.json')
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for dataset,outputs in [('fashion_mnist',10),('femnist',62)]:
        data=study.DATA/f'{dataset}-11'
        x=np.load(data/'features.npy',mmap_mode='r');y=np.load(data/'labels.npy')
        c=study.config(dataset,'fedavg',11)
        split=read(study.OUT/f'v11-{dataset}-fedavg-none-s11'/'splits.json')
        torch.manual_seed(11)
        model=study.create_model(dataset,x.shape[1:],outputs)
        bx=torch.tensor(np.array(x[split['train'][0][:16]]))
        model.train();train_logits=model(bx).detach()
        model.eval();eval_logits=model(bx).detach()
        assert torch.equal(train_logits,eval_logits)
        assert train_logits.shape==(len(bx),outputs) and torch.isfinite(train_logits).all()
        sizes=np.array([len(ids) for ids in split['train']])
        rows.append(dict(dataset=dataset,shape=list(x.shape),dtype=str(x.dtype),
            feature_range=[float(x.min()),float(x.max())],label_range=[int(y.min()),int(y.max())],
            total_parameters=sum(p.numel() for p in model.parameters()),
            parameters={n:p.numel() for n,p in model.named_parameters()},
            buffers=list(dict(model.named_buffers())),train_eval_identical=True,
            optimizer_steps_min_max=[int(np.ceil(sizes/c.batch_size).min()),int(np.ceil(sizes/c.batch_size).max())],
            initial_logits_std=float(train_logits.std())))
    c=study.config('femnist','fedavg',11)
    from dataclasses import replace
    c=replace(c,device='cpu')
    data=study.DATA/'femnist-11'
    x=np.load(data/'features.npy');y=np.load(data/'labels.npy')
    split=read(study.OUT/'v11-femnist-fedavg-none-s11'/'splits.json')
    torch.manual_seed(11)
    base=study.create_model('femnist',x.shape[1:],62)
    initial={n:p.detach().clone() for n,p in base.named_parameters()}
    # _train does not write events. Keep its logger path separate from frozen runs.
    runner=BenchmarkRunner(c,OUT/'training-helper')
    clients=[]
    fidelity=[]
    for i,ids in enumerate(split['train']):
        full=copy.deepcopy(base)
        runner._train(full,x[ids],y[ids],1,i)
        fixed=copy.deepcopy(base)
        capped_train(fixed,x[ids],y[ids],c,i,4)
        if math.ceil(len(ids)/16)==4:
            fidelity.append(all(torch.equal(full.state_dict()[n],fixed.state_dict()[n]) for n in full.state_dict()))
        clients.append(dict(client=i,train_count=len(ids),local_steps=math.ceil(len(ids)/16),
                            full_epoch=delta_metrics(full,initial),four_steps=delta_metrics(fixed,initial)))
    assert fidelity and all(fidelity)
    print('Completed 50 paired local-update probes on CPU',flush=True)
    # Fixed budget and sample rule; no stopping on an observed success threshold.
    pool=np.concatenate(split['train'])
    ids=np.array([pool[np.flatnonzero(y[pool]==label)[0]] for label in np.unique(y[pool])])
    assert len(ids)==62
    bx=torch.from_numpy(x[ids]);by=torch.from_numpy(y[ids]).long()
    fitted=copy.deepcopy(base)
    optimizer=torch.optim.SGD(fitted.parameters(),lr=.05,momentum=.9)
    fit=[]
    for step in range(201):
        fitted.train();logits=fitted(bx)
        loss=nn.functional.cross_entropy(logits,by)
        if step in (0,10,25,50,100,200):
            fit.append(dict(step=step,loss=float(loss.detach()),accuracy=float((logits.argmax(1)==by).float().mean())))
        if step==200:break
        optimizer.zero_grad();loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in fitted.parameters())
        optimizer.step()
    aggregates={}
    for kind in ('full_epoch','four_steps'):
        norms=np.array([r[kind]['norm'] for r in clients])
        aggregates[kind]=dict(mean_update_norm=float(norms.mean()),
            size_norm_correlation=correlation([r['train_count'] for r in clients],norms),
            mean_layer_energy_share={group:float(np.mean([r[kind]['layer_squared_norm'][group]/r[kind]['norm']**2 for r in clients]))
                                    for group in ('conv','dense_hidden','classifier')})
    summaries=[]
    for seed in study.SEEDS:
        for arm in study.CLEAN_ARMS:
            path=study.OUT/f'v11-femnist-{arm}-none-s{seed}'
            if (path/'completion.json').exists():
                s=study.load_completed(path,study.config('femnist',arm,seed))
                summaries.append(dict(seed=seed,arm=arm,final_accuracy=s['final_accuracy'],
                                      initial_model_hash=s['initial_model_hash']))
    assert all(len({r['initial_model_hash'] for r in summaries if r['seed']==s})==1 for s in study.SEEDS)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),device='cpu',
        scope='Initial-model training-only numerical probes. Not matched GPU trajectories or generalization evidence.',
        frozen_inputs_verified=True,architectures=rows,clients=clients,local_update_summary=aggregates,
        capped_training_prefix_parity=all(fidelity),memorization=fit,clean_femnist_summaries=summaries,
        source_sha256=study.digest(__file__))
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(local_update_summary=aggregates,memorization=fit),indent=2),flush=True)
    print('Saved',OUT/'audit.json')


if __name__=='__main__':main()
