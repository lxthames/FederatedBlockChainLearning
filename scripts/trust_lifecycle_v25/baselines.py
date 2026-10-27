"""Clean fine-tuning and activation-ranked Fine-Pruning adaptation.

Only repair clients provide pruning calibration and training examples. Verification
panels are disjoint. No test arrays or attack information enter this module.
Reference: https://arxiv.org/abs/1805.12185 . Adaptation: prune the final hidden
ReLU layer of the shared CNN, with a fixed 2 percentage-point repair-side limit.
"""
import copy
import json
import time
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from scripts.trust_online_v21.online import Outcome, OnlineRepair, select_committed
from scripts.trust_supervised_v18.repair import (allocate,measure,verify,STAGES,
    partition_binding,cpu_sgd_step,RepairPolicy)
from scripts.trust_interleave_v19.repair import make_plan,scheduled,plan_hash
from scripts.trust_anchor_v20.repair import repair_model,AnchorPolicy
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash,assert_hash
from scripts.trust_periodic_v22.online import DEPLOYMENT


def prune_units(model, indices):
    """Zero incoming weights/bias and outgoing weights: dead ReLUs stay dead."""
    with torch.no_grad():
        model[7].weight[indices]=0
        model[7].bias[indices]=0
        model[9].weight[:,indices]=0


def fine_tune(source,local,seed,rnd,policy,pruning=False):
    candidate=copy.deepcopy(source)
    for p in candidate.parameters(): p.requires_grad_(True)
    candidate.eval()
    device=next(candidate.parameters()).device
    ranking=[];pruned=[];calibration_evaluations=0
    if pruning:
        if not (isinstance(candidate,torch.nn.Sequential) and
                isinstance(candidate[7],torch.nn.Linear) and isinstance(candidate[8],torch.nn.ReLU)):
            raise ValueError('Fine-Pruning adaptation requires the declared CNN head')
        sums=torch.zeros(candidate[7].out_features,device=device)
        count=0;correct=0
        with torch.no_grad():
            for x,y in local.values():
                bx=torch.as_tensor(x,device=device);by=torch.as_tensor(y,device=device)
                h=candidate[:9](bx)
                sums+=h.sum(0);count+=len(by)
                correct+=int((candidate[9](h).argmax(1)==by).sum())
        calibration_evaluations+=count
        ranking=torch.argsort(sums/count,stable=True).cpu().tolist()
        # Clean repair-side calibration only; no test or verifier model selection.
        original=snapshot(candidate)
        for fraction in np.arange(.05,.951,.05):
            n=int(len(ranking)*fraction)
            candidate.load_state_dict(original)
            ids=ranking[:n];prune_units(candidate,ids)
            hits=0
            with torch.no_grad():
                for x,y in local.values():
                    hits+=int((candidate(torch.as_tensor(x,device=device)).argmax(1)==torch.as_tensor(y,device=device)).sum())
            calibration_evaluations+=count
            if hits/count < correct/count-.02-1e-12: break
            pruned=ids
        candidate.load_state_dict(original);prune_units(candidate,pruned)
    plan=make_plan(local,seed,rnd,policy)
    work=dict(optimizer_steps=0,training_examples=0,proposal_examples=0,
        diagnostic_examples=calibration_evaluations,transmitted_bytes=0,model_transmissions=0)
    model_bytes=sum(v.numel()*v.element_size() for v in candidate.state_dict().values())
    # Equal optimizer steps and training presentations to V20 (clean + second view).
    for spec in scheduled(plan,'interleaved'):
        x,y=local[spec['client']];ids=np.asarray(spec['ids'])
        bx=torch.as_tensor(x[ids],device=device);by=torch.as_tensor(y[ids],device=device)
        candidate.train();candidate.zero_grad(set_to_none=True)
        loss=F.cross_entropy(candidate(torch.cat((bx,bx))),torch.cat((by,by)))
        if not torch.isfinite(loss): raise ValueError('nonfinite baseline repair loss')
        loss.backward();cpu_sgd_step(candidate,policy.learning_rate)
        prune_units(candidate,pruned)
        work['optimizer_steps']+=1;work['training_examples']+=2*len(by)
        work['model_transmissions']+=2;work['transmitted_bytes']+=2*model_bytes
    # Activation sums and pruning accuracy votes are simulated aggregate disclosures.
    work['transmitted_bytes']+=len(ranking)*8+calibration_evaluations*8
    candidate.eval()
    return candidate,dict(work=work,pruned_indices=pruned,activation_ranking=ranking,
        pruning_accuracy_tolerance=.02,plan_sha256=plan_hash(plan),
        objective='fine_pruning_adaptation' if pruning else 'clean_fine_tuning')


class RepairController:
    def __init__(self,method,forced=False,policy=RepairPolicy(),deployment=DEPLOYMENT):
        if method not in ('periodic','finetune','fineprune'): raise ValueError(method)
        self.method=method;self.forced=forced;self.repair=policy;self.deployment=deployment

    def due(self,rnd):
        p=self.deployment
        return p.first_round<=rnd<=p.final_round and (rnd-p.first_round)%p.interval==0

    def observe(self,*args):
        # V22 periodic admission does not use its optional peak-memory gate.
        return dict(extra_transmitted_bytes=0,note='no optional peak-memory gate')

    def attempt(self,source,x,y,split,rnd,contributors,seed):
        started=time.perf_counter();parent_hash=state_hash(snapshot(source))
        groups=allocate(split.validation,contributors,seed,rnd,self.repair)
        local={c:(x[ids].copy(),y[ids].copy()) for c,ids in groups['repair'].items()}
        before={s:{c:measure(source,x[ids],y[ids]) for c,ids in groups[s].items()} for s in STAGES}
        if self.method=='periodic':
            candidate,telemetry=repair_model(source,local,'anchored_suppress',seed,rnd,self.repair,anchor=AnchorPolicy())
        else:
            candidate,telemetry=fine_tune(source,local,seed,rnd,self.repair,self.method=='fineprune')
        after={s:{c:measure(candidate,x[ids],y[ids]) for c,ids in groups[s].items()} for s in STAGES}
        decision=verify(before,after,self.repair)
        decision['base_verifier_passed']=bool(decision['passed'])
        decision['forced_admission']=self.forced
        if self.forced: decision['passed']=True
        committed=select_committed(source,candidate,decision)
        assert_hash(source,parent_hash,'baseline repair source preservation')
        panels=dict(before=before,after=after)
        response_bytes=len(json.dumps(panels,sort_keys=True).encode())
        model_bytes=sum(v.numel()*v.element_size() for v in source.state_dict().values())
        work=telemetry['work']
        payload=dict(arm=self.method,forced=self.forced,round=rnd,decision=decision,
            source_model_hash=parent_hash,candidate_model_hash=state_hash(snapshot(candidate)),
            committed_model_hash=state_hash(snapshot(committed)),held=not decision['passed'],
            work=work,verification_presentations=sum(m['example_evaluations'] for p in panels.values() for members in p.values() for m in members.values()),
            extra_transmitted_bytes=work['transmitted_bytes']+work.get('teacher_transmitted_bytes',0)+
                (2*sum(len(groups[s]) for s in STAGES)+1)*model_bytes+response_bytes,
            seconds=time.perf_counter()-started,repair_policy=asdict(self.repair))
        return Outcome(source,candidate,committed,payload,telemetry,panels,
            dict(binding=partition_binding(groups),groups={s:{str(c):ids.tolist() for c,ids in members.items()} for s,members in groups.items()}))

    persist=staticmethod(OnlineRepair.persist)
