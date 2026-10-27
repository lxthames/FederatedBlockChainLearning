"""V26 repair and client/update baseline adapters.

The NAD adapter is a documented federated adaptation: a clean teacher is trained
from the repair panels, then the poisoned parent is updated with clean CE plus
normalized convolutional attention transfer to that teacher. It uses no test
arrays or attack labels. FLTrust is the client/update baseline supplied by the
frozen V17 aggregation implementation.
"""
import copy, json, time
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from scripts.trust_online_v21.online import Outcome, select_committed
from scripts.trust_supervised_v18.repair import allocate,measure,verify,STAGES,cpu_sgd_step,RepairPolicy,partition_binding
from scripts.trust_interleave_v19.repair import scheduled,make_plan,plan_hash
from scripts.trust_anchor_v20.repair import repair_model,AnchorPolicy
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash,assert_hash
from scripts.trust_periodic_v22.online import DEPLOYMENT


def _attention(model, bx):
    # Shared small CNN: use the last convolutional activation map.
    h=model[:6](bx)
    a=h.abs().mean(dim=1)
    return a/(a.flatten(1).norm(p=2,dim=1,keepdim=True).view(-1,1,1)+1e-8)


def _nad_tune(source, local, seed, rnd, policy):
    teacher=copy.deepcopy(source); student=copy.deepcopy(source)
    device=next(source.parameters()).device
    plan=make_plan(local,seed,rnd,policy)
    work=dict(teacher_optimizer_steps=0,student_optimizer_steps=0,training_examples=0,
              attention_examples=0,teacher_transmitted_bytes=0,transmitted_bytes=0,
              proposal_examples=0,diagnostic_examples=0,model_transmissions=0)
    model_bytes=sum(v.numel()*v.element_size() for v in source.state_dict().values())
    # Stage 1: clean teacher on the same repair schedule.
    for spec in scheduled(plan,'interleaved'):
        x,y=local[spec['client']]; ids=np.asarray(spec['ids'])
        bx=torch.as_tensor(x[ids],device=device); by=torch.as_tensor(y[ids],device=device)
        teacher.train(); teacher.zero_grad(set_to_none=True)
        loss=F.cross_entropy(teacher(torch.cat((bx,bx))),torch.cat((by,by)))
        loss.backward(); cpu_sgd_step(teacher,policy.learning_rate)
        work['teacher_optimizer_steps']+=1; work['training_examples']+=2*len(by)
    teacher.eval()
    # Stage 2: student repair with clean CE and normalized attention transfer.
    for spec in scheduled(plan,'interleaved'):
        x,y=local[spec['client']]; ids=np.asarray(spec['ids'])
        bx=torch.as_tensor(x[ids],device=device); by=torch.as_tensor(y[ids],device=device)
        with torch.no_grad(): target=_attention(teacher,torch.cat((bx,bx)))
        student.train(); student.zero_grad(set_to_none=True)
        batch=torch.cat((bx,bx)); labels=torch.cat((by,by))
        logits=student(batch)
        loss=F.cross_entropy(logits,labels)+F.mse_loss(_attention(student,batch),target)
        if not torch.isfinite(loss): raise ValueError('nonfinite NAD loss')
        loss.backward(); cpu_sgd_step(student,policy.learning_rate)
        work['student_optimizer_steps']+=1; work['attention_examples']+=len(labels)
        work['training_examples']+=len(labels)
    work['proposal_examples']=work['training_examples']
    work['model_transmissions']=2*(work['teacher_optimizer_steps']+work['student_optimizer_steps'])
    work['teacher_transmitted_bytes']=work['teacher_optimizer_steps']*model_bytes
    work['transmitted_bytes']=work['model_transmissions']*model_bytes
    return student,dict(work=work,plan_sha256=plan_hash(plan),objective='nad_teacher_attention_adaptation',
        teacher_steps=work['teacher_optimizer_steps'],student_steps=work['student_optimizer_steps'],attention_weight=1.0)


class RepairController:
    def __init__(self,method,forced=False,policy=RepairPolicy(),deployment=DEPLOYMENT):
        if method not in ('periodic','nad'): raise ValueError(method)
        self.method=method; self.forced=forced; self.repair=policy; self.deployment=deployment
    def due(self,rnd):
        p=self.deployment
        return p.first_round<=rnd<=p.final_round and (rnd-p.first_round)%p.interval==0
    def observe(self,*args): return dict(extra_transmitted_bytes=0,note='no optional peak-memory gate')
    def attempt(self,source,x,y,split,rnd,contributors,seed):
        started=time.perf_counter(); parent_hash=state_hash(snapshot(source))
        groups=allocate(split.validation,contributors,seed,rnd,self.repair)
        local={c:(x[ids].copy(),y[ids].copy()) for c,ids in groups['repair'].items()}
        before={s:{c:measure(source,x[ids],y[ids]) for c,ids in groups[s].items()} for s in STAGES}
        if self.method=='periodic': candidate,telemetry=repair_model(source,local,'anchored_suppress',seed,rnd,self.repair,anchor=AnchorPolicy())
        else: candidate,telemetry=_nad_tune(source,local,seed,rnd,self.repair)
        after={s:{c:measure(candidate,x[ids],y[ids]) for c,ids in groups[s].items()} for s in STAGES}
        decision=verify(before,after,self.repair); decision['base_verifier_passed']=bool(decision['passed']); decision['forced_admission']=self.forced
        if self.forced: decision['passed']=True
        committed=select_committed(source,candidate,decision)
        assert_hash(source,parent_hash,'baseline repair source preservation')
        panels=dict(before=before,after=after); response_bytes=len(json.dumps(panels,sort_keys=True).encode())
        model_bytes=sum(v.numel()*v.element_size() for v in source.state_dict().values()); work=telemetry['work']
        payload=dict(arm=self.method,forced=self.forced,round=rnd,decision=decision,
          source_model_hash=parent_hash,candidate_model_hash=state_hash(snapshot(candidate)),
          committed_model_hash=state_hash(snapshot(committed)),held=not decision['passed'],work=work,
          verification_presentations=sum(m['example_evaluations'] for p in panels.values() for members in p.values() for m in members.values()),
          extra_transmitted_bytes=work['transmitted_bytes']+work.get('teacher_transmitted_bytes',0)+(2*sum(len(groups[s]) for s in STAGES)+1)*model_bytes+response_bytes,
          seconds=time.perf_counter()-started,repair_policy=asdict(self.repair),baseline_adapter='nad_v1' if self.method=='nad' else 'v20_periodic')
        return Outcome(source,candidate,committed,payload,telemetry,panels,dict(binding=partition_binding(groups),groups={s:{str(c):ids.tolist() for c,ids in members.items()} for s,members in groups.items()}))
    persist=staticmethod(__import__('scripts.trust_online_v21.online',fromlist=['OnlineRepair']).OnlineRepair.persist)
