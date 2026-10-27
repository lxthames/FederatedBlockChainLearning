"""Fixed-example private reference monitoring, without attack/evaluation inputs."""
import copy
from dataclasses import asdict,dataclass
import json
import time

import numpy as np
import torch
from scripts.trust_online_v21.online import OnlineRepair as BaseRepair,DeploymentPolicy,select_committed
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash,assert_hash
from scripts.trust_repair_v16_fixed.research.splits import stream,fingerprint

DEPLOYMENT=DeploymentPolicy(first_round=25,interval=25,final_round=100)


@dataclass(frozen=True)
class MemoryPolicy:
    step_tolerance: float = .0025
    peak_tolerance: float = .015


def panel_decision(before,after,peak,policy=MemoryPolicy()):
    if set(before)!=set(after) or set(before)!=set(peak) or not before:
        raise ValueError('incomplete paired memory responses')
    n=sum(v['n'] for v in before.values())
    for client in before:
        b,a,p=before[client],after[client],peak[client]
        if (b['n']<2 or b['n']!=a['n'] or b['binding']!=a['binding'] or
            not 0<=b['correct']<=p<=b['n'] or not 0<=a['correct']<=a['n']):
            raise ValueError('memory examples or peak changed')
    source=sum(v['correct'] for v in before.values())/n
    candidate=sum(v['correct'] for v in after.values())/n
    reference=sum(peak.values())/n
    floor=max(source-policy.step_tolerance,reference-policy.peak_tolerance)
    return dict(passed=bool(candidate>=floor-1e-12),n=n,source_accuracy=source,
                candidate_accuracy=candidate,reference_accuracy=reference,floor=floor,
                step_accuracy_delta=candidate-source,peak_accuracy_delta=candidate-reference)


class OnlineRepair(BaseRepair):
    def __init__(self,arm,deployment=DEPLOYMENT,repair=None,anchor=None,memory=MemoryPolicy()):
        if arm not in ('v20_periodic','v20_peak'):raise ValueError('unknown V22 arm')
        kwargs={}
        if repair is not None:kwargs['repair']=repair
        if anchor is not None:kwargs['anchor']=anchor
        super().__init__('v20_repair',deployment,**kwargs)
        if not 0<=memory.step_tolerance<=memory.peak_tolerance<1:raise ValueError('invalid memory tolerances')
        self.variant=arm;self.memory=memory;self.ids=None;self.peaks={};self.current={};self.last_round=0
        self.current_hash=None;self.fixed_binding=None

    def clean_counts(self,model,x,y,clients):
        device=next(model.parameters()).device;values={};model.eval()
        with torch.no_grad():
            for client in clients:
                ids=self.ids[client]
                logits=model(torch.as_tensor(x[ids],dtype=torch.float32,device=device))
                if not torch.isfinite(logits).all():raise ValueError('nonfinite private monitor response')
                labels=torch.as_tensor(y[ids],dtype=torch.long,device=device)
                values[client]=dict(n=len(ids),correct=int((logits.argmax(1)==labels).sum().cpu()),
                                    binding=fingerprint(x[ids],y[ids]))
        return values

    def observe(self,model,x,y,split,rnd,seed):
        started=time.perf_counter()
        if rnd!=self.last_round+1:raise ValueError('private reference monitoring must cover every round in order')
        if self.ids is None:
            self.ids={client:np.asarray(ids)[stream(seed,'v22-fixed-private-examples',client).permutation(len(ids))][
                :self.repair.examples_per_client] for client,ids in enumerate(split.validation)}
            if any(len(ids)<2 for ids in self.ids.values()):raise ValueError('insufficient fixed private examples')
            self.fixed_binding={str(c):fingerprint(ids) for c,ids in self.ids.items()}
        if self.fixed_binding!={str(c):fingerprint(ids) for c,ids in self.ids.items()}:
            raise ValueError('fixed monitor example IDs changed')
        before=state_hash(snapshot(model));values=self.clean_counts(model,x,y,self.ids)
        if self.current and any(values[c]['binding']!=self.current[c]['binding'] for c in values):
            raise ValueError('fixed monitor examples changed')
        self.current=values
        self.peaks={c:max(self.peaks.get(c,0),v['correct']) for c,v in values.items()}
        self.current_hash=before;self.last_round=rnd;assert_hash(model,before,'private reference inference')
        responses=dict(round=rnd,model_hash=before,clients=values,peak_correct_counts=self.peaks,
                       fixed_index_bindings=self.fixed_binding)
        response_bytes=len(json.dumps(responses,sort_keys=True,separators=(',',':')).encode())
        model_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values())
        return dict(responses,example_presentations=sum(v['n'] for v in values.values()),
            model_transmissions=len(values),extra_transmitted_bytes=len(values)*model_bytes+response_bytes,
            response_bytes=response_bytes,seconds=time.perf_counter()-started)

    def attempt(self,source,x,y,split,rnd,contributors,seed):
        if self.last_round!=rnd or self.current_hash!=state_hash(snapshot(source)):
            raise ValueError('current private reference not measured on the repair source')
        outcome=super().attempt(source,x,y,split,rnd,contributors,seed);started=time.perf_counter()
        reports={};details={};examples=0
        for stage in ('screen','verify'):
            clients=[int(c) for c in outcome.partitions['groups'][stage]]
            before={c:self.current[c] for c in clients}
            after=self.clean_counts(outcome.candidate,x,y,clients)
            peak={c:self.peaks[c] for c in clients}
            reports[stage]=panel_decision(before,after,peak,self.memory)
            details[stage]=dict(before=before,after=after,peak_correct_counts=peak)
            examples+=sum(v['n'] for v in after.values())
        assert_hash(outcome.candidate,outcome.payload['candidate_model_hash'],'private memory gate inference')
        decision=copy.deepcopy(outcome.payload['decision']);original=decision['passed']
        memory_pass=all(v['passed'] for v in reports.values())
        decision.update(base_verifier_passed=original,memory_gate=reports,memory_gate_applied=self.variant=='v20_peak')
        decision['passed']=bool(original and (memory_pass or self.variant=='v20_periodic'))
        outcome.committed=select_committed(source,outcome.candidate,decision)
        response_bytes=len(json.dumps(details,sort_keys=True,separators=(',',':')).encode())
        outcome.payload.update(arm=self.variant,decision=decision,held=not decision['passed'],memory_policy=asdict(self.memory),
            committed_model_hash=state_hash(snapshot(outcome.committed)),memory_candidate_presentations=examples,
            memory_response_bytes=response_bytes,
            extra_transmitted_bytes=outcome.payload['extra_transmitted_bytes']+response_bytes,
            seconds=outcome.payload['seconds']+time.perf_counter()-started)
        outcome.panels['fixed_example_memory']=details
        return outcome

    @staticmethod
    def persist(outcome,folder,rnd):
        BaseRepair.persist(outcome,folder,rnd)
        if rnd!=100:
            for name,model in (('candidate',outcome.candidate),('committed',outcome.committed)):
                state=snapshot(model)
                torch.save(dict(model_hash=state_hash(state),state=state),folder/'online'/f'round-{rnd:03}'/f'{name}.pt')
