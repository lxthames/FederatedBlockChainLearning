"""Attack-agnostic online repair. No test data or attack configuration imports."""
import copy
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import time

import torch
from scripts.trust_anchor_v20.repair import repair_model as anchored_repair,AnchorPolicy
from scripts.trust_interleave_v19.repair import repair_model as baseline_repair,make_plan,plan_hash
from scripts.trust_supervised_v18.repair import RepairPolicy,allocate,partition_binding,measure,verify,STAGES,checked_load
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash,assert_hash

ARMS=('clip_only','current','v19_repair','v20_repair')


def select_committed(source,candidate,decision):
    if state_hash(snapshot(candidate))!=decision['scored_model_hash']:
        raise ValueError('candidate changed after scoring')
    result=copy.deepcopy(source)
    checked_load(result,snapshot(candidate if decision['passed'] else source),'online selected model')
    return result


@dataclass(frozen=True)
class DeploymentPolicy:
    first_round: int = 10
    interval: int = 1
    final_round: int = 100


@dataclass
class Outcome:
    source: object
    candidate: object
    committed: object
    payload: dict
    telemetry: dict
    panels: dict
    partitions: dict


class OnlineRepair:
    def __init__(self,arm,deployment=DeploymentPolicy(),repair=RepairPolicy(),anchor=AnchorPolicy()):
        if arm not in ('v19_repair','v20_repair'):raise ValueError('online repair arm required')
        if min(deployment.first_round,deployment.interval)<1 or deployment.final_round<deployment.first_round:
            raise ValueError('invalid fixed deployment schedule')
        self.arm,self.deployment,self.repair,self.anchor=arm,deployment,repair,anchor

    def due(self,rnd):
        p=self.deployment
        return p.first_round<=rnd<=p.final_round and (rnd-p.first_round)%p.interval==0

    def attempt(self,source,x,y,split,rnd,contributors,seed):
        if not self.due(rnd):raise ValueError('unscheduled repair')
        start=time.perf_counter();source_hash=state_hash(snapshot(source))
        groups=allocate(split.validation,contributors,seed,rnd,self.repair)
        local={client:(x[ids].copy(),y[ids].copy()) for client,ids in groups['repair'].items()}
        plan=make_plan(local,seed,rnd,self.repair)
        before={stage:{client:measure(source,x[ids],y[ids]) for client,ids in groups[stage].items()}
                for stage in STAGES}
        if self.arm=='v19_repair':
            candidate,telemetry=baseline_repair(source,local,'hard_patch','interleaved',seed,rnd,self.repair,plan)
        else:
            candidate,telemetry=anchored_repair(source,local,'anchored_suppress',seed,rnd,self.repair,plan,self.anchor)
        after={stage:{client:measure(candidate,x[ids],y[ids]) for client,ids in groups[stage].items()}
               for stage in STAGES}
        decision=verify(before,after,self.repair)
        committed=select_committed(source,candidate,decision)
        committed_hash=state_hash(snapshot(committed));candidate_hash=state_hash(snapshot(candidate))
        assert_hash(source,source_hash,'online repair source preservation')
        expected=candidate_hash if decision['passed'] else source_hash
        if committed_hash!=expected:raise ValueError('online admission selected a different model')
        # Two scored models delivered to each validator, plus one selected model
        # returned to the aggregator. Responses have actual serialized size.
        model_bytes=sum(v.numel()*v.element_size() for v in source.state_dict().values())
        panels=dict(before=before,after=after)
        response_bytes=len(json.dumps(panels,sort_keys=True,separators=(',',':')).encode())
        panel_transmissions=2*sum(len(groups[s]) for s in STAGES)
        verification_presentations=sum(m['example_evaluations'] for panel in panels.values()
                                       for members in panel.values() for m in members.values())
        work=telemetry['work']
        extra=(work['transmitted_bytes']+work.get('teacher_transmitted_bytes',0)+
               (panel_transmissions+1)*model_bytes+response_bytes)
        payload=dict(arm=self.arm,round=rnd,source_model_hash=source_hash,candidate_model_hash=candidate_hash,
            committed_model_hash=committed_hash,decision=decision,held=not decision['passed'],
            deployment=asdict(self.deployment),repair_policy=asdict(self.repair),
            anchor_policy=asdict(self.anchor) if self.arm=='v20_repair' else None,
            plan_sha256=plan_hash(plan),work=work,verification_presentations=verification_presentations,
            panel_model_transmissions=panel_transmissions,selected_model_return_transmissions=1,
            panel_response_bytes=response_bytes,extra_transmitted_bytes=extra,seconds=time.perf_counter()-start,
            note='A rejected repair retains the quorum aggregate. Role/example reuse across rounds is possible.')
        return Outcome(source,candidate,committed,payload,telemetry,panels,
                       dict(binding=partition_binding(groups),groups={s:{str(c):ids.tolist() for c,ids in members.items()}
                                                                  for s,members in groups.items()}))

    @staticmethod
    def persist(outcome,folder,rnd):
        path=Path(folder)/'online'/f'round-{rnd:03}'
        path.mkdir(parents=True,exist_ok=False)
        for name,value in (('decision',outcome.payload),('repair',outcome.telemetry),
                           ('panels',outcome.panels),('partitions',outcome.partitions)):
            (path/f'{name}.json').write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding='utf-8')
        if rnd in (10,30,100):
            for name,model in (('candidate',outcome.candidate),('committed',outcome.committed)):
                state=snapshot(model)
                torch.save(dict(model_hash=state_hash(state),state=state),path/f'{name}.pt')
