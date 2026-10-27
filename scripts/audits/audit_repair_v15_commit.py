"""Isolated instrumented replay of V15; never edits frozen experiment sources."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.trust_repair_v15 import study
from scripts.trust_repair_v15.repair import RepairRound
from scripts.trust_repair_v15.research import runner_v2 as runner_module
from scripts.trust_repair_v15.research.runner_v2 import model_hash
from scripts.trust_repair_v15.research.splits import fingerprint


class FinishedReplay(Exception):
    pass


def state_hash(state):
    return fingerprint(*[v.detach().cpu().numpy() for v in state.values()])


def run(arm,seed,rounds):
    folder=study.ROOT/'artifacts'/f'repair-v15-commit-audit-{arm}-{seed}-{time.time_ns()}'
    c=study.config('fashion_mnist',arm,seed)
    runner=runner_module.BenchmarkRunner(c,folder)
    runner.challenge_arm=arm
    runner.challenge_thresholds=study.read(study.JOB/'calibration.json')['thresholds'][c.dataset]
    original_apply=runner_module._apply_delta
    original_class=runner_module.ChallengeRound
    original_log=runner.logger.log
    records=[]
    context={}
    reference=study.read(study.OUT/c.experiment_id/'chain-trace.json')

    class CheckedRound(RepairRound):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            context.update(model=self.parent,round=self.rnd,record=dict(round=self.rnd,
                incoming_hash=model_hash(self.parent),updates=[dict(client=u.client_id,
                    sample_count=u.sample_count,hash=state_hash(u.state)) for u in self.updates]))

        def evaluate(self,result,stage,omitted=None):
            before=model_hash(self.parent)
            delta_before=state_hash(result.state)
            verdict=super().evaluate(result,stage,omitted)
            assert before==model_hash(self.parent),'validation mutated parent'
            assert delta_before==state_hash(result.state),'validation mutated proposed delta'
            return verdict

        def aggregate(self,result):
            before=model_hash(self.parent)
            context['record']['proposed_delta_hash']=state_hash(result.state)
            checked=copy.deepcopy(self.parent)
            original_apply(checked,dict(result.state))
            context['record']['proposed_model_hash']=model_hash(checked)
            if self.rnd==9:
                torch.save(dict(parent={k:v.detach().cpu() for k,v in self.parent.state_dict().items()},
                    proposed_delta={k:v.detach().cpu() for k,v in result.state.items()},
                    expected={k:v.detach().cpu() for k,v in checked.state_dict().items()},
                    updates=[dict(client=u.client_id,sample_count=u.sample_count,state=u.state) for u in self.updates]),
                    folder/'round-9-boundary.pt')
            returned,held=super().aggregate(result)
            assert before==model_hash(self.parent),'aggregate gate mutated parent'
            context['record'].update(held=held,returned_delta_hash=state_hash(returned.state),
                accepted=list(returned.accepted_client_ids))
            checked=copy.deepcopy(self.parent)
            original_apply(checked,dict(returned.state))
            context['expected']=model_hash(checked)
            return returned,held

    def checked_apply(model,delta):
        is_global=model is context.get('model')
        if is_global:
            assert state_hash(delta)==context['record']['returned_delta_hash'],'delta changed before commit'
            assert model_hash(model)==context['record']['incoming_hash'],'parent changed before commit'
        original_apply(model,delta)
        if is_global:
            actual=model_hash(model)
            assert actual==context['expected'],'committed model differs from evaluated return'
            context['record']['committed_hash']=actual

    def log(kind,rnd,payload):
        original_log(kind,rnd,payload)
        if kind=='round_completed':
            record=context['record']
            record.update(accuracy=payload['accuracy'],reference_hash=reference[rnd]['model_hash'],
                matches_reference=record['committed_hash']==reference[rnd]['model_hash'])
            records.append(record)
            print(json.dumps({k:record[k] for k in ('round','accuracy','held','matches_reference','committed_hash')}),flush=True)
            study.dump(folder/'audit-progress.json',records)
            if rnd>=rounds:raise FinishedReplay()

    runner_module._apply_delta=checked_apply
    runner_module.ChallengeRound=CheckedRound
    runner.logger.log=log
    data=study.DATA/f'{c.dataset}-{seed}'
    arrays=[np.load(data/f'{name}.npy',allow_pickle=False) for name in
        ('features','labels','development_features','development_labels')]
    x,y,ex,ey=arrays
    error=None
    try:
        runner.run_arrays(x,y,lambda:study.create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(ex,ey))
    except FinishedReplay:
        pass
    except Exception:
        import traceback
        error=traceback.format_exc()
        raise
    finally:
        runner_module._apply_delta=original_apply
        runner_module.ChallengeRound=original_class
        study.dump(folder/'audit.json',dict(arm=arm,seed=seed,records=records,error=error,
            frozen_inputs_match=study.inputs()==study.read(study.JOB/'frozen-inputs.json')))
        print('AUDIT_DIRECTORY',folder,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('aggregate','repair'),default='aggregate')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--rounds',type=int,default=11)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        run(args.arm,args.seed,args.rounds)
