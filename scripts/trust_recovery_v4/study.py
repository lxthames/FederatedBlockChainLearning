"""Guarded recovery development; fixed design, bounded work, exclusive lock."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .femnist import dump,digest
from .research.config import ExperimentConfig
from .research.runner_v2 import BenchmarkRunner
from .research.models import create_model
from .runtime import exclusive_lock,replay

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/guarded-recovery-v4'
OUT=ROOT/'results/guarded-recovery-v4'
DATA=ROOT/'data/trust-revision-v3-development'
SEEDS=(11,17,23)
VARIANTS=('rank_trust','guarded_recovery','guarded_penalties')

def manifest():
    rows=[]
    for attack in ('none','low_and_slow','model_replacement'):
        for variant in VARIANTS:
            for seed in SEEDS:
                c=ExperimentConfig(experiment_id=f'v4-{variant}-{attack}-s{seed}',dataset='femnist',
                    num_clients=50,participation_rate=.4,malicious_fraction=.3,rounds=40,local_epochs=1,
                    seed=seed,attack=attack,defense='two_layer',batch_size=16,learning_rate=.05,
                    optimizer_momentum=.9,device='cuda',num_classes=62,sampling_policy='bounded',
                    vote_rule='rank_loss',candidate_clip_ratio=2,trust_reward=.02,rejection_trust_penalty=.02,
                    ablation='validation_only' if variant=='rank_trust' else 'full',
                    recovery_penalties_enabled=variant=='guarded_penalties',
                    natural_splits_path=str(DATA/f'femnist-{seed}/natural-splits.json'))
                rows.append({'variant':variant,'config':c.to_dict()})
    return rows

def complete(records,variant,attack):
    rows=[r for r in records if r['variant']==variant and r['attack']==attack]
    return {r['seed']:r for r in rows} if sorted(r['seed'] for r in rows)==list(SEEDS) else None

def gate(records,variant):
    base=complete(records,'rank_trust','none');group=complete(records,variant,'none')
    if base is None or group is None:return {'passed':False,'reason':'incomplete clean groups'}
    effects=[group[s]['final_utility']-base[s]['final_utility'] for s in SEEDS]
    available=all(group[s]['unsupported_rounds']<=4 for s in SEEDS)
    # Stricter than previous study: any clean monetary penalty blocks attack work.
    no_penalties=all(group[s]['honest_validator_penalty_amount']==0 for s in SEEDS)
    passed=bool(np.mean(effects)>=-.05 and min(effects)>=-.10 and available and no_penalties)
    return {'passed':passed,'paired_clean_effects':effects,'availability_passed':available,'no_clean_penalties':no_penalties}

def analyze(records):
    compact=[]
    for r in records:
        compact.append({k:r.get(k) for k in ('variant','seed','attack','final_utility','recovery_count',
            'unsupported_rounds','honest_validator_penalty_amount','wrongful_slashing_count',
            'mean_targeted_attack_success','detection_false_positive_rate')})
    dump(JOB/'analysis.json',{'clean_gates':{v:gate(records,v) for v in VARIANTS},'runs':compact,
        'publication_ready':False,'limitation':'Reused development writer sets and seeds. Guarded checkpoint benefit is an empirical heuristic, not poisoning proof or a confidence guarantee.'})

def hashes():
    paths=list(Path(__file__).parent.rglob('*.py'))+[ROOT/'blockchain/contracts/TrustLedgerV4.sol',ROOT/'blockchain/build/TrustLedgerV4.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}

def run():
    with exclusive_lock(JOB/'coordinator.lock'):
        state={'status':'starting','pid':os.getpid(),'completed':0,'failed':0,'held':0,'systems_failed':0,
            'total':27,'publication_ready':False,'started_utc':datetime.now(timezone.utc).isoformat()}
        dump(JOB/'status.json',state)
        try:
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            torch.set_num_threads(2)
            data={}
            for seed in SEEDS:
                folder=DATA/f'femnist-{seed}';p=json.loads((folder/'provenance.json').read_text())
                for name,h in p['files'].items():
                    if digest(folder/name)!=h:raise ValueError('Data changed')
                    data[str((folder/name).relative_to(ROOT))]=h
            rows=manifest();frozen={'source':hashes(),'data':data,'manifest':rows}
            path=JOB/'frozen-inputs.json'
            if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen study changed')
            dump(path,frozen)
            dump(JOB/'environment.json',{'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'concurrent_workloads':True})
            records=[];held=[];replayed=False
            for row in rows:
                if hashes()!=frozen['source']:raise ValueError('Frozen source changed')
                c=ExperimentConfig(**row['config']);variant=row['variant']
                state.update(status='running',current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat());dump(JOB/'status.json',state)
                if c.attack!='none':
                    cg=gate(records,variant)
                    if not cg['passed']:
                        held.append({'experiment_id':c.experiment_id,'clean_gate':cg});state['held']+=1
                        dump(JOB/'held.json',held);continue
                output=OUT/c.experiment_id
                try:
                    if (output/'summary.json').exists():
                        s=json.loads((output/'summary.json').read_text())
                        expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
                        if s.get('config_hash')!=expected or s.get('status')!='completed':raise ValueError('Invalid completed output')
                    else:
                        if output.exists() and any(output.iterdir()):output.rename(output.with_name(output.name+f'.interrupted-{time.time_ns()}'))
                        folder=DATA/f'femnist-{c.seed}'
                        x,y,ex,ey=[np.load(folder/f'{n}.npy',allow_pickle=False) for n in ('features','labels','development_features','development_labels')]
                        s=BenchmarkRunner(c,output).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],62),(ex,ey))
                    s.update(variant=variant,evaluation_role='training-only development holdout');dump(output/'summary.json',s)
                    records.append(s);state['completed']+=1;analyze(records)
                    if variant=='guarded_recovery' and not replayed:
                        try:replay(c,output,s);replayed=True
                        except Exception:
                            state['systems_failed']+=1
                            with (JOB/'systems-failures.jsonl').open('a') as f:f.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                except Exception:
                    state['failed']+=1
                    with (JOB/'failures.jsonl').open('a') as f:f.write(json.dumps({'config':c.to_dict(),'error':traceback.format_exc()})+'\n')
                dump(JOB/'status.json',state)
            analyze(records)
            state['status']='completed_with_gaps' if state['failed'] or state['held'] or state['systems_failed'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());raise
        finally:
            state['updated_utc']=state['finished_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)

if __name__=='__main__':run()
