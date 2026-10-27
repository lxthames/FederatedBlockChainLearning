"""Small literature-guided screening with explicit clean and attack gates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import hashlib
import json
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from .femnist import dump,digest
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/trust-math-lab-v1'
OUT=ROOT/'results/trust-math-lab-v1'
DATA=ROOT/'data/trust-revision-v3-development'
SEEDS=(11,17,23)
APPROACHES=('paired_loss','scaled_loss','rank_loss')
VARIANTS=('fedavg','v3_full','validation_only_dynamic','accuracy_fixed','accuracy_clipped_fixed')+APPROACHES

def make_config(variant,seed,attack='none'):
    fixed=variant not in ('v3_full','validation_only_dynamic')
    return ExperimentConfig(experiment_id=f'math-{variant}-{attack}-s{seed}',
        dataset='femnist',num_clients=50,participation_rate=.4,malicious_fraction=.3,
        seed=seed,rounds=40,local_epochs=1,batch_size=16,learning_rate=.05,optimizer_momentum=.9,
        device='cuda',num_classes=62,sampling_policy='bounded',attack=attack,
        defense='fedavg' if variant=='fedavg' else 'two_layer',
        ablation='full' if variant=='v3_full' else 'validation_only',
        trust_reward=0 if fixed else .02,rejection_trust_penalty=0 if fixed else .02,
        vote_rule=variant if variant in APPROACHES else 'accuracy',
        candidate_clip_ratio=2. if variant in APPROACHES or variant=='accuracy_clipped_fixed' else 0.,
        natural_splits_path=str(DATA/f'femnist-{seed}/natural-splits.json'))

def manifest():
    rows=[{'stage':'clean','variant':v,'config':make_config(v,s).to_dict()} for s in SEEDS for v in VARIANTS]
    rows += [{'stage':'attack_control','variant':'fedavg','config':make_config('fedavg',s,a).to_dict()}
             for a in ('low_and_slow','model_replacement') for s in SEEDS]
    rows += [{'stage':'attack_screen','variant':v,'config':make_config(v,s,a).to_dict()}
             for v in APPROACHES for a in ('low_and_slow','model_replacement') for s in SEEDS]
    return rows

def source_hashes():
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}

def clean_gate(records,variant):
    b={r['seed']:r for r in records if r['variant']=='fedavg' and r['attack']=='none'}
    v={r['seed']:r for r in records if r['variant']==variant and r['attack']=='none'}
    if set(b)!=set(SEEDS) or set(v)!=set(SEEDS):return {'passed':False,'reason':'incomplete seed groups'}
    effects=[v[s]['final_utility']-b[s]['final_utility'] for s in SEEDS]
    learning=all(b[s]['final_utility']>b[s]['development_majority_accuracy']+.05 for s in SEEDS)
    return {'passed':bool(learning and np.mean(effects)>=-.05 and min(effects)>=-.10),
            'paired_clean_effects':effects,'baseline_learning_passed':learning}

def attack_gate(records,attack):
    b={r['seed']:r for r in records if r['variant']=='fedavg' and r['attack']=='none'}
    a={r['seed']:r for r in records if r['variant']=='fedavg' and r['attack']==attack}
    if set(a)!=set(SEEDS) or set(b)!=set(SEEDS):return {'passed':False,'reason':'incomplete attack controls'}
    effects=[b[s]['final_utility']-a[s]['final_utility'] if attack=='low_and_slow'
             else a[s]['mean_targeted_attack_success']-b[s]['mean_clean_trigger_prediction_rate'] for s in SEEDS]
    threshold=.005 if attack=='low_and_slow' else .05
    return {'passed':bool(np.mean(effects)>threshold),'effects':effects,'threshold':threshold}

def analyze(records):
    clean={v:clean_gate(records,v) for v in VARIANTS}
    attacks={a:attack_gate(records,a) for a in ('low_and_slow','model_replacement')}
    compact=[]
    for r in records:
        n=sum(r['client_submission_counts'].values());rejected=sum(r['client_rejection_counts'].values())
        compact.append({k:r.get(k) for k in ('variant','attack','seed','final_utility','recovery_count',
            'unsupported_rounds','honest_validator_penalty_amount','mean_targeted_attack_success')}|
            {'accepted_fraction':1-rejected/max(1,n)})
    dump(JOB/'analysis.json',{'clean_gates':clean,'attack_gates':attacks,'runs':compact,
         'publication_ready':False,'scope':'Exploratory FEMNIST screening; successful vote rules still require recovery reintegration, independent development replication and final-protocol validation.'})

def run():
    JOB.mkdir(parents=True,exist_ok=True)
    state={'status':'starting','pid':os.getpid(),'completed':0,'failed':0,'held':0,'total':48,
        'publication_ready':False,'started_utc':datetime.now(timezone.utc).isoformat()}
    dump(JOB/'status.json',state)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(2)
        files={}
        for seed in SEEDS:
            folder=DATA/f'femnist-{seed}'
            p=json.loads((folder/'provenance.json').read_text())
            for name,h in p['files'].items():
                if digest(folder/name)!=h:raise RuntimeError('Training-only data changed')
                files[str((folder/name).relative_to(DATA))]=h
        rows=manifest();frozen={'source':source_hashes(),'data':files,'manifest':rows}
        path=JOB/'frozen-inputs.json'
        if path.exists() and json.loads(path.read_text())!=frozen:raise RuntimeError('Frozen design changed')
        dump(path,frozen)
        dump(JOB/'environment.json',{'torch':torch.__version__,'numpy':np.__version__,
            'gpu':torch.cuda.get_device_name(0),'threads':2,'concurrent_workloads':True})
        records=[];held=[]
        for row in rows:
            c=ExperimentConfig(**row['config']);v=row['variant']
            if source_hashes()!=frozen['source']:raise RuntimeError('Frozen source changed')
            state.update(status='running',stage=row['stage'],current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
            if row['stage']=='attack_screen':
                cg=clean_gate(records,v);ag=attack_gate(records,c.attack)
                if not cg['passed'] or not ag['passed']:
                    held.append({'experiment_id':c.experiment_id,'clean_gate':cg,'attack_gate':ag})
                    state['held']+=1;dump(JOB/'held.json',held);continue
            output=OUT/c.experiment_id
            try:
                if (output/'summary.json').exists():
                    s=json.loads((output/'summary.json').read_text())
                    expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
                    if expected!=s['config_hash']:raise RuntimeError('Config mismatch')
                else:
                    if output.exists() and any(output.iterdir()):output.rename(output.with_name(output.name+f'.interrupted-{time.time_ns()}'))
                    folder=DATA/f'femnist-{c.seed}'
                    x,y,ex,ey=[np.load(folder/f'{n}.npy',allow_pickle=False) for n in ('features','labels','development_features','development_labels')]
                    s=BenchmarkRunner(c,output).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],62),(ex,ey))
                # Safe completion after a crash between runner summary and metadata.
                ey=np.load(DATA/f'femnist-{c.seed}/development_labels.npy',allow_pickle=False)
                s.update(variant=v,development_majority_accuracy=float(np.bincount(ey).max()/len(ey)),evaluation_role='training-only development holdout')
                dump(output/'summary.json',s)
                records.append(s);state['completed']+=1;analyze(records)
            except Exception:
                state['failed']+=1
                with (JOB/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'config':c.to_dict(),'error':traceback.format_exc()})+'\n')
            dump(JOB/'status.json',state)
        state['status']='completed_with_gaps' if state['failed'] or state['held'] else 'completed'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)

if __name__=='__main__':run()
