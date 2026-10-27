"""Bounded, training-only development; no confirmation or official-test access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
from .femnist import dump, digest
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner
from .research.dataio import load_experiment_arrays
from .research.splits import stream

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/trust-revision-v3-development'
OUT=ROOT/'results/trust-revision-v3-development'
DATA=ROOT/'data/trust-revision-v3-development'
SEEDS=(11,17,23)

def hashes():
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}

def split_training_pool(x,y,split,seed):
    """Carve evaluation out of training; retain private validation/reference roles."""
    hold=[]; groups=[]
    for client,indices in enumerate(split['train']):
        order=stream(seed,'development-holdout',client).permutation(indices)
        n=max(1,len(order)//5)
        if len(order)-n<1: raise ValueError('insufficient development training data')
        hold.append(order[:n]);groups.append(order[n:])
    keep=np.sort(np.concatenate(groups+[np.asarray(v) for v in split['validation']]+[np.asarray(split['reference'])]))
    held=np.concatenate(hold)
    remap={int(old):new for new,old in enumerate(keep)}
    new=dict(split)
    for name,values in [('train',groups),('validation',split['validation'])]:
        new[name]=[[remap[int(i)] for i in g] for g in values]
    new['reference']=[remap[int(i)] for i in split['reference']]
    new['seed']=seed;new['test']=[];new['test_clients']=[]
    offset=0
    for group in hold:
        new['test_clients'].append(list(range(offset,offset+len(group))));offset+=len(group)
    assert not set(held)&set(keep)
    assert len(held)+len(keep)==len(y)
    return x[keep],y[keep],x[held],y[held],new

def prepare_data():
    DATA.mkdir(parents=True,exist_ok=True)
    for seed,source_seed in zip(SEEDS,(3,5,7)):
        for dataset in ('fashion_mnist','femnist'):
            folder=DATA/f'{dataset}-{seed}'
            if (folder/'provenance.json').exists():
                record=json.loads((folder/'provenance.json').read_text())
                if any(digest(folder/n)!=h for n,h in record['files'].items()):
                    raise RuntimeError('Development data changed')
                continue
            folder.mkdir(exist_ok=True)
            if dataset=='femnist':
                source=ROOT/f'data/femnist-tff/seed-{source_seed}'
                # Read training pool only, never test_features/test_labels.
                x=np.load(source/'features.npy');y=np.load(source/'labels.npy')
                split=json.loads((source/'natural-splits.json').read_text())
                x,y,ex,ey,new=split_training_pool(x,y,split,seed)
                dump(folder/'natural-splits.json',new)
                source_info={'source_training_seed':source_seed,
                    'source_files':{n:digest(source/n) for n in ('features.npy','labels.npy','natural-splits.json')}}
            else:
                c=ExperimentConfig('load-training',dataset,50,0.4,seed=seed)
                x,y=load_experiment_arrays(c,ROOT/'data')
                # Exact pixel duplicates cannot straddle development evaluation.
                seen=set();keep=[]
                for i,v in enumerate(x):
                    h=hashlib.sha256(v.tobytes()).digest()
                    if h not in seen: keep.append(i);seen.add(h)
                x,y=x[keep],y[keep]
                tr=[];ev=[]
                for label in np.unique(y):
                    idx=stream(seed,'development-holdout',int(label)).permutation(np.flatnonzero(y==label))
                    n=max(1,len(idx)//5);ev.extend(idx[:n]);tr.extend(idx[n:])
                ex,ey=x[ev],y[ev];x,y=x[tr],y[tr]
                source_info={'source':'official training pool only','exact_pixel_deduplicated':True}
            for name,value in [('features',x),('labels',y),('development_features',ex),('development_labels',ey)]:
                np.save(folder/f'{name}.npy',value,allow_pickle=False)
            dump(folder/'provenance.json',dict(source_info,seed=seed,
                evaluation_role='development only; not publication test',
                files={p.name:digest(p) for p in folder.iterdir() if p.suffix in {'.json','.npy'}}))

def manifest():
    rows=[]
    for dataset in ('femnist','fashion_mnist'):
        for seed in SEEDS:
            # Same optimizer/budget for both methods. One local epoch is retained.
            for method in ('fedavg','two_layer'):
                for attack in ('none','low_and_slow','model_replacement'):
                    c=ExperimentConfig('pending',dataset,50,0.4,malicious_fraction=0.3,
                        dirichlet_alpha=0.5 if dataset=='fashion_mnist' else None,
                        rounds=100 if dataset=='femnist' else 40,local_epochs=1,seed=seed,
                        attack=attack,defense=method,device='cuda',sampling_policy='bounded',
                        learning_rate=0.05 if dataset=='femnist' else 0.01,
                        optimizer_momentum=0.9 if dataset=='femnist' else 0.0,
                        batch_size=16 if dataset=='femnist' else 64,
                        num_classes=62 if dataset=='femnist' else 10,
                        natural_splits_path=str(DATA/f'{dataset}-{seed}/natural-splits.json') if dataset=='femnist' else None)
                    rows.append(replace(c,experiment_id=f'v3-dev-{dataset}-{method}-{attack}-s{seed}'))
    # Run all clean feasibility cells before attacked cells; do not tune on attacks.
    return sorted(rows,key=lambda c:c.attack!='none')

def analyze(records):
    groups=[]
    for dataset in ('femnist','fashion_mnist'):
        subset=[r for r in records if r['dataset']==dataset]
        clean={r['seed']:r for r in subset if r['defense']=='fedavg' and r['attack']=='none'}
        full={r['seed']:r for r in subset if r['defense']=='two_layer' and r['attack']=='none'}
        complete=set(clean)==set(SEEDS) and set(full)==set(SEEDS)
        entry={'dataset':dataset,'clean_groups_complete':complete,'publication_ready':False}
        if complete:
            differences=[full[s]['final_utility']-clean[s]['final_utility'] for s in SEEDS]
            # Explicit heuristic feasibility gates, not statistical guarantees.
            learning=all(clean[s]['final_utility']>clean[s]['development_majority_accuracy']+0.05 for s in SEEDS)
            entry.update(clean_fedavg=[clean[s]['final_utility'] for s in SEEDS],
                clean_full=[full[s]['final_utility'] for s in SEEDS],paired_clean_differences=differences,
                clean_learning_passed=learning,
                clean_cost_passed=bool(np.mean(differences)>=-0.05 and min(differences)>=-0.10))
        effects={}
        for attack in ('low_and_slow','model_replacement'):
            attacked={r['seed']:r for r in subset if r['defense']=='fedavg' and r['attack']==attack}
            if set(attacked)==set(SEEDS) and set(clean)==set(SEEDS):
                values=[clean[s]['final_utility']-attacked[s]['final_utility'] if attack=='low_and_slow'
                    else attacked[s]['mean_targeted_attack_success']-clean[s]['mean_clean_trigger_prediction_rate'] for s in SEEDS]
                threshold=0.005 if attack=='low_and_slow' else 0.05
                effects[attack]={'effects':values,'numerical_gate_passed':bool(np.mean(values)>threshold)}
        entry['attack_gates']=effects;groups.append(entry)
    dump(JOB/'analysis.json',{'groups':groups,'note':'Development only. No automatic confirmation. Failed clean learning invalidates attack interpretation.'})

def run():
    JOB.mkdir(parents=True,exist_ok=True)
    state={'status':'preparing_training_only_data','pid':os.getpid(),'completed':0,'failed':0,'total':36,
           'publication_ready':False,'started_utc':datetime.now(timezone.utc).isoformat()}
    dump(JOB/'status.json',state)
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        torch.set_num_threads(2);prepare_data();rows=manifest()
        frozen={'code':hashes(),'manifest':[c.to_dict() for c in rows],
                'data':{str(p.relative_to(DATA)):digest(p) for p in sorted(DATA.rglob('provenance.json'))}}
        path=JOB/'frozen-inputs.json'
        if path.exists() and json.loads(path.read_text())!=frozen: raise RuntimeError('Frozen inputs changed')
        dump(path,frozen)
        dump(JOB/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
             'gpu':torch.cuda.get_device_name(0),'concurrent_workloads':True})
        records=[]
        for c in rows:
            if hashes()!=frozen['code']: raise RuntimeError('Frozen source changed')
            state.update(status='running',current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
            folder=DATA/f'{c.dataset}-{c.seed}';output=OUT/c.experiment_id
            try:
                if (output/'summary.json').exists():
                    s=json.loads((output/'summary.json').read_text())
                    expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
                    if s['config_hash']!=expected: raise RuntimeError('Completed config mismatch')
                    if 'development_majority_accuracy' not in s:
                        ey=np.load(folder/'development_labels.npy',allow_pickle=False)
                        s['development_majority_accuracy']=float(np.bincount(ey).max()/len(ey))
                        s['evaluation_role']='development-only training-pool holdout'
                        dump(output/'summary.json',s)
                else:
                    if output.exists() and any(output.iterdir()):
                        output.rename(output.with_name(output.name+'.interrupted-'+str(__import__('time').time_ns())))
                    x,y,ex,ey=[np.load(folder/f'{n}.npy',allow_pickle=False) for n in ('features','labels','development_features','development_labels')]
                    s=BenchmarkRunner(c,output).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes),(ex,ey))
                    s['development_majority_accuracy']=float(np.bincount(ey).max()/len(ey))
                    s['evaluation_role']='development-only training-pool holdout'
                    dump(output/'summary.json',s)
                records.append(s);state['completed']+=1;analyze(records)
            except Exception:
                state['failed']+=1
                with (JOB/'failures.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'config':c.to_dict(),'error':traceback.format_exc()})+'\n')
            dump(JOB/'status.json',state)
        state['status']='completed_with_failures' if state['failed'] else 'completed'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)

if __name__=='__main__':run()
