"""Finite clean-first pilot: at most 96 cells, no automatic confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'2')

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .pairing import compare_pair
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner
from .runtime import exclusive_lock

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/trust-revision-v3-development'
JOB=ROOT/'artifacts/screen-v11'
OUT=ROOT/'results/screen-v11'
POLICY=ROOT/'docs/research/SCREEN_V11.md'
AUDIT=ROOT/'docs/research/TRUST_EXTENSION_BASELINE_AUDIT.md'
SEEDS=(11,17,23)
DATASETS=('fashion_mnist','femnist')
CONTROLS=('fedavg','clip_only','fedval')
CANDIDATES=('rank_no_penalty','scaled_quorum')
CLEAN_ARMS=CONTROLS+('rank_original',)+CANDIDATES
ATTACKS=('low_and_slow','model_replacement')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    temp.replace(path)


def config(dataset,arm,seed,attack='none'):
    if dataset not in DATASETS or arm not in CLEAN_ARMS or attack not in ('none',)+ATTACKS:
        raise ValueError('Unknown pilot cell')
    if arm=='rank_original' and attack!='none':
        raise ValueError('Original rank is a clean diagnostic only; prior held attacks remain held')
    femnist=dataset=='femnist'
    dynamic=arm in {'rank_original','rank_no_penalty'}
    return ExperimentConfig(experiment_id=f'v11-{dataset}-{arm}-{attack}-s{seed}',
        dataset=dataset,num_clients=50,participation_rate=.4,malicious_fraction=.3,
        dirichlet_alpha=None if femnist else .5,seed=seed,rounds=40 if femnist else 30,
        attack=attack,attack_start_round=11,num_classes=62 if femnist else 10,
        batch_size=16 if femnist else 64,learning_rate=.05 if femnist else .01,
        optimizer_momentum=.9,device='cuda',sampling_policy='bounded',
        defense=arm if arm in {'fedavg','fedval'} else 'two_layer',
        vote_rule='scaled_loss' if arm=='scaled_quorum' else 'rank_loss',
        ablation='validation_only',candidate_clip_ratio=2.,
        natural_splits_path=str(DATA/f'{dataset}-{seed}/natural-splits.json') if femnist else None,
        validation_distribution='private',
        trust_updates='dynamic' if dynamic else 'fixed',
        trust_weighting='dynamic' if dynamic else 'fixed',
        trust_eligibility='dynamic' if dynamic else 'fixed',
        trust_reward=.02 if dynamic else 0.,
        rejection_trust_penalty=.02 if arm=='rank_original' else 0.,
        rejection_trust_policy='rank_rejection' if arm=='rank_original' else 'none',
        ranking_keep_fraction=1. if arm=='clip_only' else .8,
        loss_tolerance=0.,loss_standard_errors=2.)


def manifest():
    # All 36 clean cells precede any attack. Controls run regardless of clean
    # gate, with failure labels; candidate attacks require all six clean gates.
    rows=[{'arm':a,'config':config(d,a,s).to_dict()} for d in DATASETS for s in SEEDS for a in CLEAN_ARMS]
    rows += [{'arm':a,'config':config(d,a,s,t).to_dict()} for t in ATTACKS
             for d in DATASETS for s in SEEDS for a in CONTROLS+CANDIDATES]
    return rows


def inputs():
    data={}
    for d in DATASETS:
        for seed in SEEDS:
            folder=DATA/f'{d}-{seed}'
            provenance=json.loads((folder/'provenance.json').read_text())
            names=['features.npy','labels.npy','development_features.npy','development_labels.npy']
            if d=='femnist':names.append('natural-splits.json')
            for name in names:
                actual=digest(folder/name)
                if provenance['files'].get(name)!=actual:raise ValueError(f'Data changed: {folder/name}')
                data[str((folder/name).relative_to(ROOT))]=actual
            data[str((folder/'provenance.json').relative_to(ROOT))]=digest(folder/'provenance.json')
    return {'source':{str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
        'data':data,'manifest':manifest(),'policy_sha256':digest(POLICY),'baseline_audit_sha256':digest(AUDIT),
        'evaluation_role':'search-exposed development follow-up','automatic_confirmation':False,'publication_ready':False}


def freeze():
    value=inputs();path=JOB/'frozen-inputs.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen V11 changed')
    dump(path,value)
    return value


FILES=('config.json','summary.json','splits.json','events.jsonl','chain-trace.json')


def seal(folder):
    dump(folder/'completion.json',{name:digest(folder/name) for name in FILES})


def load_completed(folder,c):
    if json.loads((folder/'completion.json').read_text())!={name:digest(folder/name) for name in FILES}:
        raise ValueError('Completed artifact digest mismatch')
    summary=json.loads((folder/'summary.json').read_text())
    expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
    if summary.get('config_hash')!=expected or summary.get('status')!='completed':
        raise ValueError('Completed configuration/status mismatch')
    return summary


def clean_cell(records,arm,dataset,seed):
    find=lambda method:next((r for r in records if r['arm']==method and r['dataset']==dataset
                            and r['seed']==seed and r['attack']=='none'),None)
    b,r=find('fedavg'),find(arm)
    if b is None or r is None:return {'passed':False,'reason':'missing clean cell'}
    pair=compare_pair(r,b)
    delta=r['final_utility']-b['final_utility']
    return {'passed':bool(pair['matched'] and b['final_utility']>=.2 and delta>=-.05
                         and r['unsupported_rounds']<=3 and r['honest_validator_penalty_amount']==0),
            'utility_difference':delta,'pairing':pair,'seed':seed,'dataset':dataset}


def candidate_gate(records,arm):
    cells=[clean_cell(records,arm,d,s) for d in DATASETS for s in SEEDS]
    return {'passed':all(r['passed'] for r in cells),'cells':cells,
            'rule':'All six matched clean cells: baseline >=20%, candidate loss <=5 points, unsupported <=3, no honest monetary penalty.'}


def enrich(summary):
    # Summaries stay immutable; coordinator annotations live in analysis only.
    c=summary['configuration'];onset=c['attack_start_round']
    summary=dict(summary)
    post_asr=summary['targeted_attack_success_history'][onset-1:]
    summary['post_onset_mean_asr']=float(np.mean(post_asr)) if summary['attack']=='model_replacement' and post_asr else None
    clean=summary.get('clean_trigger_prediction_history',[])
    post_clean=clean[onset-1:]
    summary['post_onset_mean_clean_trigger_rate']=float(np.mean(post_clean)) if post_clean else None
    return summary


def analyze(records,state):
    comparisons=[]
    for r in records:
        b=next((b for b in records if b['arm']=='fedavg' and b['dataset']==r['dataset']
                and b['seed']==r['seed'] and b['attack']==r['attack']),None)
        if b is None:continue
        clean=next((b for b in records if b['arm']=='fedavg' and b['dataset']==r['dataset']
                    and b['seed']==r['seed'] and b['attack']=='none'),None)
        valid=None
        if clean and r['attack']=='low_and_slow':valid=clean['final_utility']-b['final_utility']>=.01
        if clean and r['attack']=='model_replacement':
            valid=b['post_onset_mean_asr']-clean['post_onset_mean_clean_trigger_rate']>=.10
        pair=compare_pair(r,b)
        comparisons.append({'arm':r['arm'],'dataset':r['dataset'],'seed':r['seed'],'attack':r['attack'],
            **pair,'clean_gate':clean_cell(records,r['arm'],r['dataset'],r['seed']),
            'attack_effective_vs_fedavg':valid,
            'utility_difference':r['final_utility']-b['final_utility'] if pair['matched'] else None,
            'asr_reduction':b['post_onset_mean_asr']-r['post_onset_mean_asr'] if pair['matched'] and r['attack']=='model_replacement' else None})
    groups=[]
    for d in DATASETS:
        for arm in CLEAN_ARMS:
            for attack in ('none',)+ATTACKS:
                if arm=='rank_original' and attack!='none':continue
                rows=[r for r in records if r['dataset']==d and r['arm']==arm and r['attack']==attack]
                matched=[r for r in comparisons if r['dataset']==d and r['arm']==arm and r['attack']==attack]
                complete=sorted(r['seed'] for r in rows)==list(SEEDS) and len(matched)==3 and all(r['matched'] for r in matched)
                groups.append({'dataset':d,'arm':arm,'attack':attack,'complete':complete,'seeds':[r['seed'] for r in rows],
                    'mean_utility':float(np.mean([r['final_utility'] for r in rows])) if complete else None,
                    'mean_asr':float(np.mean([r['post_onset_mean_asr'] for r in rows])) if complete and attack=='model_replacement' else None,
                    'mean_benign_exclusion':float(np.mean([r['benign_update_exclusion_rate'] for r in rows])) if complete and attack=='none' else None})
    keys=('arm','dataset','seed','attack','final_utility','post_onset_mean_asr','lower_decile_client_utility',
          'benign_update_exclusion_rate','unsupported_rounds','inconclusive_counts','quality_rejection_counts',
          'runtime_seconds_total','validation_work','trusted_server_samples')
    report={'status':state,'runs':[{k:r[k] for k in keys} for r in records],
        'candidate_gates':{a:candidate_gate(records,a) for a in CANDIDATES},'comparisons':comparisons,'groups':groups,
        'publication_ready':False,'automatic_promotion':False,
        'limitations':['Three search-exposed seeds; descriptive development, no significance or novelty claim.',
                       'FedVal is paper-based and has central class-complete reference access and a different evaluation budget.',
                       'Paired-loss tolerance is a fixed heuristic under correlated reused validation data, not a confidence guarantee.',
                       'Held cells are explicit; no threshold changes or automatic confirmation.']}
    dump(JOB/'analysis.json',report)
    lines=['# V11 clean-first screening pilot','',
        '| Dataset | Arm | Condition | Seeds | Mean utility | Mean targeted ASR |',
        '|---|---|---|---|---:|---:|']
    for g in groups:
        utility=f"{g['mean_utility']*100:.2f}%" if g['complete'] else 'incomplete / held'
        asr=f"{g['mean_asr']*100:.2f}%" if g['mean_asr'] is not None else 'n/a'
        lines.append(f"| {g['dataset']} | {g['arm']} | {g['attack']} | {g['seeds']} | {utility} | {asr} |")
    lines+=['']+report['limitations']
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        path=JOB/'frozen-inputs.json'
        if not path.exists():raise ValueError('Freeze V11 first')
        frozen=json.loads(path.read_text())
        if inputs()!=frozen:raise ValueError('Frozen inputs changed')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(2)
        dump(JOB/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
            'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0)})
        state={'status':'running','pid':os.getpid(),'completed':0,'failed':0,'held':0,
               'total':len(frozen['manifest']),'publication_ready':False,'started_utc':datetime.now(timezone.utc).isoformat()}
        records=[];dump(JOB/'status.json',state)
        try:
            for row in frozen['manifest']:
                if inputs()!=frozen:raise ValueError('Frozen source/data/policy changed')
                c=ExperimentConfig(**row['config']);arm=row['arm'];folder=OUT/c.experiment_id
                state['current']=c.experiment_id;dump(JOB/'status.json',state)
                if c.attack!='none' and arm in CANDIDATES and not candidate_gate(records,arm)['passed']:
                    dump(folder/'held.json',candidate_gate(records,arm));state['held']+=1
                    analyze(records,state);dump(JOB/'status.json',state);continue
                try:
                    if (folder/'completion.json').exists():summary=load_completed(folder,c)
                    else:
                        if folder.exists():
                            target=folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                            if folder.resolve().parent!=OUT.resolve() or target.resolve().parent!=OUT.resolve():raise ValueError('Result path escapes V11 root')
                            folder.rename(target)
                        data=DATA/f'{c.dataset}-{c.seed}'
                        x,y,ex,ey=[np.load(data/f'{n}.npy',allow_pickle=False) for n in
                                  ('features','labels','development_features','development_labels')]
                        summary=BenchmarkRunner(c,folder).run_arrays(x,y,
                            lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(ex,ey))
                        if inputs()!=frozen:raise ValueError('Inputs changed during cell')
                        seal(folder)
                    records.append(dict(enrich(summary),arm=arm));state['completed']+=1
                except Exception:
                    state['failed']+=1
                    with (JOB/'failures.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                analyze(records,state);dump(JOB/'status.json',state)
            state['status']='completed_with_gaps' if state['failed'] or state['held'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());raise
        finally:
            state['finished_utc']=datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json',state);analyze(records,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.freeze:
        freeze();print(f'Frozen {len(manifest())} maximum V11 cells; training not started.')
    else:print(json.dumps(manifest(),indent=2))
