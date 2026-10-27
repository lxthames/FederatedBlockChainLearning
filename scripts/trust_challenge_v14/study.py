"""Finite clean-calibrated V14 pilot; four mechanism arms and clipping control."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(name,'2')
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.trust_security_v13 import study as parent
from .challenges import QUANTILE, ARMS
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner
from .runtime import exclusive_lock

ROOT=Path(__file__).resolve().parents[2]
DATA=parent.DATA
OUT=ROOT/'results/challenge-v14'
JOB=ROOT/'artifacts/challenge-v14'
POLICY=ROOT/'docs/research/CHALLENGE_V14.md'
SEEDS=(11,17,23)
DATASETS=('fashion_mnist','femnist')
FILES=parent.FILES+('challenge-settings.json',)
read=parent.read
dump=parent.dump
digest=parent.digest


def config(dataset,arm,seed,attack='none'):
    if arm not in ARMS or dataset not in DATASETS or seed not in SEEDS or attack not in ('none','model_replacement'):
        raise ValueError('unknown V14 cell')
    original=parent.config(dataset,'clip_only' if arm=='clip_only' else parent.CANDIDATE,seed,attack).to_dict()
    original.update(experiment_id=f'v14-{dataset}-{arm}-{attack}-s{seed}',protocol_version='trust-challenge-v14')
    return ExperimentConfig(**original)


def manifest():
    # Six instrumented clipping cells supply clean-only calibration and the control.
    cells=[(d,'clip_only',s,'none') for d in DATASETS for s in SEEDS]
    cells += [(d,a,s,'none') for d in DATASETS for s in SEEDS for a in ARMS if a!='clip_only']
    cells += [(d,a,s,'model_replacement') for d in DATASETS for s in SEEDS for a in ARMS]
    return [dict(dataset=d,arm=a,seed=s,attack=t,config=config(d,a,s,t).to_dict()) for d,a,s,t in cells]


def inputs():
    if parent.inputs()!=read(parent.JOB/'frozen-inputs.json'):raise ValueError('frozen V13 source/data changed')
    return dict(source={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
        parent_freeze_sha256=digest(parent.JOB/'frozen-inputs.json'),policy_sha256=digest(POLICY),
        manifest=manifest(),quantile=QUANTILE,threshold_source='clean clipping validation telemetry only',
        automatic_confirmation=False,publication_ready=False)


def freeze():
    value=inputs();path=JOB/'frozen-inputs.json'
    if path.exists() and read(path)!=value:raise ValueError('frozen V14 changed')
    dump(path,value)


def verify_cell(folder,c,settings):
    if read(folder/'completion.json')!={n:digest(folder/n) for n in FILES}:raise ValueError('V14 completion seal changed')
    summary=read(folder/'summary.json')
    expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
    if summary['status']!='completed' or summary['config_hash']!=expected:raise ValueError('config/status mismatch')
    if read(folder/'challenge-settings.json')!=settings:raise ValueError('challenge binding mismatch')
    if read(folder/'diagnostic-inputs.json')!=parent.EMPTY_BINDING or summary['diagnostic_input_binding']!=parent.EMPTY_BINDING:
        raise ValueError('unexpected mass schedule')
    return summary


def calibrate():
    thresholds={};bindings={}
    for d in DATASETS:
        scores={'candidate':[],'aggregate':[]}
        for s in SEEDS:
            c=config(d,'clip_only',s);folder=OUT/c.experiment_id
            verify_cell(folder,c,dict(arm='calibrate',thresholds=None,calibration_sha256=None))
            bindings[c.experiment_id]=digest(folder/'completion.json')
            for line in (folder/'events.jsonl').read_text().splitlines():
                e=json.loads(line)
                if e['event_type']=='challenge_vote':
                    p=e['payload'];scores[p['stage']].append(p['score'])
        if any(not values for values in scores.values()):raise ValueError('incomplete calibration telemetry')
        thresholds[d]={stage:float(np.quantile(values,QUANTILE,method='higher')) for stage,values in scores.items()}
    value=dict(thresholds=thresholds,source_completion_digests=bindings,quantile=QUANTILE,
        rule='higher empirical 99.5th percentile by dataset and candidate/aggregate stage; no independence/FPR guarantee')
    path=JOB/'calibration.json'
    if path.exists() and read(path)!=value:raise ValueError('sealed calibration changed')
    dump(path,value)
    return value


def gate(records,arm):
    cells=[]
    for d in DATASETS:
        for s in SEEDS:
            clean=next((r for r in records if (r['dataset'],r['seed'],r['arm'],r['attack'])==(d,s,arm,'none')),None)
            base=next((r for r in records if (r['dataset'],r['seed'],r['arm'],r['attack'])==(d,s,'clip_only','none')),None)
            ok=bool(clean and base and parent.method_pair(clean,base)['matched'] and base['final_utility']>=.2
                and clean['final_utility']>=base['final_utility']-.05 and clean['unsupported_rounds']<=3
                and clean['honest_validator_penalty_amount']==0)
            cells.append(dict(dataset=d,seed=s,passed=ok))
    return dict(passed=all(c['passed'] for c in cells),cells=cells)


def analyze(records,state):
    groups=[];condition_pairs=[]
    for r in records:
        if r['attack']=='none':continue
        clean=next((a for a in records if a['dataset']==r['dataset'] and a['seed']==r['seed'] and a['arm']==r['arm'] and a['attack']=='none'),None)
        condition_pairs.append(dict(experiment=r['experiment_id'],**parent.condition_pair(clean,r)))
    for d in DATASETS:
        for a in ARMS:
            for t in ('none','model_replacement'):
                cells=[r for r in records if (r['dataset'],r['arm'],r['attack'])==(d,a,t)]
                complete=sorted(r['seed'] for r in cells)==list(SEEDS)
                if t!='none':complete=complete and all(next(p for p in condition_pairs if p['experiment']==r['experiment_id'])['matched'] for r in cells)
                avg=lambda key:float(np.mean([r[key] for r in cells])) if complete else None
                groups.append(dict(dataset=d,arm=a,attack=t,complete=complete,mean_accuracy=avg('final_utility'),
                    mean_asr=avg('post_onset_mean_asr') if t!='none' else None,
                    endpoint_asr=avg('endpoint_asr') if t!='none' else None,
                    honest_exclusion=avg('honest_update_exclusion_rate'),unsupported_rounds=avg('unsupported_rounds'),
                    runtime_seconds=avg('runtime_seconds_total')))
    compact=[{k:r[k] for k in ('experiment_id','dataset','arm','seed','attack','final_utility','post_onset_mean_asr',
        'endpoint_asr','honest_update_exclusion_rate','malicious_update_acceptance_rate','unsupported_rounds',
        'validation_work','runtime_seconds_total')} for r in records]
    dump(JOB/'analysis.json',dict(status=state,groups=groups,runs=compact,condition_pairs=condition_pairs,
        gates={a:gate(records,a) for a in ARMS},publication_ready=False,automatic_promotion=False))
    lines=['# V14 challenge pilot','','All results are exposed-seed development outcomes. Partial groups have no means.',
        'Attack efficacy and superiority require separate assessment; no automatic promotion.', '',
        '| Dataset | Arm | Attack | Accuracy | Mean ASR | Endpoint ASR |','|---|---|---|---:|---:|---:|']
    fmt=lambda x:'pending/held' if x is None else f'{100*x:.2f}%'
    for g in groups:lines.append(f"| {g['dataset']} | {g['arm']} | {g['attack']} | {fmt(g['mean_accuracy'])} | {fmt(g['mean_asr'])} | {fmt(g['endpoint_asr'])} |")
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        frozen=read(JOB/'frozen-inputs.json')
        if inputs()!=frozen:raise ValueError('frozen inputs changed')
        if read(parent.JOB/'status.json')['status']!='completed':raise ValueError('V13 incomplete')
        with exclusive_lock(parent.JOB/'coordinator.lock'):pass
        for row in read(parent.JOB/'frozen-inputs.json')['manifest']:
            parent.load_completed(parent.OUT/row['config']['experiment_id'],parent.ExperimentConfig(**row['config']))
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(2)
        dump(JOB/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        state=dict(status='running',pid=os.getpid(),total=len(manifest()),completed=0,held=0,failed=0,
            started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        records=[];calibration=None;calibration_digest=None;dump(JOB/'status.json',state)
        try:
            for idx,row in enumerate(frozen['manifest']):
                if inputs()!=frozen:raise ValueError('frozen inputs changed during pilot')
                c=ExperimentConfig(**row['config']);folder=OUT/c.experiment_id
                state['current']=c.experiment_id;dump(JOB/'status.json',state)
                if idx>=6:
                    if calibration is None:
                        calibration=calibrate();calibration_digest=digest(JOB/'calibration.json')
                    elif digest(JOB/'calibration.json')!=calibration_digest:
                        raise ValueError('calibration changed during pilot')
                    thresholds=calibration['thresholds'][c.dataset]
                else:thresholds=None;calibration_digest=None
                arm='calibrate' if idx<6 else row['arm']
                settings=dict(arm=arm,thresholds=thresholds,calibration_sha256=calibration_digest)
                if c.attack!='none' and row['arm']!='clip_only' and not gate(records,row['arm'])['passed']:
                    dump(folder/'held.json',gate(records,row['arm']));state['held']+=1
                    analyze(records,state);dump(JOB/'status.json',state);continue
                try:
                    if (folder/'completion.json').exists():summary=verify_cell(folder,c,settings)
                    else:
                        if folder.exists():
                            target=folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                            if folder.resolve().parent!=OUT.resolve() or target.resolve().parent!=OUT.resolve():raise ValueError('invalid result path')
                            folder.rename(target)
                        runner=BenchmarkRunner(c,folder);runner.challenge_arm=arm;runner.challenge_thresholds=thresholds
                        dump(folder/'diagnostic-inputs.json',parent.EMPTY_BINDING);dump(folder/'challenge-settings.json',settings)
                        data=DATA/f'{c.dataset}-{c.seed}'
                        x,y,ex,ey=[np.load(data/f'{n}.npy',allow_pickle=False) for n in
                            ('features','labels','development_features','development_labels')]
                        summary=runner.run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(ex,ey))
                        if inputs()!=frozen:raise ValueError('inputs changed during cell')
                        dump(folder/'completion.json',{n:digest(folder/n) for n in FILES})
                        verify_cell(folder,c,settings)
                    # Calibration telemetry must be numerically passive on original clipping control.
                    if idx<6:
                        old=read(parent.OUT/parent.config(c.dataset,'clip_only',c.seed).experiment_id/'chain-trace.json')
                        new=read(folder/'chain-trace.json')
                        if [r['model_hash'] for r in old]!=[r['model_hash'] for r in new]:raise ValueError('calibration changed control trajectory')
                    records.append(dict(parent.enrich(summary,folder),arm=row['arm']));state['completed']+=1
                except Exception:
                    state['failed']+=1
                    dump(JOB/f'failure-{c.experiment_id}.json',dict(error=traceback.format_exc()))
                    if idx<6:raise RuntimeError('clean calibration failed')
                analyze(records,state);dump(JOB/'status.json',state)
            state['status']='completed_with_gaps' if state['failed'] or state['held'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());raise
        finally:
            state['finished_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state);analyze(records,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze();print(f'Frozen {len(manifest())} maximum cells')
    elif args.run:run()
    else:print(json.dumps(manifest(),indent=2))
