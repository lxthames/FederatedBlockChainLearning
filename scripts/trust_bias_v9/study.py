"""Prespecified clean-first mechanism pilot. No automatic confirmatory expansion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'2')
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.data_preparation import run_comparison_v8 as v8
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner
from .femnist import dump,digest
from .runtime import exclusive_lock

ROOT=v8.ROOT
JOB=ROOT/'artifacts/bias-v9-pilot'
OUT=ROOT/'results/bias-v9-pilot'
POLICY=ROOT/'docs/research/BIAS_V9_PILOT.md'
VARIANTS=('probe','macro_probe','margin_probe','clip_only')
SEEDS=(11,17,23)


def config(variant,dataset,seed,attack='none'):
    d=v8.config(dataset,'rank_trust',attack,seed).to_dict()
    d.update(protocol_version='trust-bias-v9',validation_variant='original' if variant in ('original','clip_only') else variant,
             experiment_id=f'v9-{dataset}-{variant}-{attack}-s{seed}')
    if variant=='clip_only':d.update(ranking_keep_fraction=1.,trust_reward=0.,rejection_trust_penalty=0.)
    if variant=='margin_probe':d.update(loss_tolerance=0.,loss_standard_errors=2.)
    return ExperimentConfig(**d)


def manifest():
    rows=[]
    for dataset in ('fashion_mnist','femnist'):
        rows.append({'method':'rank_trust','parity':True,'config':config('original',dataset,11).to_dict()})
    for dataset,seeds in (('femnist',SEEDS),('fashion_mnist',(11,))):
        for variant in VARIANTS:
            for seed in seeds:
                rows.append({'method':variant,'parity':False,'config':config(variant,dataset,seed).to_dict()})
    for attack in ('low_and_slow','model_replacement'):
        for variant in VARIANTS:
            for seed in SEEDS:
                rows.append({'method':variant,'parity':False,'config':config(variant,'femnist',seed,attack).to_dict()})
    return rows


def hashes():
    result=v8.source_hashes()
    result.update({str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))})
    return result


def reuse_cells():
    rows=list(v8.reuse_cells())
    for row in v8.manifest():
        c=v8.ExperimentConfig(**row['config']);folder=v8.OUT/c.experiment_id
        if (folder/'summary.json').exists():rows.append((row['method'],c,folder))
    return rows


def frozen_inputs():
    if v8.frozen_inputs()!=json.loads((v8.JOB/'frozen-inputs.json').read_text()):raise ValueError('V8 bindings changed')
    reused={}
    for method,c,folder in reuse_cells():
        v8.v7.load_completed(folder,c)
        for name in ('summary.json','events.jsonl','chain-trace.json','splits.json'):
            reused[str((folder/name).relative_to(ROOT))]=digest(folder/name)
    return {'source':hashes(),'data':v8.frozen_inputs()['data'],'reused':reused,
            'manifest':manifest(),'policy_sha256':digest(POLICY),
            'diagnosis_sha256':digest(JOB/'diagnosis.json'),'publication_ready':False,
            'primary_sources':{'zeno':'https://proceedings.mlr.press/v97/xie19b.html',
                               'heterogeneity':'https://proceedings.mlr.press/v206/allouah23a.html'}}


def group_gate(records,variant):
    gates=[]
    for dataset,seeds in (('femnist',SEEDS),('fashion_mnist',(11,))):
        for seed in seeds:
            gates.append({'dataset':dataset,'seed':seed,**v8.clean_gate(records,variant,seed,dataset)})
    return {'passed':all(g['passed'] for g in gates),'cells':gates,
            'rule':'All three FEMNIST clean seeds and Fashion seed 11 must pass the unchanged per-seed V6 gate.'}


def original_folder(dataset):
    if dataset=='femnist':return v8.OUT/v8.config(dataset,'rank_trust','none',11).experiment_id
    return next(p for m,c,p in v8.reuse_cells() if m=='rank_trust' and c.seed==11 and c.attack=='none')


def check_parity(folder,dataset):
    expected=original_folder(dataset)
    def models(p):return [r['model_hash'] for r in json.loads((p/'chain-trace.json').read_text()) if 'model_hash' in r]
    a=json.loads((folder/'summary.json').read_text());b=json.loads((expected/'summary.json').read_text())
    passed=(models(folder)==models(expected) and a['participants']==b['participants']
            and a['final_utility']==b['final_utility'] and a['data_fingerprint']==b['data_fingerprint'])
    if not passed:raise ValueError(f'V9 unchanged-rule parity failed for {dataset}')
    return {'dataset':dataset,'passed':passed,'source_trace_sha256':digest(expected/'chain-trace.json'),
            'new_trace_sha256':digest(folder/'chain-trace.json')}


def enrich(folder,summary,method):
    s=v8.v6.enrich(folder,summary,method)
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    rounds=[e['payload'] for e in events if e['event_type']=='round_completed']
    s['benign_update_exclusion_rate']=(sum(len(p['participants'])-len(p['accepted_client_ids']) for p in rounds)
        /sum(len(p['participants']) for p in rounds)) if s['attack']=='none' else None
    s['evaluation_role']='search-exposed training-only development holdout; not confirmatory'
    dump(folder/'summary.json',s)
    return s


def analyze(records,state):
    comparisons=[p for r in records if (p:=v8.paired_row(records,r)) is not None]
    compact=[{k:r.get(k) for k in ('dataset','method','attack','seed','final_utility','post_onset_mean_asr',
              'benign_update_exclusion_rate','honest_update_exclusion_rate','runtime_seconds_total','recovery_count')} for r in records]
    report={'status':state,'clean_gates':{v:group_gate(records,v) for v in VARIANTS},'runs':compact,
            'comparisons':comparisons,'publication_ready':False,'automatic_promotion':False,
            'limitations':'Descriptive development comparison, heuristic private validation under missing class support; no transferred theorem or recovery/penalty claim.'}
    dump(JOB/'analysis.json',report)
    lines=['# Validation-bias mechanism pilot','',report['limitations'],'',
           '| Dataset | Method | Seed | Condition | Utility | Mean targeted ASR |',
           '|---|---|---:|---|---:|---:|']
    for r in compact:
        # Main report shows candidate cells plus immediate original/FedAvg controls.
        if r['method'] not in VARIANTS+('rank_trust','fedavg'):continue
        asr=f"{100*r['post_onset_mean_asr']:.1f}%" if r['attack']=='model_replacement' else 'n/a'
        lines.append(f"| {r['dataset']} | {r['method']} | {r['seed']} | {r['attack']} | {100*r['final_utility']:.1f}% | {asr} |")
    lines+=['','Gate and missing-cell details: analysis.json. Original control rows are reused; parity runs are separate.']
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        frozen=frozen_inputs();path=JOB/'frozen-inputs.json'
        if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen V9 inputs changed')
        dump(path,frozen)
        state={'status':'starting','pid':os.getpid(),'completed':0,'failed':0,'held':0,'total_new':42,
               'parity_passed':[],'reused':len(reuse_cells()),'publication_ready':False,
               'started_utc':datetime.now(timezone.utc).isoformat()}
        records=[];dump(JOB/'status.json',state)
        try:
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            torch.set_num_threads(2)
            dump(JOB/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
                 'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0)})
            records=[v8.v7.load_completed(folder,c) for _,c,folder in reuse_cells()]
            for row in frozen['manifest']:
                if hashes()!=frozen['source'] or digest(POLICY)!=frozen['policy_sha256']:raise ValueError('Frozen source/policy changed')
                c=ExperimentConfig(**row['config']);method=row['method'];folder=OUT/c.experiment_id
                state.update(status='running',current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state)
                if not row['parity'] and c.attack!='none' and not group_gate(records,method)['passed']:
                    dump(folder/'held.json',group_gate(records,method));state['held']+=1
                    analyze(records,state);dump(JOB/'status.json',state);continue
                try:
                    if (folder/'summary.json').exists():s=v8.v7.load_completed(folder,c)
                    else:
                        if folder.exists() and any(folder.iterdir()):folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                        datafolder=v8.v7.DATA/f'{c.dataset}-{c.seed}'
                        for p,h in frozen['data'].items():
                            if Path(p).parent==datafolder.relative_to(ROOT) and digest(ROOT/p)!=h:raise ValueError('Data changed')
                        x,y,ex,ey=[np.load(datafolder/f'{n}.npy',allow_pickle=False) for n in
                                   ('features','labels','development_features','development_labels')]
                        s=BenchmarkRunner(c,folder).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes),(ex,ey))
                    s=enrich(folder,s,method)
                    if row['parity']:
                        state['parity_passed'].append(check_parity(folder,c.dataset))
                    else:records.append(s)
                    state['completed']+=1
                except Exception:
                    state['failed']+=1
                    with (JOB/'failures.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                    if row['parity']:raise
                analyze(records,state);dump(JOB/'status.json',state)
            state['status']='completed_with_gaps' if state['failed'] or state['held'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());raise
        finally:
            state['finished_utc']=datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json',state);analyze(records,state)


if __name__=='__main__':run()
