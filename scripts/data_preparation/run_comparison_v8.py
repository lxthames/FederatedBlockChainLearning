"""Fixed baseline comparison and natural-client transfer, using frozen V6 code."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'2')
from dataclasses import replace
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.data_preparation import run_rank_replication_v7 as v7
from scripts.trust_literature_v6 import study as v6
from scripts.trust_literature_v6.femnist import dump,digest
from scripts.trust_literature_v6.runtime import exclusive_lock
from scripts.trust_literature_v6.research.config import ExperimentConfig
from scripts.trust_literature_v6.research.models import create_model
from scripts.trust_literature_v6.research.runner_v2 import BenchmarkRunner

ROOT=v7.ROOT
JOB=ROOT/'artifacts/comparison-v8'
OUT=ROOT/'results/comparison-v8'
POLICY=ROOT/'docs/research/COMPARISON_V8.md'
AUDIT=ROOT/'docs/research/TRUST_EXTENSION_BASELINE_AUDIT.md'
SEEDS=(11,17,23)
ATTACKS=('none','low_and_slow','model_replacement')
BASELINES=('coordinate_median','fltrust','fedval')


def config(dataset,method,attack,seed):
    c=replace(v6.config(method,attack),seed=seed,archive_updates=False,
              experiment_id=f'v8-{dataset}-{method}-{attack}-s{seed}')
    if dataset=='femnist':
        c=replace(c,dataset=dataset,num_classes=62,rounds=40,batch_size=16,
                  learning_rate=.05,dirichlet_alpha=None,
                  natural_splits_path=str(v7.DATA/f'femnist-{seed}/natural-splits.json'))
    return c


def manifest():
    rows=[]
    for attack in ATTACKS:
        for seed in SEEDS:
            for method in ('fedavg','rank_trust'):
                rows.append({'method':method,'config':config('femnist',method,attack,seed).to_dict()})
    for attack in ATTACKS:
        for seed in SEEDS:
            for method in BASELINES:
                if (method,attack,seed)==('coordinate_median','none',11):continue
                rows.append({'method':method,'config':config('fashion_mnist',method,attack,seed).to_dict()})
    return rows


def source_hashes():
    return dict(v7.source_hashes(),**{str(Path(__file__).relative_to(ROOT)):digest(__file__)})


def reuse_cells():
    rows=[]
    for attack in ATTACKS:
        for seed in SEEDS:
            for method in ('fedavg','rank_trust'):
                if seed==11:
                    c=v6.config(method,attack);folder=v6.OUT/c.experiment_id
                else:
                    row=next(r for r in v7.manifest() if r['method']==method and
                             r['config']['seed']==seed and r['config']['attack']==attack)
                    c=ExperimentConfig(**row['config']);folder=v7.OUT/c.experiment_id
                rows.append((method,c,folder))
    row=v7.manifest()[-1];c=ExperimentConfig(**row['config'])
    rows.append(('coordinate_median',c,v7.OUT/c.experiment_id))
    return rows


def frozen_inputs():
    # Prove that all reused experiments still have their original bindings.
    if v7.frozen_inputs()!=json.loads((v7.JOB/'frozen-inputs.json').read_text()):
        raise ValueError('V7 binding changed')
    data={}
    for dataset in ('fashion_mnist','femnist'):
        for seed in SEEDS:
            folder=v7.DATA/f'{dataset}-{seed}'
            p=json.loads((folder/'provenance.json').read_text())
            for name,h in p['files'].items():
                if digest(folder/name)!=h:raise ValueError('Development data changed')
                data[str((folder/name).relative_to(ROOT))]=h
            data[str((folder/'provenance.json').relative_to(ROOT))]=digest(folder/'provenance.json')
    reused={}
    for method,c,folder in reuse_cells():
        v7.load_completed(folder,c)
        for name in ('summary.json','chain-trace.json','events.jsonl','splits.json'):
            reused[str((folder/name).relative_to(ROOT))]=digest(folder/name)
    return {'source':source_hashes(),'data':data,'reused':reused,'manifest':manifest(),
            'policy_sha256':digest(POLICY),'baseline_audit_sha256':digest(AUDIT),
            'automatic_confirmation':False,'publication_ready':False}


def clean_gate(records,method,seed,dataset):
    return v6.clean_gate([r for r in records if r['seed']==seed and r['dataset']==dataset],method)


def paired_row(records,r):
    base=next((b for b in records if b['method']=='fedavg' and b['dataset']==r['dataset']
               and b['seed']==r['seed'] and b['attack']==r['attack']),None)
    clean=next((b for b in records if b['method']=='fedavg' and b['dataset']==r['dataset']
                and b['seed']==r['seed'] and b['attack']=='none'),None)
    if base is None:return None
    matched=all(r[k]==base[k] for k in ('participants','data_fingerprint','test_fingerprint'))
    valid=None
    if clean and r['attack']=='low_and_slow':valid=clean['final_utility']-base['final_utility']>=.01
    if clean and r['attack']=='model_replacement':
        valid=base['post_onset_mean_asr']-clean['post_onset_mean_clean_trigger_rate']>=.10
    return {'dataset':r['dataset'],'method':r['method'],'attack':r['attack'],'seed':r['seed'],
            'matched':matched,'effective_attack_vs_fedavg':bool(valid) if valid is not None else None,
            'utility_effect_vs_fedavg':r['final_utility']-base['final_utility'],
            'asr_reduction_vs_fedavg':base['post_onset_mean_asr']-r['post_onset_mean_asr']
                if r['attack']=='model_replacement' else None,
            'clean_gate':clean_gate(records,r['method'],r['seed'],r['dataset']),
            'honest_exclusion':r['honest_update_exclusion_rate']}


def analyze(records,state):
    comparisons=[p for r in records if (p:=paired_row(records,r)) is not None]
    groups=[]
    for dataset,methods in (('fashion_mnist',('fedavg','rank_trust')+BASELINES),
                           ('femnist',('fedavg','rank_trust'))):
        for attack in ATTACKS:
            for method in methods:
                rows=[r for r in records if r['dataset']==dataset and r['attack']==attack and r['method']==method]
                complete=sorted(r['seed'] for r in rows)==list(SEEDS)
                group={'dataset':dataset,'attack':attack,'method':method,'seeds':[r['seed'] for r in rows],
                       'complete':complete,'mean_final_utility':None,'mean_post_onset_asr':None}
                if complete:
                    group['mean_final_utility']=float(np.mean([r['final_utility'] for r in rows]))
                    if attack=='model_replacement':group['mean_post_onset_asr']=float(np.mean([r['post_onset_mean_asr'] for r in rows]))
                groups.append(group)
    report={'status':state,'paired_comparisons':comparisons,'groups':groups,'publication_ready':False,
            'automatic_promotion':False,'limitations':[
                'Development seeds, training-only holdouts, and differing dataset-specific horizons/hyperparameters.',
                'FedVal is the audited paper-based implementation, not verified author-code equivalence.',
                'Reference-data access differs from private-committee access; no equal-validation-budget claim.',
                'Baseline attack cells run regardless of clean utility; the unchanged gate holds new FEMNIST rank attack cells.',
                'No recovery benefit, ledger deployment, CIFAR-10 confirmation, or independent validator-corruption claim.']}
    dump(JOB/'analysis.json',report)
    lines=['# Matched baseline and transfer development comparison','',
           '| Dataset | Method | Condition | Seeds | Mean final utility | Mean targeted ASR |',
           '|---|---|---|---|---:|---:|']
    for g in groups:
        utility=f"{100*g['mean_final_utility']:.1f}%" if g['complete'] else 'incomplete'
        asr=f"{100*g['mean_post_onset_asr']:.1f}%" if g['mean_post_onset_asr'] is not None else 'n/a'
        lines.append(f"| {g['dataset']} | {g['method']} | {g['attack']} | {g['seeds']} | {utility} | {asr} |")
    lines+=['']+report['limitations']
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        frozen=frozen_inputs();path=JOB/'frozen-inputs.json'
        if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen V8 inputs changed')
        dump(path,frozen)
        state={'status':'starting','pid':os.getpid(),'completed':0,'failed':0,'held':0,
               'total_new':len(frozen['manifest']),'reused':19,'publication_ready':False,
               'started_utc':datetime.now(timezone.utc).isoformat()}
        records=[];dump(JOB/'status.json',state)
        try:
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            torch.set_num_threads(2)
            dump(JOB/'environment.json',{'gpu':torch.cuda.get_device_name(0),'python':platform.python_version(),
                 'versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','scikit-learn')}})
            for method,c,folder in reuse_cells():records.append(v7.load_completed(folder,c))
            for row in frozen['manifest']:
                if source_hashes()!=frozen['source'] or digest(POLICY)!=frozen['policy_sha256'] or digest(AUDIT)!=frozen['baseline_audit_sha256']:
                    raise ValueError('Frozen source/policy/audit changed')
                method=row['method'];c=ExperimentConfig(**row['config']);folder=OUT/c.experiment_id
                state.update(status='running',current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state)
                gate=clean_gate(records,method,c.seed,c.dataset)
                if c.dataset=='femnist' and method=='rank_trust' and c.attack!='none' and not gate['passed']:
                    state['held']+=1;dump(folder/'held.json',gate)
                    analyze(records,state);dump(JOB/'status.json',state);continue
                try:
                    if (folder/'summary.json').exists():s=v7.load_completed(folder,c)
                    else:
                        if folder.exists() and any(folder.iterdir()):
                            folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                        datafolder=v7.DATA/f'{c.dataset}-{c.seed}'
                        for p,h in frozen['data'].items():
                            if Path(p).parent==datafolder.relative_to(ROOT) and digest(ROOT/p)!=h:raise ValueError('Data changed')
                        x,y,ex,ey=[np.load(datafolder/f'{n}.npy',allow_pickle=False) for n in
                                   ('features','labels','development_features','development_labels')]
                        s=BenchmarkRunner(c,folder).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes),(ex,ey))
                    records.append(v6.enrich(folder,s,method));state['completed']+=1
                except Exception:
                    state['failed']+=1
                    with (JOB/'failures.jsonl').open('a',encoding='utf-8') as handle:
                        handle.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                analyze(records,state);dump(JOB/'status.json',state)
            state['status']='completed_with_gaps' if state['failed'] or state['held'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());raise
        finally:
            state['finished_utc']=datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json',state);analyze(records,state)


if __name__=='__main__':run()
